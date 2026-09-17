"""
Link import jobs.

Downloads a YouTube video with yt-dlp into the container's ephemeral /tmp,
checks it against the editor's limits, pushes the MP4 to private storage with
a one-time signed upload URL and then deletes every temp file it made. It uses
the same saved YouTube session as the clipping worker when one is configured,
and falls back to cookie-free clients otherwise. Nothing survives a restart.
"""

from __future__ import annotations

import base64
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from urllib.parse import urlsplit

import requests


class _VidKrakenError(Exception):
    """A VidKraken failure that permits the existing yt-dlp fallback."""


# VidKraken has renamed its length field before, so try the most specific names
# first and fall back to anything that merely looks like a duration.
_DURATION_TIERS = (
    ("durationseconds", "lengthseconds", "durationinseconds"),
    ("duration", "length"),
)


class _VidKrakenClient:
    """Self-contained VidKraken client.

    This intentionally lives in importer.py because older Railway build
    recipes copied only importer.py and worker.py into the image. Keeping the
    client here prevents a missing helper file from crashing the container.
    """

    VidKrakenError = _VidKrakenError
    BASE = os.environ.get("VIDKRAKEN_API_URL", "https://vidkraken.com/api/v2").rstrip("/")
    QUALITY = os.environ.get("VIDKRAKEN_FORMAT", "1080")
    POLL_SECONDS = float(os.environ.get("VIDKRAKEN_POLL_SECONDS", 3))
    JOB_TIMEOUT = float(os.environ.get("VIDKRAKEN_TIMEOUT_SECONDS", 1800))
    DONE = {"COMPLETED", "COMPLETE", "SUCCESS", "SUCCEEDED", "DONE", "READY", "FINISHED"}
    BROKEN = {"FAILED", "ERROR", "CANCELLED", "CANCELED"}

    @staticmethod
    def key() -> str:
        return os.environ.get("VIDKRAKEN_API_KEY", "").strip()

    def enabled(self) -> bool:
        return bool(self.key())

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key()}", "Content-Type": "application/json"}

    def _download_headers(self) -> dict:
        # VidKraken requires the API key on every request, including the
        # proxy.vidkraken.com CDN URL returned by a completed download job.
        return {"Authorization": f"Bearer {self.key()}", "Accept": "*/*"}

    @staticmethod
    def _message(payload: dict, fallback: str) -> str:
        for field in ("error", "message", "errorMessage", "failureReason"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return fallback

    # -- response field discovery ------------------------------------------
    # VidKraken describes its payloads in prose rather than a pinned schema and
    # has renamed fields before, so the client walks the whole response for the
    # values it needs. A renamed key then costs a log line, not a broken job.

    @staticmethod
    def _walk(payload, prefix: str = ""):
        if isinstance(payload, dict):
            for key, value in payload.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield path, str(key).lower(), value
                yield from _VidKrakenClient._walk(value, path)
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                path = f"{prefix}[{index}]"
                yield path, prefix.lower(), value
                yield from _VidKrakenClient._walk(value, path)

    @staticmethod
    def _number(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if text and text.count(".") < 2 and all(ch in "0123456789." for ch in text):
                try:
                    return float(text)
                except ValueError:
                    return None
        return None

    @classmethod
    def _find_number(cls, payload, tiers) -> float:
        """First number found under a key matching each tier, in order."""
        for names in tiers:
            best = 0.0
            for _path, key, value in cls._walk(payload):
                number = cls._number(value)
                if number is None or not any(name in key for name in names):
                    continue
                if 0 < number <= 86400 and number > best:
                    best = number
            if best:
                return best
        return 0.0

    @classmethod
    def _find_string(cls, payload, names) -> str | None:
        for path, key, value in cls._walk(payload):
            if not isinstance(value, str) or not value.strip():
                continue
            if not any(name in key for name in names):
                continue
            if any(bad in path.lower() for bad in ("thumbnail", "image", "avatar")):
                continue
            return value.strip()
        return None

    def _find_url(self, payload) -> str | None:
        """The hosted media link, chosen by scoring every URL in the payload."""
        tokens = ("downloadurl", "download", "cdn", "fileurl", "mediaurl", "url", "link")
        scored: list[tuple[int, str]] = []
        for path, _key, value in self._walk(payload):
            if not isinstance(value, str) or not value.lower().startswith(("http://", "https://")):
                continue
            low = path.lower()
            if any(bad in low for bad in ("thumbnail", "image", "avatar", "logo", "icon", "docs")):
                continue
            score = 0
            for position, token in enumerate(tokens):
                if token in low:
                    score = len(tokens) - position
                    break
            scored.append((score, value))
        return max(scored)[1] if scored else None

    def _status_of(self, payload) -> str:
        status = payload.get("status") if isinstance(payload, dict) else None
        if isinstance(status, str) and status.strip():
            return status.strip().upper()
        for _path, key, value in self._walk(payload):
            if not any(word in key for word in ("status", "state", "phase")):
                continue
            if isinstance(value, str) and value.strip() and "://" not in value:
                return value.strip().upper()
        return ""

    def _ready(self, kind: str, payload) -> bool:
        """True once a payload carries the answer we asked for, status or not."""
        if kind == "download":
            return bool(self._find_url(payload))
        return bool(self._find_number(payload, _DURATION_TIERS))

    @staticmethod
    def _job_id(payload: dict) -> str | None:
        for field in ("jobId", "job_id", "id", "requestId", "taskId"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def describe(cls, payload) -> str:
        """A compact key/type summary of a response, for the worker log."""
        parts: list[str] = []
        for path, _key, value in cls._walk(payload):
            if path.count(".") > 1 or "[" in path:
                continue
            parts.append(f"{path}:{type(value).__name__}")
            if len(parts) >= 20:
                break
        return ", ".join(parts) or "empty response"

    def _post(self, path: str, body: dict) -> dict:
        try:
            response = requests.post(
                f"{self.BASE}/{path}", headers=self._headers(), json=body, timeout=60
            )
        except requests.RequestException as exc:
            raise self.VidKrakenError(f"VidKraken is unreachable: {exc}") from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.ok:
            raise self.VidKrakenError(
                self._message(payload, f"VidKraken returned HTTP {response.status_code}")
            )
        return payload

    def _poll(self, kind: str, job_id: str) -> dict:
        deadline = time.time() + self.JOB_TIMEOUT
        while time.time() < deadline:
            try:
                response = requests.get(
                    f"{self.BASE}/{kind}/{job_id}", headers=self._headers(), timeout=60
                )
            except requests.RequestException as exc:
                raise self.VidKrakenError(f"VidKraken is unreachable: {exc}") from exc
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if not response.ok:
                raise self.VidKrakenError(
                    self._message(payload, f"VidKraken returned HTTP {response.status_code}")
                )
            status = self._status_of(payload)
            if status in self.DONE:
                return payload
            if status in self.BROKEN:
                raise self.VidKrakenError(
                    self._message(payload, "VidKraken could not fetch that video.")
                )
            if self._ready(kind, payload):
                # A finished job that hands back its answer without saying so.
                return payload
            time.sleep(self.POLL_SECONDS)
        raise self.VidKrakenError("VidKraken took too long to prepare that video.")

    def info(self, url: str) -> dict:
        """Video length and title, without queueing a download."""
        started = self._post("info", {"url": url})
        job_id = self._job_id(started)
        data = self._poll("info", job_id) if job_id else started
        return {
            "title": self._find_string(data, ("title",)),
            "duration": self._find_number(data, _DURATION_TIERS),
            "raw": data,
        }

    def _link(self, payload: dict) -> str:
        link = self._find_url(payload)
        if not link:
            raise self.VidKrakenError(
                "VidKraken finished but its response had no download link "
                f"({self.describe(payload)})"
            )
        return link

    def fetch(
        self,
        url: str,
        fmt: str,
        path: str,
        start: float | None = None,
        end: float | None = None,
        on_bytes=None,
    ) -> int:
        body: dict = {"url": url, "format": fmt}
        if start is not None and end is not None:
            low = max(0, int(math.floor(start)))
            high = max(low + 1, int(math.ceil(end)))
            body["startTime"] = low
            body["endTime"] = high
        started = self._post("download", body)
        job_id = self._job_id(started)
        completed = self._poll("download", job_id) if job_id else started
        link = self._link(completed)
        total = 0
        last_status = 0
        for attempt in range(4):
            try:
                with requests.get(
                    link,
                    headers=self._download_headers(),
                    stream=True,
                    timeout=1800,
                ) as response:
                    last_status = response.status_code
                    if response.ok:
                        with open(path, "wb") as handle:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                total += len(chunk)
                                if on_bytes:
                                    on_bytes(len(chunk))
                        break
                    if response.status_code != 429 and response.status_code < 500:
                        raise self.VidKrakenError(
                            f"The prepared file couldn't be read (HTTP {response.status_code})."
                        )
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        wait = max(1.0, min(float(retry_after), 30.0))
                    except ValueError:
                        wait = min(2 ** attempt, 15)
            except requests.RequestException as exc:
                if attempt == 3:
                    raise self.VidKrakenError(
                        f"The prepared file couldn't be downloaded: {exc}"
                    ) from exc
                wait = min(2 ** attempt, 15)
            if attempt < 3:
                time.sleep(wait)
        if total == 0 and last_status:
            raise self.VidKrakenError(
                f"The prepared file couldn't be read after retrying (HTTP {last_status})."
            )
        if total == 0:
            raise self.VidKrakenError("The prepared file was empty.")
        return total

    def status_line(self) -> str:
        if not self.enabled():
            return "vidkraken: no API key set, falling back to yt-dlp for YouTube"
        try:
            response = requests.get(f"{self.BASE}/me", headers=self._headers(), timeout=30)
            if response.ok:
                return f"vidkraken: connected ({response.text[:200]})"
            return (
                f"vidkraken: key rejected (HTTP {response.status_code}). "
                "Check VIDKRAKEN_API_KEY."
            )
        except requests.RequestException as exc:
            return f"vidkraken: unreachable - {exc}"


vidkraken = _VidKrakenClient()

APP_URL = os.environ.get("APP_URL", "http://localhost:8080").rstrip("/")
SECRET = os.environ.get("CLIP_WORKER_SECRET", "")
HEADERS = {"x-worker-secret": SECRET}


def _is_youtube(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host.endswith(("youtube.com", "youtu.be", "youtube-nocookie.com"))

TMP_ROOT = os.environ.get("IMPORT_TMP_DIR", tempfile.gettempdir())
TMP_PREFIX = "myclipai-import-"
MAX_BYTES = int(os.environ.get("IMPORT_MAX_BYTES", 500 * 1024 * 1024))
MAX_SECONDS = int(os.environ.get("IMPORT_MAX_SECONDS", 600))
ORPHAN_AGE_SECONDS = int(os.environ.get("IMPORT_ORPHAN_AGE_SECONDS", 3600))

SIGNIN_HINT = (
    "This video requires sign-in and can't be imported yet. "
    "Try a public video link instead."
)


class ImportError_(Exception):
    """A message that is safe to show the person who pasted the link."""


def api(path: str) -> str:
    return f"{APP_URL}/api/public/worker/imports/{path}"


def claim_import():
    r = requests.post(api("claim"), headers=HEADERS, json={"worker_id": os.environ.get("HOSTNAME", "worker")}, timeout=30)
    r.raise_for_status()
    return r.json().get("job")


def report(job_id: str, stage: str, pct: int) -> None:
    try:
        requests.post(
            api("progress"),
            headers=HEADERS,
            json={"job_id": job_id, "stage": stage, "progress": max(0, min(100, int(pct)))},
            timeout=20,
        )
    except requests.RequestException:
        pass


def finish(job_id: str, **body) -> None:
    try:
        requests.post(api("finish"), headers=HEADERS, json={"job_id": job_id, **body}, timeout=60)
    except requests.RequestException:
        pass


def cleanup_orphans() -> None:
    """Clears temp folders left behind by crashes or previous deploys."""
    now = time.time()
    for path in glob.glob(os.path.join(TMP_ROOT, TMP_PREFIX + "*")):
        try:
            if now - os.path.getmtime(path) > ORPHAN_AGE_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def friendly(message: str) -> str:
    m = message.lower()
    if any(
        s in m
        for s in (
            "sign in to confirm",
            "not a bot",
            "cookies",
            "members-only",
            "member only",
            "join this channel",
            "age-restricted",
            "age restricted",
            "confirm your age",
            "private video",
            "login required",
            "requires authentication",
        )
    ):
        return SIGNIN_HINT
    if "unavailable" in m or "does not exist" in m or "404" in m:
        return "That video isn't available. Check the link and try again."
    if "live" in m and "stream" in m:
        return "That link is a live stream, so it can't be imported yet."
    if "429" in m or "rate" in m:
        return "YouTube is rate-limiting us right now. Try again in a few minutes."
    return f"We couldn't download that video: {message[:200]}"


def probe(path: str) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        timeout=300,
    )
    return json.loads(out)


def download_public(url: str, workdir: str, job_id: str,
                    segment_start: float | None = None,
                    segment_end: float | None = None) -> str:
    import yt_dlp

    target = os.path.join(workdir, "source.%(ext)s")

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                if total > MAX_BYTES and segment_start is None:
                    raise ImportError_(
                        f"That video is over the {MAX_BYTES // (1024 * 1024)} MB import limit."
                    )
                report(job_id, "downloading", 5 + int(done / total * 75))

    opts = {
        "outtmpl": target,
        # Best quality at or below 1080p, delivered as mp4.
        # Prefer H.264 (avc1): it plays everywhere and the in-browser editor can
        # re-encode it. AV1/VP9 are only a last resort.
        "format": (
            "bestvideo[height<=1080][vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]"
            "/best[height<=1080][vcodec^=avc1][ext=mp4]"
            "/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
            "/best[height<=1080]"
        ),
        "merge_output_format": "mp4",
        "noprogress": True,
        "quiet": True,
        "retries": 2,
        "concurrent_fragment_downloads": 4,
        "progress_hooks": [hook],
        "remote_components": ["ejs:github"],
    }

    # Only YouTube blocks datacenter addresses, so only YouTube leaves through
    # the metered household exit. Everything else goes direct and costs nothing.
    if _is_youtube(url):
        proxy = os.environ.get("DOWNLOAD_PROXY") or os.environ.get("PROXY_URL")
        if proxy:
            opts["proxy"] = proxy


    # Use the same saved YouTube session as the clipping worker when one exists.
    cookies_raw = os.environ.get("YOUTUBE_COOKIES_TXT")
    cookies_b64 = os.environ.get("YOUTUBE_COOKIES_B64")
    if cookies_raw or cookies_b64:
        cookies_path = os.path.join(workdir, "cookies.txt")
        cookie_bytes = (
            cookies_raw.encode("utf-8", "replace") if cookies_raw else base64.b64decode(cookies_b64)
        )
        with open(cookies_path, "w", encoding="utf-8") as fh:
            fh.write(cookie_bytes.decode("utf-8", "replace"))
        opts["cookiefile"] = cookies_path

    if segment_start is not None and segment_end is not None:
        if segment_end <= segment_start or segment_end - segment_start > MAX_SECONDS:
            raise ImportError_("That clip range can't be prepared for the editor.")
        # Download only the source interval needed by this clip. This lets an
        # hour-long source safely produce a small, uncropped editor input.
        opts["download_ranges"] = lambda _info, _ydl: [{
            "start_time": max(0.0, segment_start),
            "end_time": segment_end,
        }]
        opts["force_keyframes_at_cuts"] = True

    def attempt(options: dict) -> None:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = float((info or {}).get("duration") or 0)
            if duration <= 0:
                raise ImportError_("We couldn't read that video's length — it may be live.")
            if duration > MAX_SECONDS and segment_start is None:
                raise ImportError_(
                    f"That video is {int(duration // 60)} minutes long. The editor imports up to "
                    f"{MAX_SECONDS // 60} minutes."
                )
            ydl.download([url])

    # Try the saved session first, then cookie-free clients that usually pass
    # for public videos.
    attempts: list[dict] = [opts]
    base = {k: v for k, v in opts.items() if k != "cookiefile"}
    for clients in (["android_vr"], ["tv"], ["ios"], ["mweb"], ["web_safari"], ["web_embedded"]):
        variant = dict(base)
        variant["extractor_args"] = {"youtube": {"player_client": clients}}
        attempts.append(variant)
        if "cookiefile" in opts:
            with_cookies = dict(variant)
            with_cookies["cookiefile"] = opts["cookiefile"]
            attempts.append(with_cookies)

    # A dead, expired or unpaid proxy must not stop a download that can still
    # go out directly: retry every variant with the proxy removed.
    if opts.get("proxy"):
        attempts.extend(
            {k: v for k, v in variant.items() if k != "proxy"} for variant in list(attempts)
        )

    last_error: Exception | None = None
    for options in attempts:
        try:
            attempt(options)
            last_error = None
            break
        except ImportError_:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            # Clear any partial files before the next attempt.
            for leftover in glob.glob(os.path.join(workdir, "source.*")):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
    if last_error is not None:
        raise ImportError_(friendly(str(last_error))) from last_error

    for name in sorted(os.listdir(workdir)):
        if name.startswith("source."):
            return os.path.join(workdir, name)
    raise ImportError_("The video downloaded but no file was produced.")


def download_link(url: str, workdir: str, job_id: str,
                  segment_start: float | None = None,
                  segment_end: float | None = None) -> str:
    """YouTube goes through the paid download service when one is configured;
    everything else, and any failure, falls back to yt-dlp."""
    if _is_youtube(url) and vidkraken.enabled():
        path = os.path.join(workdir, "source.mp4")
        seen = 0

        def meter(count: int) -> None:
            # Stop paying for a file we are going to refuse anyway.
            nonlocal seen
            seen += count
            if seen > MAX_BYTES:
                raise ImportError_(
                    f"That video is over the {MAX_BYTES // (1024 * 1024)} MB import limit."
                )

        try:
            report(job_id, "downloading", 10)
            if segment_start is not None and segment_end is not None:
                if segment_end <= segment_start or segment_end - segment_start > MAX_SECONDS:
                    raise ImportError_("That clip range can't be prepared for the editor.")
                size = vidkraken.fetch(url, vidkraken.QUALITY, path,
                                       start=float(segment_start), end=float(segment_end),
                                       on_bytes=meter)
            else:
                size = vidkraken.fetch(url, vidkraken.QUALITY, path, on_bytes=meter)
            if size > MAX_BYTES:
                raise ImportError_(
                    f"That video is over the {MAX_BYTES // (1024 * 1024)} MB import limit."
                )
            report(job_id, "downloading", 78)
            return path
        except ImportError_:
            raise
        except vidkraken.VidKrakenError as exc:
            print(f"vidkraken import failed, falling back to yt-dlp: {exc}")
            try:
                os.remove(path)
            except OSError:
                pass
    return download_public(url, workdir, job_id, segment_start, segment_end)


def upload(path: str, upload_url: str) -> None:
    with open(path, "rb") as fh:
        resp = requests.put(
            upload_url,
            data=fh,
            headers={"content-type": "video/mp4", "x-upsert": "true"},
            timeout=1800,
        )
    if not resp.ok:
        raise ImportError_("We couldn't store the downloaded video. Please try again.")


def transcribe_words(path: str, workdir: str) -> list[dict]:
    """Create real word timings for the editor's animated caption track."""
    try:
        from faster_whisper import WhisperModel
        audio = os.path.join(workdir, "captions.wav")
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", audio],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=900,
        )
        model = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
        segments, _info = model.transcribe(audio, word_timestamps=True, vad_filter=True)
        return [
            {"text": word.word.strip(), "start": round(float(word.start), 3), "end": round(float(word.end), 3)}
            for segment in segments
            for word in (segment.words or [])
            if word.word.strip()
        ]
    except Exception:  # captions are optional; importing the playable video is primary
        return []


def process_import(job: dict) -> None:
    job_id = job["id"]
    workdir = tempfile.mkdtemp(prefix=TMP_PREFIX, dir=TMP_ROOT)
    try:
        report(job_id, "downloading", 5)
        segment_start = job.get("segment_start_seconds")
        segment_end = job.get("segment_end_seconds")
        direct = job.get("source_download_url")
        if direct:
            original = os.path.join(workdir, "original.mp4")
            with requests.get(direct, stream=True, timeout=1800) as resp:
                if not resp.ok:
                    raise ImportError_("The original upload is no longer available. Re-upload it to use Original frame.")
                with open(original, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
            path = original
        else:
            path = download_link(job["url"], workdir, job_id, segment_start, segment_end)

        if direct and segment_start is not None and segment_end is not None:
            trimmed = os.path.join(workdir, "source-segment.mp4")
            subprocess.check_call([
                "ffmpeg", "-y", "-ss", f"{float(segment_start):.3f}",
                "-t", f"{float(segment_end) - float(segment_start):.3f}", "-i", path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", trimmed,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
            path = trimmed

        size = os.path.getsize(path)
        if size == 0:
            raise ImportError_("The downloaded file was empty.")
        if size > MAX_BYTES:
            raise ImportError_(
                f"That video is {size / 1024 / 1024:.0f} MB — the import limit is "
                f"{MAX_BYTES // (1024 * 1024)} MB. Trim it and upload the file instead."
            )

        meta = probe(path)
        duration = float(meta.get("format", {}).get("duration") or 0)
        if duration > MAX_SECONDS:
            raise ImportError_(
                f"That video is {int(duration // 60)} minutes long. The editor imports up to "
                f"{MAX_SECONDS // 60} minutes."
            )
        video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})

        # Always transcribe the exact file delivered to the editor. For an
        # edited clip this is already the matching source segment, so these
        # timestamps begin at zero and stay aligned with preview and export.
        report(job_id, "transcribing", 82)
        caption_words = transcribe_words(path, workdir)

        report(job_id, "uploading", 85)
        upload(path, job["upload_url"])

        finish(
            job_id,
            status="ready",
            storage_path=job["storage_path"],
            size_bytes=size,
            duration_seconds=round(duration, 2),
            width=int(video.get("width") or 0),
            height=int(video.get("height") or 0),
            caption_words=caption_words,
        )
        print(f"import {job_id}: stored {size / 1024 / 1024:.1f} MB")
    except ImportError_ as exc:
        finish(job_id, status="failed", error=str(exc))
    except Exception:  # noqa: BLE001
        print(traceback.format_exc())
        finish(job_id, status="failed", error="Something went wrong importing that video.")
    finally:
        # Railway's disk is ephemeral and shared — always clean up.
        shutil.rmtree(workdir, ignore_errors=True)
        cleanup_orphans()
