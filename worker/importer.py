"""
Link import jobs.

Downloads a *public* YouTube video with yt-dlp (no cookies, no sign-in) into
the container's ephemeral /tmp, checks it against the editor's limits, pushes
the MP4 to private storage with a one-time signed upload URL and then deletes
every temp file it made. Nothing is expected to survive a restart.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback

import requests

APP_URL = os.environ.get("APP_URL", "http://localhost:8080").rstrip("/")
SECRET = os.environ.get("CLIP_WORKER_SECRET", "")
HEADERS = {"x-worker-secret": SECRET}

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
        # Public videos only: never attach cookies or credentials here.
    }

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

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
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
    except ImportError_:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImportError_(friendly(str(exc))) from exc

    for name in sorted(os.listdir(workdir)):
        if name.startswith("source."):
            return os.path.join(workdir, name)
    raise ImportError_("The video downloaded but no file was produced.")


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
            path = download_public(job["url"], workdir, job_id, segment_start, segment_end)

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

        if job.get("editor_clip_id"):
            caption_words = []
        else:
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
