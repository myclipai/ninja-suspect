"""
MyClip AI render worker.

Polls the app for clip jobs, downloads the real source video, transcribes it,
asks the app's AI which moments to cut, renders true 1080x1920 vertical MP4s
with FFmpeg (facecam-aware layout, smoothed subject tracking, burned-in
captions) and uploads each finished file back to the app.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field

import numpy as np
import requests

import importer

APP_URL = os.environ.get("APP_URL", "http://localhost:8080").rstrip("/")
SECRET = os.environ.get("CLIP_WORKER_SECRET", "")
WORKER_ID = os.environ.get("FLY_MACHINE_ID") or os.environ.get("HOSTNAME") or f"worker-{uuid.uuid4().hex[:8]}"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "10"))
MAX_SOURCE_SECONDS = int(os.environ.get("MAX_SOURCE_SECONDS", "5400"))  # 90 minutes
WINDOW_PAD_SECONDS = float(os.environ.get("CLIP_WINDOW_PAD_SECONDS", "8"))

OUT_W, OUT_H = 1080, 1920
HEADERS = {"x-worker-secret": SECRET}


# --------------------------------------------------------------------------- #
# app API
# --------------------------------------------------------------------------- #
def api(path: str) -> str:
    return f"{APP_URL}/api/public/worker/{path}"


def claim():
    r = requests.post(api("claim"), headers=HEADERS, json={"worker_id": WORKER_ID}, timeout=30)
    r.raise_for_status()
    return r.json().get("job")


class Cancelled(Exception):
    """The user pressed Cancel while this job was running."""


def progress(job_id: str, stage: str, pct: int, **extra):
    """Reports a stage and checks whether the user has cancelled meanwhile."""
    body = {"job_id": job_id, "stage": stage, "progress": max(0, min(100, int(pct)))}
    body.update(extra)
    try:
        r = requests.post(api("progress"), headers=HEADERS, json=body, timeout=30)
        if r.ok and r.json().get("cancelled"):
            raise Cancelled()
    except requests.RequestException:
        pass


def finish(job_id: str, status: str, error: str | None = None,
           downloaded_bytes: int | None = None):
    body = {"job_id": job_id, "status": status}
    if error:
        body["error"] = error[:1900]
    if downloaded_bytes is not None:
        body["downloaded_bytes"] = int(downloaded_bytes)
    requests.post(api("finish"), headers=HEADERS, json=body, timeout=60)


def upload_clip(job_id: str, path: str, meta: dict, thumb: str | None = None):
    handles = []
    try:
        fh = open(path, "rb")
        handles.append(fh)
        files = {
            "meta": (None, json.dumps({**meta, "job_id": job_id}), "application/json"),
            "file": (os.path.basename(path), fh, "video/mp4"),
        }
        if thumb and os.path.exists(thumb):
            th = open(thumb, "rb")
            handles.append(th)
            files["thumb"] = (os.path.basename(thumb), th, "image/jpeg")
        r = requests.post(api("clip"), headers=HEADERS, files=files, timeout=900)
    finally:
        for h in handles:
            h.close()
    if not r.ok:
        raise RuntimeError(f"Upload failed [{r.status_code}]: {r.text[:300]}")


def analyze(job_id: str, duration: float, clip_count: int, transcript, signals):
    r = requests.post(
        api("analyze"),
        headers=HEADERS,
        json={
            "job_id": job_id,
            "duration_seconds": duration,
            "clip_count": clip_count,
            "transcript": transcript,
            "signals": signals,
        },
        timeout=600,
    )
    if not r.ok:
        raise UserFacingError(r.json().get("error", f"Moment selection failed ({r.status_code}).")
                              if r.headers.get("content-type", "").startswith("application/json")
                              else f"Moment selection failed ({r.status_code}).")
    return r.json().get("clips", [])


class UserFacingError(Exception):
    """An error whose message is safe and useful to show the person waiting."""


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #
def run(cmd: list[str], timeout: int = 7200):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr[-800:]}")
    return proc.stdout


def probe(path: str) -> dict:
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ], timeout=300)
    return json.loads(out)


# --------------------------------------------------------------------------- #
# 1. reading the source with as few paid bytes as possible
# --------------------------------------------------------------------------- #
VIDEO_FORMAT = (
    "bestvideo[height<=1080][vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=1080][vcodec^=avc1][ext=mp4]"
    "/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=1080]"
)
AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio/best"
PLAYER_CLIENTS = (["android_vr"], ["tv"], ["ios"], ["mweb"], ["web_safari"], ["web_embedded"])


def link_setup(job: dict, workdir: str) -> tuple[str, dict]:
    """yt-dlp settings shared by every link download a job makes."""
    url = job["source_url"]
    platform = job.get("platform") or ""
    opts = {
        "merge_output_format": "mp4",
        "noprogress": True,
        "quiet": True,
        "retries": 3,
        "concurrent_fragment_downloads": 4,
        # YouTube serves a JS challenge; allow yt-dlp to fetch its solver.
        "remote_components": ["ejs:github"],
    }
    if platform == "youtube":
        # Only YouTube blocks datacenter addresses, so only its downloads leave
        # through the metered household exit. Twitch, Kick and uploads go
        # direct and cost nothing.
        proxy = os.environ.get("DOWNLOAD_PROXY") or os.environ.get("PROXY_URL")
        if proxy:
            opts["proxy"] = proxy
        cookies_raw = os.environ.get("YOUTUBE_COOKIES_TXT")
        cookies_b64 = os.environ.get("YOUTUBE_COOKIES_B64")
        if cookies_raw or cookies_b64:
            cookies_path = os.path.join(workdir, "cookies.txt")
            cookie_bytes = (
                cookies_raw.encode("utf-8", "replace") if cookies_raw else base64.b64decode(cookies_b64)
            )
            # Netscape cookies must be readable as text; replace any invalid bytes.
            with open(cookies_path, "w", encoding="utf-8") as fh:
                fh.write(cookie_bytes.decode("utf-8", "replace"))
            opts["cookiefile"] = cookies_path
    if platform == "twitch_channel":
        # Most recent VOD from the channel.
        url = url.rstrip("/") + "/videos"
        opts["playlistend"] = 1
    return url, opts


class JobSource:
    """Where a job's video comes from, and the least of it we have to fetch.

    Uploaded files and saved windows arrive as a file we already own, so
    nothing metered is spent on them. A link is read in two passes — audio only
    to pick the moments, then just the chosen windows to render — so a 24
    minute source pays for about a minute of picture instead of all of it.
    """

    def __init__(self, job: dict, workdir: str):
        self.job = job
        self.workdir = workdir
        self.local: str | None = None
        self.opts: dict = {}
        self.url = job["source_url"]
        self.duration = 0.0
        self.used = 0
        self.offset = float((job.get("render_options") or {}).get("window_offset_seconds") or 0)
        self._seen: dict[str, int] = {}

    # -- set up ----------------------------------------------------------
    def prepare(self) -> None:
        direct = self.job.get("source_download_url")
        if direct:
            self.local = self._fetch_whole(direct)
        else:
            self.url, self.opts = link_setup(self.job, self.workdir)

    def _fetch_whole(self, url: str) -> str:
        """Reads a file we already keep in private storage: no metered bytes."""
        path = os.path.join(self.workdir, "stored.mp4")
        with requests.get(url, stream=True, timeout=1800) as resp:
            if not resp.ok:
                raise UserFacingError("The copy of that video we kept can't be read any more.")
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
        if os.path.getsize(path) == 0:
            raise UserFacingError("That file is empty.")
        return path

    # -- metadata --------------------------------------------------------
    def read_duration(self) -> tuple[float, str | None]:
        """How long the source is, without downloading the picture."""
        if self.local:
            self.duration = float(probe(self.local)["format"]["duration"])
            return self.duration, None

        info = self._ladder(lambda _o: None, download=False)
        duration = float(info.get("duration") or 0)
        if duration <= 0:
            raise UserFacingError("We couldn't read that video's length — it may still be live.")
        if duration > MAX_SOURCE_SECONDS:
            raise UserFacingError(
                f"That video is {int(duration / 60)} minutes long. Please use one under "
                f"{MAX_SOURCE_SECONDS // 60} minutes."
            )
        self.duration = duration
        return duration, info.get("title")

    # -- audio -----------------------------------------------------------
    def audio_file(self) -> str:
        """Something ffmpeg can pull sound out of, at the cheapest price."""
        if self.local:
            return self.local

        def mutate(options: dict) -> None:
            options["outtmpl"] = os.path.join(self.workdir, "audio.%(ext)s")
            options["format"] = AUDIO_FORMAT

        self._ladder(mutate, download=True)
        return self._claim("audio.")

    # -- picture ---------------------------------------------------------
    def window(self, start: float, end: float) -> tuple[str, float]:
        """Fetches this moment, plus a little room to breathe, and returns the
        file together with the source time its first second sits at."""
        if self.local:
            return self.local, self.offset

        lo = max(0.0, start - WINDOW_PAD_SECONDS)
        hi = min(self.duration or end + WINDOW_PAD_SECONDS, end + WINDOW_PAD_SECONDS)

        def mutate(options: dict) -> None:
            options["outtmpl"] = os.path.join(self.workdir, "window.%(ext)s")
            options["format"] = VIDEO_FORMAT
            options["download_ranges"] = lambda _info, _ydl: [{"start_time": lo, "end_time": hi}]
            options["force_keyframes_at_cuts"] = True

        self._ladder(mutate, download=True)
        return self._claim("window."), lo

    # -- the download ladder ---------------------------------------------
    def _ladder(self, mutate, download: bool) -> dict:
        """Tries the saved session, then cookie-free clients, until one works."""
        base = dict(self.opts)
        variants: list[dict] = [base]
        if (self.job.get("platform") or "") == "youtube":
            # Cookie-free fallbacks: these YouTube clients usually pass without
            # a signed-in session, which is what public videos need. The saved
            # session is also retried with alternative clients, because the bot
            # check is tied to the client YouTube thinks is asking.
            without = {k: v for k, v in base.items() if k != "cookiefile"}
            for clients in PLAYER_CLIENTS:
                variant = dict(without)
                variant["extractor_args"] = {"youtube": {"player_client": clients}}
                variants.append(variant)
                if "cookiefile" in base:
                    with_cookies = dict(variant)
                    with_cookies["cookiefile"] = base["cookiefile"]
                    variants.append(with_cookies)

        last: Exception | None = None
        for index, options in enumerate(variants):
            if index:
                # Short backoff: YouTube's block often clears between clients.
                time.sleep(min(2 * index, 8))
                self._clear_partials()
            mutate(options)
            try:
                return self._run(options, download)
            except Exception as exc:  # noqa: BLE001 - the message is shown to the user
                last = exc
        raise UserFacingError(
            twitch_friendly_error(str(last), self.job.get("platform") or "")
        ) from last

    def _run(self, options: dict, download: bool) -> dict:
        import yt_dlp

        options["progress_hooks"] = [self._count_bytes]
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(self.url, download=download)
        if info.get("entries"):
            entries = [e for e in info["entries"] if e]
            if not entries:
                raise UserFacingError("That Twitch channel has no videos we can use.")
            info = entries[0]
        return info

    def _count_bytes(self, state: dict) -> None:
        """Adds up what actually crossed the metered connection."""
        name = state.get("filename") or "?"
        done = int(state.get("downloaded_bytes") or 0)
        if done > self._seen.get(name, 0):
            self.used += done - self._seen.get(name, 0)
            self._seen[name] = done

    def _clear_partials(self) -> None:
        for name in os.listdir(self.workdir):
            if name.startswith(("source.", "audio.", "window.", "stored.")):
                try:
                    os.remove(os.path.join(self.workdir, name))
                except OSError:
                    pass

    def _claim(self, prefix: str) -> str:
        """Names the file a download just produced so the next one starts clean."""
        made = [os.path.join(self.workdir, n) for n in os.listdir(self.workdir)
                if n.startswith(prefix)]
        if not made:
            raise UserFacingError("The video downloaded but no file was produced.")
        newest = max(made, key=os.path.getmtime)
        final = os.path.join(self.workdir, f"{prefix[:-1]}-{uuid.uuid4().hex[:8]}.mp4")
        os.rename(newest, final)
        for leftover in made:
            if leftover != newest and os.path.isfile(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
        return final


def twitch_friendly_error(message: str, platform: str) -> str:
    m = message.lower()
    if "subscriber" in m or "subscribers only" in m:
        return "That VOD is for subscribers only, so it can't be clipped."
    if "private" in m:
        return "That video is private."
    if "does not exist" in m or "404" in m or "unavailable" in m:
        return "That video no longer exists — Twitch VODs expire after a few weeks."
    if "page needs to be reloaded" in m or "challenge" in m:
        return ("YouTube changed its download protection. The render machine needs its "
                "downloader updated (yt-dlp + a JavaScript runtime) before this link will work.")
    if "sign in to confirm" in m or "not a bot" in m:
        return ("YouTube is blocking downloads from this machine right now. "
                "Try again in a few minutes, or use a different video link.")
    if "age" in m and "restrict" in m:
        return "That video is age-restricted and can't be downloaded."
    if "geo" in m or "not available in your" in m:
        return "That video is blocked in the region this render machine runs in."
    if "429" in m or "rate" in m and "limit" in m:
        return "The platform is rate-limiting us right now. Try again in a few minutes."
    if platform.startswith("twitch"):
        return f"Twitch wouldn't give us that video: {message[:200]}"
    return f"We couldn't download that video: {message[:200]}"


# --------------------------------------------------------------------------- #
# 2-3. transcribe + signals
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[dict] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)


def transcribe(video_path: str, workdir: str) -> Transcript:
    from faster_whisper import WhisperModel

    audio = os.path.join(workdir, "audio.wav")
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio])

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio, word_timestamps=True, vad_filter=True)

    result = Transcript()
    for seg in segments:
        result.segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
        for w in (seg.words or []):
            result.words.append(Word(float(w.start), float(w.end), w.word.strip()))
    return result


def audio_signals(workdir: str, transcript: Transcript, duration: float) -> list[dict]:
    """Loudness and speech count per 30 second window, read off the audio that
    transcription already left on disk.

    Only the sound is fetched before moments are chosen, so these signals come
    from a file that costs about a megabyte a minute instead of the whole
    picture. They still point the AI at the loud, busy parts of a video.
    """
    import wave

    window = 30.0
    windows = max(1, int(duration // window) + 1)
    audio = os.path.join(workdir, "audio.wav")
    signals: list[dict] = []

    loud = []
    try:
        with wave.open(audio, "rb") as wf:
            rate = wf.getframerate()
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
        per = int(rate * window)
        for i in range(windows):
            chunk = frames[i * per:(i + 1) * per]
            loud.append(float(np.sqrt(np.mean(chunk**2)) / 327.68) if chunk.size else 0.0)
    except Exception:  # noqa: BLE001 - signals are best effort
        loud = [0.0] * windows

    spoken = [0] * windows
    for word in transcript.words:
        idx = int(word.start // window)
        if 0 <= idx < windows:
            spoken[idx] += 1

    for i in range(windows):
        signals.append({
            "start": i * window,
            "end": min(duration, (i + 1) * window),
            "loudness": round(loud[i] if i < len(loud) else 0.0, 2),
            "words": spoken[i],
        })
    return signals


# --------------------------------------------------------------------------- #
# 4. framing analysis
# --------------------------------------------------------------------------- #
@dataclass
class Framing:
    facecam: tuple[int, int, int, int] | None  # x, y, w, h in source pixels
    track: list[tuple[float, float]]           # (time, centre x as fraction of width)


def analyse_framing(video_path: str, start: float, end: float, width: int, height: int) -> Framing:
    """Find a facecam box (if any) and a smoothed horizontal tracking path."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    samples = max(6, min(30, int((end - start) / 2)))
    boxes: list[tuple[int, int, int, int]] = []
    track: list[tuple[float, float]] = []
    prev_gray = None

    for i in range(samples):
        t = start + (end - start) * (i / max(1, samples - 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = 640 / max(1, frame.shape[1])
        small = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(28, 28))

        centre = None
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            boxes.append((int(fx / scale), int(fy / scale), int(fw / scale), int(fh / scale)))
            centre = (fx + fw / 2) / small.shape[1]
        elif prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            cols = diff.sum(axis=0)
            if cols.sum() > 0:
                centre = float(np.argmax(np.convolve(cols, np.ones(64) / 64, mode="same"))) / small.shape[1]
        prev_gray = gray
        track.append((t, centre if centre is not None else 0.5))

    cap.release()

    facecam = None
    if len(boxes) >= max(3, samples // 3):
        xs = np.array([b[0] + b[2] / 2 for b in boxes])
        ys = np.array([b[1] + b[3] / 2 for b in boxes])
        ws = np.array([b[2] for b in boxes])
        # A facecam sits still in one part of the frame and is small.
        if xs.std() < width * 0.06 and ys.std() < height * 0.06 and ws.mean() < width * 0.22:
            cw = int(min(width * 0.42, ws.mean() * 4.2))
            ch = int(cw * 9 / 16)
            cx = int(np.clip(xs.mean() - cw / 2, 0, width - cw))
            cy = int(np.clip(ys.mean() - ch / 2, 0, height - ch))
            facecam = (cx, cy, cw, ch)

    # Smooth the tracking path so the crop never snaps.
    if track:
        values = np.array([c for _, c in track])
        kernel = np.ones(5) / 5
        smoothed = np.convolve(np.pad(values, 2, mode="edge"), kernel, mode="valid")
        track = [(t, float(v)) for (t, _), v in zip(track, smoothed)]

    return Framing(facecam=facecam, track=track or [(start, 0.5)])


def crop_x_expression(track: list[tuple[float, float]], start: float, crop_w: int, width: int) -> str:
    """A piecewise-linear ffmpeg expression that pans the crop with the action."""
    max_x = max(0, width - crop_w)
    points = [(max(0.0, t - start), float(np.clip(c * width - crop_w / 2, 0, max_x))) for t, c in track]
    if len(points) == 1:
        return f"{points[0][1]:.1f}"
    expr = f"{points[-1][1]:.1f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(points, points[1:]))):
        span = max(0.001, t1 - t0)
        seg = f"({x0:.1f}+({x1 - x0:.1f})*(t-{t0:.3f})/{span:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


# --------------------------------------------------------------------------- #
# 5. captions
# --------------------------------------------------------------------------- #
def write_captions(words: list[Word], start: float, end: float, colour: str, position: str, path: str) -> bool:
    chosen = [w for w in words if w.end > start and w.start < end]
    if not chosen:
        return False

    def ass_colour(hexcode: str) -> str:
        h = hexcode.lstrip("#")
        if len(h) != 6:
            return "&H00FFFFFF"
        return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()

    align = {"top": 8, "middle": 5, "bottom": 2}.get(position, 2)
    margin_v = {"top": 220, "middle": 0, "bottom": 320}.get(position, 320)

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {OUT_W}", f"PlayResY: {OUT_H}", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BackColour,Bold,Italic,"
        "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV,Encoding",
        f"Style: Cap,DejaVu Sans,72,{ass_colour(colour)},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,"
        f"1,5,2,{align},90,90,{margin_v},1",
        "", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    def stamp(value: float) -> str:
        value = max(0.0, value)
        h, rem = divmod(value, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"

    # Group words into short phrases that fit a 9:16 frame.
    group: list[Word] = []
    for word in chosen:
        group.append(word)
        text = " ".join(w.text for w in group)
        if len(text) >= 28 or word.text.endswith((".", "!", "?", ",")):
            lines.append(
                f"Dialogue: 0,{stamp(group[0].start - start)},{stamp(group[-1].end - start)},Cap,,0,0,0,,{text}"
            )
            group = []
    if group:
        text = " ".join(w.text for w in group)
        lines.append(
            f"Dialogue: 0,{stamp(group[0].start - start)},{stamp(group[-1].end - start)},Cap,,0,0,0,,{text}"
        )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return True


# --------------------------------------------------------------------------- #
# 6. render one clip
# --------------------------------------------------------------------------- #
def render_clip(video_path: str, workdir: str, job: dict, clip: dict, words: list[Word],
                width: int, height: int, offset: float = 0.0) -> tuple[str, str | None]:
    """Renders one clip. `offset` is the source time the file's first second
    sits at, so a short window file and a whole source are handled the same."""
    start, end = float(clip["start"]) - offset, float(clip["end"]) - offset
    # Clamp to the real duration so an over-eager timestamp can't kill ffmpeg.
    meta = probe(video_path)
    real = float(meta["format"]["duration"])
    start = min(max(0.0, start), max(0.0, real - 1.0))
    end = min(max(start + 1.0, end), real)
    duration = end - start
    if offset:
        words = [Word(w.start - offset, w.end - offset, w.text) for w in words]
    framing = analyse_framing(video_path, start, end, width, height)

    layout = (job.get("layout") or "auto").lower()
    options = job.get("render_options") or {}
    zoom = float(options.get("zoom") or 1)
    if layout in ("gameplay", "facecam", "zoomed", "original"):
        framing.facecam = None
    elif layout == "streamer" and not framing.facecam:
        # Asked for the streamer stack but no camera was found: use the top
        # third of the frame as the camera region.
        framing.facecam = (0, 0, width, max(1, height // 3))

    if layout == "original":
        # Full uncropped source, fit to the frame width, centered, with the
        # empty space filled by a blurred, darkened copy of the video.
        filter_complex = (
            f"[0:v]split[bgs][fgs];"
            f"[bgs]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=24:2,eq=brightness=-0.12[bg];"
            f"[fgs]scale={OUT_W}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
        )
    elif framing.facecam:
        fx, fy, fw, fh = framing.facecam
        cam_h = 620
        play_h = OUT_H - cam_h
        crop_w = min(width, int(height * (OUT_W / play_h)))
        x_expr = crop_x_expression(framing.track, start, crop_w, width)
        filter_complex = (
            f"[0:v]crop={fw}:{fh}:{fx}:{fy},scale={OUT_W}:{cam_h}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{cam_h},setsar=1[cam];"
            f"[0:v]crop={crop_w}:{height}:'{x_expr}':0,scale={OUT_W}:{play_h}:"
            f"force_original_aspect_ratio=increase,crop={OUT_W}:{play_h},setsar=1[play];"
            f"[cam][play]vstack=inputs=2[v]"
        )
    else:
        crop_w = min(width, int(height * OUT_W / OUT_H / max(1.0, zoom)))
        x_expr = crop_x_expression(framing.track, start, crop_w, width)
        filter_complex = (
            f"[0:v]crop={crop_w}:{height}:'{x_expr}':0,scale={OUT_W}:{OUT_H}:"
            f"force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},setsar=1[v]"
        )

    if job.get("captions_enabled", True):
        ass = os.path.join(workdir, f"cap-{uuid.uuid4().hex[:8]}.ass")
        if write_captions(words, start, end, job.get("caption_color", "#FFFFFF"),
                          job.get("caption_position", "bottom"), ass):
            escaped = ass.replace("\\", "/").replace(":", "\\:")
            filter_complex += f";[v]subtitles='{escaped}'[vout]"
        else:
            filter_complex += ";[v]null[vout]"
    else:
        filter_complex += ";[v]null[vout]"

    out = os.path.join(workdir, f"clip-{uuid.uuid4().hex[:8]}.mp4")

    def encode(filters: str) -> None:
        run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", video_path,
            "-filter_complex", filters,
            "-map", "[vout]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-r", "30",
            out,
        ])

    try:
        encode(filter_complex)
    except Exception:
        if "subtitles" not in filter_complex:
            raise
        # Caption burn-in is the fragile part; retry once without it so the
        # clip still renders instead of failing the whole job.
        encode(filter_complex.replace(f";[v]subtitles='{escaped}'[vout]", ";[v]null[vout]"))

    thumb = os.path.join(workdir, f"thumb-{uuid.uuid4().hex[:8]}.jpg")
    try:
        run([
            "ffmpeg", "-y", "-ss", f"{min(1.0, duration / 2):.2f}", "-i", out,
            "-frames:v", "1", "-q:v", "3", thumb,
        ], timeout=120)
    except Exception:  # noqa: BLE001 - a missing poster frame is not fatal
        thumb = None
    return out, thumb


# --------------------------------------------------------------------------- #
# job pipeline
# --------------------------------------------------------------------------- #
def store_window(job_id: str, path: str) -> str | None:
    """Keeps a paid-for window in private storage so it is never bought twice.

    Best effort: a window that fails to store only means the next re-render
    pays for it again, so it must never fail the job that is already running.
    """
    try:
        with open(path, "rb") as fh:
            r = requests.post(
                api("window"),
                headers=HEADERS,
                files={"job_id": (None, job_id), "file": (os.path.basename(path), fh, "video/mp4")},
                timeout=900,
            )
        if r.ok:
            return r.json().get("path")
        print(f"window store failed [{r.status_code}]: {r.text[:200]}")
    except (requests.RequestException, OSError) as exc:
        print(f"window store failed: {exc}")
    return None


def render_one(source: JobSource, workdir: str, job: dict, clip: dict, words: list[Word]):
    """Fetches just this moment and renders it.

    Returns the finished file, its poster frame, and the window it came from so
    the app can reuse that window for a later re-render or editor open.
    """
    file, window_start = source.window(float(clip["start"]), float(clip["end"]))
    meta = probe(file)
    stream = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    if not stream:
        raise UserFacingError("That file has no video track.")
    width, height = int(stream["width"]), int(stream["height"])
    window_seconds = float(meta["format"]["duration"])

    path, thumb = render_clip(file, workdir, job, clip, words, width, height, window_start)

    if source.local:
        # Nothing metered was spent here. A reused window is handed back so the
        # clip keeps pointing at the copy we already own; an uploaded source is
        # already stored whole, so there is nothing extra to keep.
        if job.get("source_kind") == "window" and job.get("source_storage_path"):
            return path, thumb, (
                job["source_storage_path"],
                window_start,
                window_start + window_seconds,
                job.get("source_expires_at"),
            )
        return path, thumb, (None, None, None, None)

    stored = store_window(job["id"], file)
    return path, thumb, (stored, window_start, window_start + window_seconds, None)


def process(job: dict):
    job_id = job["id"]
    workdir = tempfile.mkdtemp(prefix="clipjob-")
    source = JobSource(job, workdir)
    try:
        progress(job_id, "importing", 10)
        source.prepare()
        duration, title = source.read_duration()
        if title:
            progress(job_id, "importing", 70, title=title[:300])
        progress(job_id, "importing", 100, duration_seconds=duration)

        # Picking moments only needs the sound track, which costs about a
        # megabyte a minute, so the picture is never downloaded up front.
        progress(job_id, "analyzing", 25)
        transcript = transcribe(source.audio_file(), workdir)
        signals = audio_signals(workdir, transcript, duration)
        progress(job_id, "analyzing", 100)

        options = job.get("render_options") or {}
        if options.get("start") is not None and options.get("end") is not None:
            # Re-render of one existing clip with new framing/caption settings.
            clips = [{
                "start": float(options["start"]),
                "end": float(options["end"]),
                "title": options.get("title") or "Clip",
                "reason": "Re-rendered with your settings.",
                "score": 80,
            }]
            progress(job_id, "finding_clips", 100)
        else:
            progress(job_id, "finding_clips", 30)
            clips = analyze(job_id, duration, int(job.get("clip_count", 5)), transcript.segments, signals)
            if not clips:
                raise UserFacingError(
                    "We couldn't find any strong moments in this video. "
                    "Try a longer video with more speech."
                )
            progress(job_id, "finding_clips", 100)

        rendered = 0
        first_error = None
        total = len(clips)
        for index, clip in enumerate(clips):
            base = int(index / total * 100)
            step = int(100 / total)
            progress(job_id, "creating_clips", base)
            progress(job_id, "reframing", min(99, base + step // 3))
            if job.get("captions_enabled", True):
                progress(job_id, "captioning", min(99, base + step // 2))
            progress(job_id, "rendering", min(99, base + step))
            try:
                path, thumb, window = render_one(source, workdir, job, clip, transcript.words)
                window_path, window_start, window_end, window_expiry = window
                clip_meta = {
                    "title": clip.get("title", f"Clip {index + 1}")[:120],
                    "reason": clip.get("reason", "")[:500],
                    "layout": (job.get("layout") or "auto"),
                    "score": float(clip.get("score", 70)),
                    "start_seconds": float(clip["start"]),
                    "end_seconds": float(clip["end"]),
                    "window_storage_path": window_path,
                    "window_start_seconds": window_start,
                    "window_end_seconds": window_end,
                    "window_expires_at": window_expiry,
                }
                if clip.get("description"):
                    clip_meta["description"] = str(clip["description"])[:1000]
                upload_clip(job_id, path, clip_meta, thumb)
                os.remove(path)
                if thumb and os.path.exists(thumb):
                    os.remove(thumb)
                rendered += 1
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad clip shouldn't kill the job
                print(f"clip {index} failed:\n{traceback.format_exc()}")
                if first_error is None:
                    first_error = str(exc).strip().splitlines()[-1][:400]

        if rendered == 0:
            detail = f" ({first_error})" if first_error else ""
            raise UserFacingError(f"Every clip failed to render{detail}. Please try a different video.")
        finish(job_id, "completed", downloaded_bytes=source.used)
        print(f"job {job_id}: {rendered}/{total} clips rendered, {source.used / 1024 / 1024:.0f} MB fetched")

    except Cancelled:
        finish(job_id, "cancelled", "Cancelled.", downloaded_bytes=source.used)
        print(f"job {job_id} cancelled by the user")
    except UserFacingError as exc:
        finish(job_id, "failed", str(exc), downloaded_bytes=source.used)
    except Exception:  # noqa: BLE001
        print(traceback.format_exc())
        finish(job_id, "failed", "Something went wrong while processing this video. Please try again.",
               downloaded_bytes=source.used)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def check_proxy() -> None:
    """Report whether the optional download proxy actually works, at startup.

    A typo or dead proxy otherwise shows up as a confusing "YouTube is
    blocking" error on every job. This prints the truth in the logs instead.
    """
    proxy = os.environ.get("DOWNLOAD_PROXY") or os.environ.get("PROXY_URL")
    if not proxy:
        print("download proxy: none set, downloading directly")
        return
    shown = proxy.split("@")[-1] if "@" in proxy else proxy
    try:
        r = requests.get(
            "https://www.youtube.com/generate_204",
            proxies={"http": proxy, "https": proxy},
            timeout=20,
        )
        print(f"download proxy {shown}: reachable (HTTP {r.status_code})")
    except Exception as exc:  # noqa: BLE001
        print(f"download proxy {shown}: FAILED - {exc}. Check DOWNLOAD_PROXY.")


def main():
    if not SECRET:
        raise SystemExit("CLIP_WORKER_SECRET is not set.")
    print(f"worker {WORKER_ID} polling {APP_URL} every {POLL_SECONDS}s")
    # Ephemeral disk: clear anything a previous container left behind.
    importer.cleanup_orphans()
    check_proxy()

    while True:
        # Link imports are short, so they get served before render jobs.
        try:
            import_job = importer.claim_import()
        except Exception as exc:  # noqa: BLE001
            print(f"import claim failed: {exc}")
            import_job = None
        if import_job:
            print(f"claimed import {import_job['id']}")
            importer.process_import(import_job)
            continue

        try:
            job = claim()
        except Exception as exc:  # noqa: BLE001
            print(f"claim failed: {exc}")
            time.sleep(POLL_SECONDS)
            continue
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        print(f"claimed job {job['id']} ({job['platform']})")
        process(job)


if __name__ == "__main__":
    main()
