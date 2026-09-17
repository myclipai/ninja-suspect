"""
Vid Kraken — the paid YouTube download service.

It replaces the metered residential proxy for YouTube. We ask it for metadata,
for audio only, or for a trimmed slice of picture, and it hands back a CDN link
we then pull straight down (no proxy, no cookies, no yt-dlp bot checks).

Set VIDKRAKEN_API_KEY on the worker to turn it on. With no key, callers fall
back to yt-dlp exactly as before.
"""

from __future__ import annotations

import math
import os
import time

import requests

BASE = os.environ.get("VIDKRAKEN_API_URL", "https://vidkraken.com/api/v2").rstrip("/")
QUALITY = os.environ.get("VIDKRAKEN_FORMAT", "1080")
POLL_SECONDS = float(os.environ.get("VIDKRAKEN_POLL_SECONDS", 3))
JOB_TIMEOUT = float(os.environ.get("VIDKRAKEN_TIMEOUT_SECONDS", 1800))

DONE = {"COMPLETED", "COMPLETE", "SUCCESS", "SUCCEEDED", "DONE", "READY", "FINISHED"}
BROKEN = {"FAILED", "ERROR", "CANCELLED", "CANCELED"}


class VidKrakenError(Exception):
    """Something Vid Kraken couldn't do; the caller may fall back to yt-dlp."""


def key() -> str:
    return os.environ.get("VIDKRAKEN_API_KEY", "").strip()


def enabled() -> bool:
    return bool(key())


def _headers() -> dict:
    return {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"}


def _message(payload: dict, fallback: str) -> str:
    for field in ("error", "message", "errorMessage", "failureReason"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _post(path: str, body: dict) -> dict:
    try:
        resp = requests.post(f"{BASE}/{path}", headers=_headers(), json=body, timeout=60)
    except requests.RequestException as exc:  # network trouble
        raise VidKrakenError(f"Vid Kraken is unreachable: {exc}") from exc
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not resp.ok:
        raise VidKrakenError(_message(payload, f"Vid Kraken returned HTTP {resp.status_code}"))
    return payload


def _poll(kind: str, job_id: str) -> dict:
    deadline = time.time() + JOB_TIMEOUT
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE}/{kind}/{job_id}", headers=_headers(), timeout=60)
        except requests.RequestException as exc:
            raise VidKrakenError(f"Vid Kraken is unreachable: {exc}") from exc
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if not resp.ok:
            raise VidKrakenError(_message(payload, f"Vid Kraken returned HTTP {resp.status_code}"))
        status = str(payload.get("status") or "").upper()
        if status in DONE:
            return payload
        if status in BROKEN:
            raise VidKrakenError(_message(payload, "Vid Kraken could not fetch that video."))
        time.sleep(POLL_SECONDS)
    raise VidKrakenError("Vid Kraken took too long to prepare that video.")


def info(url: str) -> dict:
    """Title and length, without paying for any picture."""
    started = _post("info", {"url": url})
    job_id = started.get("jobId")
    data = _poll("info", job_id) if job_id else started
    duration = data.get("duration") or data.get("lengthSeconds") or data.get("durationSeconds")
    return {
        "title": data.get("title"),
        "duration": float(duration or 0),
        "raw": data,
    }


def _link(payload: dict) -> str:
    for field in ("downloadUrl", "url", "fileUrl", "cdnUrl", "link"):
        value = payload.get(field)
        if isinstance(value, str) and value.startswith("http"):
            return value
    raise VidKrakenError("Vid Kraken finished but returned no download link.")


def fetch(
    url: str,
    fmt: str,
    path: str,
    start: float | None = None,
    end: float | None = None,
    on_bytes=None,
) -> int:
    """Downloads a whole video, a trimmed slice, or the audio, into `path`.

    Returns the number of bytes written. `on_bytes` is called with each chunk's
    size so the caller can keep its own running total.
    """
    body: dict = {"url": url, "format": fmt}
    if start is not None and end is not None:
        lo = max(0, int(math.floor(start)))
        hi = max(lo + 1, int(math.ceil(end)))
        body["startTime"] = lo
        body["endTime"] = hi

    started = _post("download", body)
    job_id = started.get("jobId")
    done = _poll("download", job_id) if job_id else started
    link = _link(done)

    total = 0
    try:
        with requests.get(link, stream=True, timeout=1800) as resp:
            if not resp.ok:
                raise VidKrakenError(f"The prepared file couldn't be read (HTTP {resp.status_code}).")
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
                    if on_bytes:
                        on_bytes(len(chunk))
    except requests.RequestException as exc:
        raise VidKrakenError(f"The prepared file couldn't be downloaded: {exc}") from exc
    if total == 0:
        raise VidKrakenError("The prepared file was empty.")
    return total


def status_line() -> str:
    """One line for the startup log."""
    if not enabled():
        return "vidkraken: no API key set, falling back to yt-dlp for YouTube"
    try:
        resp = requests.get(f"{BASE}/me", headers=_headers(), timeout=30)
        if resp.ok:
            return f"vidkraken: connected ({resp.text[:200]})"
        return f"vidkraken: key rejected (HTTP {resp.status_code}). Check VIDKRAKEN_API_KEY."
    except requests.RequestException as exc:
        return f"vidkraken: unreachable - {exc}"
