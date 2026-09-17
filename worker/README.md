# MyClip AI render worker

<!-- sync trigger: 2026-09-13 -->

This is the machine that actually makes the videos. It polls the app for jobs,
downloads the source (YouTube, Twitch, Kick or an uploaded file), transcribes
it, asks the AI which moments are worth clipping, reframes each moment to
1080×1920, burns captions, renders real MP4s with FFmpeg and uploads them back.

A job is only marked **Completed** once at least one MP4 exists in storage.
Nothing here fakes progress.

## Settings

| Variable | Required | What it does |
| --- | --- | --- |
| `APP_URL` | yes | Base URL of the app, e.g. `https://myclipai.com` |
| `CLIP_WORKER_SECRET` | yes | Shared secret; must match the app's secret |
| `LOVABLE_API_KEY` | no | Only if you call the AI directly; the app normally does it |
| `VIDKRAKEN_API_KEY` | recommended | Vid Kraken key; handles all YouTube downloads (metadata, audio, trimmed windows) |
| `VIDKRAKEN_FORMAT` | no | Picture quality asked of Vid Kraken: `1080` (default), `720`, `480`, `360` |
| `DOWNLOAD_PROXY` | no | Only used when no Vid Kraken key is set; residential proxy for yt-dlp |
| `YOUTUBE_COOKIES_B64` | no | Base64 of a Netscape cookies file; only used by the yt-dlp fallback |
| `WHISPER_MODEL` | no | `tiny`…`large-v3` (default `small`) |
| `MAX_SOURCE_SECONDS` | no | Longest source video accepted (default 5400 = 90 minutes) |
| `MAX_AUDIO_BYTES` | no | Stops a sound-track download once it passes this size (default 160 MB) |
| `POLL_SECONDS` | no | Seconds between polls when idle (default 5) |

## Run it on Railway (recommended)

Railway keeps the machine awake for long renders, which is what this workload
needs.

Repository: `https://github.com/myclipai/ninja-suspect.git`

1. Open your Railway project (`8228b295-32de-4813-a851-09d66927b3eb`).
2. Click **Add a service** → **GitHub** → paste or select
   `myclipai/ninja-suspect`.
3. Set the service **Root Directory** to `worker`. Railway picks up
   `worker/railway.json` and builds `worker/Dockerfile`.
4. Add these **Variables**:
   - `APP_URL=https://myclipai.com`
   - `CLIP_WORKER_SECRET` (must match the app's `CLIP_WORKER_SECRET`)
   - `YOUTUBE_COOKIES_B64` (optional but strongly recommended for YouTube)
   - `WHISPER_MODEL=small`
5. Deploy. Watch the logs for a line like `worker <id> polling https://myclipai.com`.

Scaling: one replica handles one video at a time. Raise `numReplicas` in
`worker/railway.json` to render several jobs in parallel — job claiming is
lease-based, so replicas never take the same job.

Railway settings live only in `worker/railway.json`. Fly.io settings live only
in `worker/fly.toml`. Neither the app, the database nor the storage layout know
which host is in use, so moving between them changes nothing else.

## Run it on Fly.io

```bash
cd worker
fly launch --no-deploy
fly secrets set APP_URL=... CLIP_WORKER_SECRET=... YOUTUBE_COOKIES_B64=...
fly deploy
```

Fly's free trial force-stops machines after about five minutes, which kills
long renders. Add a card, or use Railway.

## Run it locally

```bash
cd worker
pip install -r requirements.txt        # needs ffmpeg + ffprobe on your PATH
export APP_URL=https://myclipai.com
export CLIP_WORKER_SECRET=...
python worker.py
```

## Housekeeping

The app exposes `POST /api/public/hooks/cleanup-jobs`, which fails jobs whose
worker vanished and deletes source files past their retention window. Schedule
it every 15 minutes.
