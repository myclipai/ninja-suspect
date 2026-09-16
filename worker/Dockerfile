FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*

# YouTube now serves a JavaScript challenge; yt-dlp solves it with a JS runtime.
ENV DENO_INSTALL=/usr/local
RUN curl -fsSL https://deno.land/install.sh | sh -s -- -y \
 && deno --version

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py importer.py ./

ENV WHISPER_MODEL=small
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
