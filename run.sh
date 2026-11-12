#!/bin/bash
# Starts the camera worker on a server (pm2 runs this). The worker key and the API address come from .env beside
# this file, which is never in git:
#   VISION_KEY=vk_...                          (Cameras page of the restaurant's company)
#   VISION_API=http://127.0.0.1:4200/api       (the WhatsMeal API on this same machine)
#   VISION_PROFILE=fast   VISION_THREADS=2     (optional; these are the defaults here)
cd "$(dirname "$0")"
set -a
[ -f .env ] && . ./.env
set +a
if [ -z "$VISION_KEY" ]; then
  echo "No VISION_KEY in $(pwd)/.env: nothing to watch. Add the key from the Cameras page and restart."
  sleep 3600
  exit 0
fi
export VISION_API="${VISION_API:-http://127.0.0.1:4200/api}"
# Two threads by default: this machine also serves three web apps and a database.
exec .venv/bin/python -u worker.py --profile "${VISION_PROFILE:-fast}" --threads "${VISION_THREADS:-2}"
