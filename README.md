# Camera worker

Reads the restaurant's CCTV cameras and reports to the app: which tables are taken, when guests sit and leave, whether
food is on the table, and — in the kitchen — anyone working without a cap or gloves, with a close-up.

## Run it

On a computer in the restaurant that can reach the cameras (Windows, Python 3.11):

```
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python worker.py --api https://YOUR-SITE/api --key vk_...
```

The key and the exact command are on the app's **Cameras** page. Cameras, and the table zones drawn on them, are set up
there too; the worker picks changes up within half a minute. The models (about 60 MB, plus CLIP 340 MB) download on the
first run into `models/` and `weights/`.

Try one camera with no app at all — it prints what it would report and saves marked-up pictures in `samples/`:

```
.venv\Scripts\python worker.py --dry dining  "rtsp://user:pass@192.168.1.20:554/stream1" --seconds 60
.venv\Scripts\python worker.py --dry kitchen "https://www.youtube.com/watch?v=..." --seconds 60
```

## What reads what

| File | Job |
|---|---|
| `streams.py` | Opens any camera address: RTSP (over TCP), HTTP/HTTPS video (MJPEG, HLS), an HTTP/HTTPS still-picture address (fetched each look; name and password in the address are used; self-signed certificates accepted), or a YouTube live link. Reopens it when it drops. `test_streams.py` checks all of them; `probe_rtsp.py <address>` says whether one RTSP camera is giving a picture |
| `dining.py` | YOLO11 people + tableware; per-table zones; seated / food / left with debouncing |
| `hygiene.py` | Pose model finds heads and hands; CLIP judges the crown (cap or not); hand colour judges gloves |
| `worker.py` | One thread per camera; talks to the app (`/api/vision/worker/*`) |

## How many cameras one computer can watch

It runs on CPU. By default a dining camera is looked at every 5 seconds and a kitchen camera every 2.5; a picture in
which nothing moved is not looked at again (for up to 20 seconds), so a quiet room costs almost nothing.

Measured on a 4-core laptop (Intel 8th-gen U, 4 threads), busy picture, no motion skipping:

| Profile | Dining look | Kitchen look | One restaurant (3 dining + 1 kitchen camera) uses |
|---|---|---|---|
| `accurate` (default) | 0.50 s | 0.54 s | about half of that computer |
| `fast` (`--profile fast`) | 0.13 s | 0.25 s | about a fifth |

`fast` found one person fewer per dining frame (8, 10, 9 against 9, 11, 10) on the sample camera; the kitchen judgements
were the same. Memory: about 1.3 GB for one dining and one kitchen camera; dining cameras share one model, each kitchen
camera adds its own.

Run `python bench.py` (add `--threads 6` to match a 6-core machine) on the computer that will do the watching: the
figures are about that machine and no other. `--dining-interval`, `--kitchen-interval` and `--threads` set the pace.
A GPU makes it faster, not more accurate.

## Known limits

- Table zones must be drawn by hand for each camera; the "detected tables" are only a starting point.
- Food on the table is judged against what the bare table normally carries, so permanent stoves and bottles do not count.
- Gloves are judged by colour: a clear or skin-coloured glove reads as a bare hand.
- The cap check is zero-shot (no training on this kitchen). On the sample kitchen it raised no false alarm, and on bare
  heads it was sure only about one time in five — it under-reports rather than accuse. Verdicts marked in the app
  (Correct / Wrong) are the data a trained check would be built from.

## On the WhatsMeal VPS

The worker also runs on the VPS, in `/var/www/whatsmeal-vision`, as the pm2 process `whatsmeal-vision` (`run.sh`, `ecosystem.config.cjs`). A push to `main` copies the code there and restarts it (`.github/workflows/deploy.yml`; repository secrets `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`).

- Its settings are in `/var/www/whatsmeal-vision/.env`, never in git: `VISION_KEY=vk_...` (from the Cameras page) and, if wanted, `VISION_API`, `VISION_PROFILE` (default `fast`), `VISION_THREADS` (default 2).
- One key is one restaurant. With no key the process idles and says so in its log.
- From the VPS it can only reach cameras with a public address (RTSP/HTTP/HTTPS or a live link). A camera on `192.168.x.x` inside a restaurant needs port forwarding or a VPN, or the worker on a PC there.
- The VPS has 4 cores shared with three web apps and a database: the `fast` profile on 2 threads handles about 4 to 6 cameras at the default intervals.
- `pm2 logs whatsmeal-vision --lines 50` shows what it is doing.
