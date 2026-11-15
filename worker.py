"""
The camera worker: watches the restaurant's cameras and tells the app what it sees.

    python worker.py --api http://localhost:4000/api --key <the key shown on the app's CCTV page>

It asks the app which cameras to watch (and re-asks every half minute, so a camera or a table zone changed in the app
is picked up without restarting), then for each camera, every couple of seconds:

    dining camera    people and tableware are detected; each table zone is followed, and "seated", "food on the table"
                     and "left" are sent as they happen, with a head-count every minute for the busy-hours chart
    kitchen camera   each person's head covering and gloves are checked; a breach seen several times running is sent
                     with a close-up of that person

A still picture goes up every half minute so the app can show the camera and let the table zones be drawn on it.

To try a camera with no app at all (prints what it would send, and saves marked-up pictures in samples/):

    python worker.py --dry dining  <camera address> --seconds 60
    python worker.py --dry kitchen <camera address> --seconds 60
"""
import argparse
import base64
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import requests

from streams import Camera

SNAPSHOT_EVERY = 30

# How hard to look. "accurate" is what the module was tested with; "fast" trades a little sight (small, far-off people)
# for speed, for a slow computer or many cameras. Measured numbers for both are in the README.
PROFILES = {
    "accurate": {"dining": ("models/yolo11s.pt", 1280), "kitchen": ("models/yolo11s-pose.pt", 960)},
    "fast": {"dining": ("models/yolo11n.pt", 960), "kitchen": ("models/yolo11n-pose.pt", 640)},
}
# Seconds between looks. A table changes slowly; a hand in the kitchen does not.
INTERVALS = {"dining": 5.0, "kitchen": 2.5}


class Stillness:
    """
    Says whether the picture has changed since it was last looked at properly.

    An empty dining room at four in the afternoon is the same picture over and over, and running a model on it is the
    whole cost of this program. Two small grey copies are compared instead (a fraction of a millisecond): if almost no
    pixels moved, the last answer still stands. It is never trusted for longer than `max_still` seconds, so a slow
    change (a guest sitting very still) is still seen.
    """

    def __init__(self, share=0.004, max_still=20.0):
        self.share, self.max_still = share, max_still
        self.last = None
        self.looked_at = 0.0

    def changed(self, frame):
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (192, 108)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        now = time.time()
        if self.last is None or now - self.looked_at >= self.max_still:
            self.last, self.looked_at = small, now
            return True
        moved = float((cv2.absdiff(small, self.last) > 18).mean())
        if moved >= self.share:
            self.last, self.looked_at = small, now
            return True
        return False
TICK_EVERY = 60
CONFIG_EVERY = 30


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jpeg_b64(frame, width=960, quality=78):
    if frame.shape[1] > width:
        frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(data.tobytes()).decode() if ok else ""


class Api:
    """The app's door for the worker. With `dry`, nothing is sent: it is printed instead."""

    def __init__(self, base, key, dry=False):
        self.base, self.key, self.dry = (base or "").rstrip("/"), key, dry

    def config(self):
        r = requests.get(f"{self.base}/vision/worker/config", headers={"x-vision-key": self.key}, timeout=15)
        r.raise_for_status()
        return r.json()

    def send(self, path, body):
        if self.dry:
            shown = {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v) for k, v in body.items()}
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {path}: {shown}")
            return True
        for attempt in range(3):
            try:
                r = requests.post(f"{self.base}/vision/worker/{path}", json=body, headers={"x-vision-key": self.key}, timeout=20)
                if r.status_code < 500:
                    return r.ok
            except requests.RequestException as error:
                print(f"[worker] could not reach the app ({error.__class__.__name__}); trying again")
            time.sleep(1.5 * (attempt + 1))
        return False


class DiningJob:
    def __init__(self, camera, api, interval, save_marked=False, profile="accurate"):
        from dining import DiningDetector, DiningWatcher, suggest_zones

        self.camera, self.api, self.interval, self.save_marked = camera, api, interval, save_marked
        self.detector = DiningDetector(*PROFILES[profile]["dining"])
        self.still = Stillness()
        self.detection = None
        self.skipped = 0
        self.watcher = DiningWatcher(camera.get("zones") or [], interval)
        self.suggest = suggest_zones
        self.last_tick = self.last_snapshot = 0.0
        self.frames = 0

    def zones(self, zones):
        self.watcher.set_zones(zones)

    def step(self, frame):
        # Nothing moved since the last proper look: what was seen then still stands (and still counts toward the
        # "seated for long enough" and "empty for long enough" clocks).
        if self.detection is None or self.still.changed(frame):
            self.detection = self.detector.detect(frame)
        else:
            self.skipped += 1
        detection = self.detection
        self.frames += 1
        # With no zones drawn yet, the tables the detector can see stand in for them, so a new camera shows something at once.
        if not self.watcher.states and detection["tables"] and self.api.dry:
            self.watcher.set_zones(self.suggest(detection["tables"]))
        events = [{"kind": kind, "zoneId": zone["id"], "tableId": zone.get("tableId", ""), "name": zone.get("name", ""), "at": now_iso(), **details} for zone, kind, details in self.watcher.update(detection)]
        if events:
            self.api.send("events", {"cameraId": self.camera["id"], "events": events})
        now = time.time()
        if now - self.last_tick >= TICK_EVERY:
            self.last_tick = now
            self.api.send("tick", {"cameraId": self.camera["id"], "at": now_iso(), **self.watcher.summary(detection)})
        if now - self.last_snapshot >= SNAPSHOT_EVERY:
            self.last_snapshot = now
            self.api.send("snapshot", {"cameraId": self.camera["id"], "at": now_iso(), "jpeg": jpeg_b64(frame), "suggestions": self.suggest(detection["tables"]), "people": len(detection["people"])})
        if self.save_marked and self.frames % 10 == 1:
            marked = detection["plot"].plot(line_width=2, font_size=10)
            h, w = marked.shape[:2]
            for state in self.watcher.states.values():
                z = state.zone
                colour = (40, 40, 220) if state.occupied else (70, 170, 70)
                cv2.rectangle(marked, (int(z["x"] * w), int(z["y"] * h)), (int((z["x"] + z["w"]) * w), int((z["y"] + z["h"]) * h)), colour, 3)
                cv2.putText(marked, f"{z.get('name', z['id'])}: {'TAKEN ' + str(state.persons) if state.occupied else 'free'}{' +food' if state.food else ''}", (int(z["x"] * w) + 4, int(z["y"] * h) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
            cv2.imwrite(f"samples/dry_dining_{self.frames:04d}.jpg", marked)


class KitchenJob:
    def __init__(self, camera, api, interval, save_marked=False, profile="accurate"):
        from hygiene import HygieneChecker, KitchenWatcher, close_up, draw

        self.camera, self.api, self.interval, self.save_marked = camera, api, interval, save_marked
        self.checker = HygieneChecker(pose_weights=PROFILES[profile]["kitchen"][0], size=PROFILES[profile]["kitchen"][1])
        self.still = Stillness()
        self.skipped = 0
        self.watcher = KitchenWatcher()
        self.close_up, self.draw = close_up, draw
        self.last_snapshot = 0.0
        self.frames = 0

    def zones(self, _zones):
        pass

    def step(self, frame):
        now = time.time()
        # A kitchen where nothing moved has nothing new to judge; the picture still goes up on time.
        if not self.still.changed(frame):
            self.skipped += 1
            if now - self.last_snapshot >= SNAPSHOT_EVERY:
                self.last_snapshot = now
                self.api.send("snapshot", {"cameraId": self.camera["id"], "at": now_iso(), "jpeg": jpeg_b64(frame), "people": 0})
            return
        people = self.checker.check(frame)
        self.frames += 1
        for kind, person in self.watcher.update(people, now):
            face = self.close_up(frame, person)
            self.api.send("hygiene", {"cameraId": self.camera["id"], "at": now_iso(), "kind": kind, "track": int(person.track), "score": round(1 - person.head_score, 2) if kind == "no_cap" else 1.0, "image": jpeg_b64(face, width=420, quality=85) if face is not None else ""})
            if self.save_marked and face is not None:
                cv2.imwrite(f"samples/dry_alert_{kind}_{self.frames:04d}.jpg", face)
        if now - self.last_snapshot >= SNAPSHOT_EVERY:
            self.last_snapshot = now
            self.api.send("snapshot", {"cameraId": self.camera["id"], "at": now_iso(), "jpeg": jpeg_b64(frame), "people": len(people)})
        if self.save_marked and self.frames % 6 == 1:
            cv2.imwrite(f"samples/dry_kitchen_{self.frames:04d}.jpg", self.draw(frame, people))


def watch(camera, api, intervals, stop, save_marked=False, profile="accurate"):
    """One camera, until told to stop. A camera that drops is reopened; one that will not open is tried again each minute."""
    interval = intervals["kitchen" if camera["kind"] == "kitchen" else "dining"]
    job = None
    while job is None and not stop.is_set():  # a model that fails to load is retried, never given up on
        try:
            job = (KitchenJob if camera["kind"] == "kitchen" else DiningJob)(camera, api, interval, save_marked, profile)
        except Exception as error:
            print(f"[worker] {camera.get('name')}: could not start ({error.__class__.__name__}: {error}); retrying in 30 s")
            stop.wait(30)
    if job is None:
        return
    stream = Camera(camera["url"])
    camera["_job"] = job
    print(f"[worker] watching {camera.get('name') or camera['id']} ({camera['kind']}): a look every {interval:g}s, {profile}")
    misses = 0
    while not stop.is_set():
        started = time.time()
        camera["_alive"] = started  # the supervisor in main() restarts a watch that stops ticking (a hung connection)
        try:
            frame = stream.read(skip=int(interval * 25))  # jump to the present: a live stream keeps buffering while a frame is thought about
            if frame is None:
                misses += 1
                # The stream dropped: reopen it at once and keep trying (a few seconds apart, then every 30 s), for as long as the camera is on.
                print(f"[worker] {camera.get('name')}: no picture; reopening")
                stream.close()
                stop.wait(min(30, 3 * misses))
                continue
            misses = 0
            job.step(frame)
        except Exception as error:  # one bad frame or a dropped connection must not end the watch
            print(f"[worker] {camera.get('name')}: {error.__class__.__name__}: {error}")
            stop.wait(5)
        stop.wait(max(0.0, interval - (time.time() - started)))
    stream.close()


def main():
    parser = argparse.ArgumentParser(description="Watches the restaurant's cameras and reports to the app.")
    parser.add_argument("--api", default=os.environ.get("VISION_API", "http://localhost:4000/api"), help="the app's API address (or set VISION_API)")
    parser.add_argument("--key", default=os.environ.get("VISION_KEY", ""), help="the worker key from the Cameras page (or set VISION_KEY, so it is not on the command line)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="accurate", help="accurate (default) or fast: quicker, a little less sharp-eyed")
    parser.add_argument("--dining-interval", type=float, default=INTERVALS["dining"], help="seconds between looks at a dining camera")
    parser.add_argument("--kitchen-interval", type=float, default=INTERVALS["kitchen"], help="seconds between looks at a kitchen camera")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads the models may use (0 = all)")
    parser.add_argument("--dry", nargs=2, metavar=("KIND", "URL"), help="try one camera without the app: dining|kitchen and its address")
    parser.add_argument("--seconds", type=int, default=0, help="stop after this long (0 = run until closed)")
    args = parser.parse_args()

    if args.threads > 0:
        import torch

        torch.set_num_threads(args.threads)
    intervals = {"dining": args.dining_interval, "kitchen": args.kitchen_interval}
    stop = threading.Event()
    threads = {}
    if args.dry:
        api = Api("", "", dry=True)
        cameras = [{"id": "dry", "name": "dry run", "kind": args.dry[0], "url": args.dry[1], "zones": []}]
    else:
        if not args.key:
            parser.error("--key is needed: copy it from the app's CCTV page")
        api = Api(args.api, args.key)
        cameras = []

    started = time.time()
    try:
        while not stop.is_set():
            if not args.dry:
                try:
                    cameras = [c for c in api.config().get("cameras", []) if c.get("active", True) and c.get("url")]
                except Exception as error:
                    print(f"[worker] the app did not answer ({error}); trying again")
            for camera in cameras:
                running = threads.get(camera["id"])
                # A watch that has not come round for 3 minutes is hung inside a dead connection: let it go and start a fresh one.
                if running and not running["thread"].is_alive() and not args.dry:
                    print(f"[worker] {camera.get('name')}: its watch had ended; starting it again")
                    running = None
                if running and not args.dry and time.time() - running["camera"].get("_alive", time.time()) > 180:
                    print(f"[worker] {camera.get('name')}: the watch hung; restarting it")
                    running["stop"].set()
                    running = None
                if running and running["url"] == camera["url"] and running["kind"] == camera["kind"]:
                    job = running["camera"].get("_job")
                    # Zones redrawn in the app reach the running watch. (A dry run has none to fetch: it keeps the ones it found.)
                    if job and not args.dry:
                        job.zones(camera.get("zones") or [])
                    continue
                if running:
                    running["stop"].set()
                own = threading.Event()
                thread = threading.Thread(target=watch, args=(camera, api, intervals, own, bool(args.dry), args.profile), daemon=True)
                threads[camera["id"]] = {"thread": thread, "stop": own, "url": camera["url"], "kind": camera["kind"], "camera": camera}
                thread.start()
            for cid in [c for c in threads if c not in {cam["id"] for cam in cameras}]:
                threads.pop(cid)["stop"].set()
            if args.seconds and time.time() - started >= args.seconds:
                break
            stop.wait(CONFIG_EVERY if not args.seconds else min(CONFIG_EVERY, max(1, args.seconds - (time.time() - started))))
    except KeyboardInterrupt:
        pass
    for running in threads.values():
        running["stop"].set()
    time.sleep(1)


if __name__ == "__main__":
    main()
