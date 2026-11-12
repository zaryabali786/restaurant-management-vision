"""
The dining room: which tables are taken, by how many, and whether there is food on them.

A table is a ZONE: a rectangle drawn once over the table and its seats on the camera's picture (the app's CCTV page does
this, and offers the tables the detector itself can see as a starting point). Each frame, people and tableware are
detected; a person counts for a zone when their seat — a point a third of the way up their body — falls inside it.

A waiter walking past must not "seat" a table and a guest leaning out of frame must not "empty" it, so nothing is decided
on one frame:

    seated   people seen in the zone in most of the last `on_seconds`
    left     nobody seen in the zone for `off_seconds` in a row
    food     more dishes/cups on the table than it carries when empty (its own baseline: this restaurant keeps a stove
             and sauce bottles on every table, so "something is on the table" would always be true)
"""
import threading
import time
from collections import deque

from ultralytics import YOLO

PERSON = "person"
TABLE = "dining table"
# What reads as a meal on a table. Bottles are left out: sauce and oil bottles live on the tables all day.
FOOD = {"bowl", "cup", "wine glass", "fork", "knife", "spoon", "pizza", "sandwich", "hot dog", "cake", "donut", "banana", "apple", "orange", "broccoli", "carrot"}


def _inside(x, y, zone, grow=0.0):
    gx, gy = zone["w"] * grow, zone["h"] * grow
    return zone["x"] - gx <= x <= zone["x"] + zone["w"] + gx and zone["y"] - gy <= y <= zone["y"] + zone["h"] + gy


class DiningDetector:
    # Every dining camera in one worker looks through the SAME loaded model: three cameras cost the memory of one. A look
    # is one camera at a time (the lock); they would only be fighting over the same CPU cores otherwise.
    _models = {}
    _lock = threading.Lock()

    def __init__(self, weights="models/yolo11s.pt", size=1280):
        with DiningDetector._lock:
            if weights not in DiningDetector._models:
                DiningDetector._models[weights] = YOLO(weights)
        self.model = DiningDetector._models[weights]
        self.size = size  # the long side the picture is looked at: larger sees small, far-off people; smaller is faster
        self.names = self.model.names

    def detect(self, frame):
        """People, tables and tableware, in fractions of the picture (0..1) so zones do not depend on the resolution."""
        h, w = frame.shape[:2]
        with DiningDetector._lock:
            result = self.model.predict(frame, conf=0.25, imgsz=self.size, verbose=False)[0]
        people, tables, food = [], [], []
        for cls, conf, (x1, y1, x2, y2) in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            name = self.names[int(cls)]
            box = {"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h, "conf": round(conf, 2)}
            if name == PERSON and conf >= 0.35:
                people.append(box)
            elif name == TABLE and conf >= 0.3:
                tables.append(box)
            elif name in FOOD:
                food.append({**box, "name": name})
        return {"people": people, "tables": tables, "food": food, "plot": result}


class ZoneState:
    def __init__(self, zone, interval, on_seconds=12, off_seconds=45):
        self.zone = zone
        self.on = deque(maxlen=max(3, round(on_seconds / interval)))
        self.off_needed = max(3, round(off_seconds / interval))
        self.empty_run = 0
        self.occupied = False
        self.persons = 0
        self.food = False
        self.food_run = 0
        self.baseline = None  # dishes seen on this table while it is empty
        self.since = time.time()

    def update(self, persons, dishes):
        """Feeds one frame's counts; returns the events it caused: ('seated' | 'left' | 'food', details)."""
        events = []
        self.on.append(persons > 0)
        self.empty_run = 0 if persons > 0 else self.empty_run + 1
        if not self.occupied:
            # What the bare table carries is learned while nobody is at it (the lowest count seen, which ignores a passing tray).
            self.baseline = dishes if self.baseline is None else min(self.baseline, dishes) if dishes < self.baseline else round(self.baseline * 0.9 + dishes * 0.1)
            if len(self.on) == self.on.maxlen and sum(self.on) >= 0.7 * len(self.on):
                self.occupied, self.persons, self.food, self.food_run, self.since = True, max(1, persons), False, 0, time.time()
                events.append(("seated", {"persons": self.persons}))
        else:
            self.persons = max(self.persons, persons)
            self.food_run = self.food_run + 1 if dishes >= (self.baseline or 0) + 2 else 0
            if not self.food and self.food_run >= 3:
                self.food = True
                events.append(("food", {}))
            if self.empty_run >= self.off_needed:
                stayed = time.time() - self.since
                self.occupied, self.since = False, time.time()
                self.on.clear()
                events.append(("left", {"persons": self.persons, "stayedSeconds": round(stayed)}))
                self.persons, self.food = 0, False
        return events


class DiningWatcher:
    def __init__(self, zones, interval=2.0):
        self.interval = interval
        self.states = {}
        self.set_zones(zones)

    def set_zones(self, zones):
        """Zones can be redrawn while the worker runs: a zone that is unchanged keeps what is known about it."""
        fresh = {}
        for zone in zones or []:
            old = self.states.get(zone["id"])
            if old and all(abs(old.zone[k] - zone[k]) < 1e-6 for k in ("x", "y", "w", "h")):
                old.zone = zone
                fresh[zone["id"]] = old
            else:
                fresh[zone["id"]] = ZoneState(zone, self.interval)
        self.states = fresh

    def update(self, detection):
        """One frame in; the events of every zone out, as `(zone, kind, details)`."""
        out = []
        for state in self.states.values():
            zone = state.zone
            persons = sum(1 for p in detection["people"] if _inside(p["x"] + p["w"] / 2, p["y"] + p["h"] * 0.67, zone, grow=0.04))
            dishes = sum(1 for f in detection["food"] if _inside(f["x"] + f["w"] / 2, f["y"] + f["h"] / 2, zone))
            for kind, details in state.update(persons, dishes):
                out.append((zone, kind, details))
        return out

    def summary(self, detection):
        return {
            "people": len(detection["people"]),
            "occupied": sum(1 for s in self.states.values() if s.occupied),
            "tables": [{"zoneId": s.zone["id"], "occupied": s.occupied, "persons": s.persons, "food": s.food, "sinceSeconds": round(time.time() - s.since)} for s in self.states.values()],
        }


def suggest_zones(tables, grow=0.22):
    """The tables the detector can see, widened to take in their seats: a starting point for whoever draws the zones."""
    zones = []
    for index, t in enumerate(sorted(tables, key=lambda b: (round(b["y"], 1), b["x"]))):
        gx, gy = t["w"] * grow, t["h"] * grow
        x, y = max(0.0, t["x"] - gx), max(0.0, t["y"] - gy)
        zones.append({"id": f"s{index + 1}", "name": f"Table {index + 1}", "x": round(x, 4), "y": round(y, 4), "w": round(min(1 - x, t["w"] + 2 * gx), 4), "h": round(min(1 - y, t["h"] + 2 * gy), 4)})
    return zones
