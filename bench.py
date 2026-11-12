"""
How fast is this computer, and what does the "fast" profile give up?

    python bench.py [--threads 6]

Runs both kinds of look (a dining frame, a kitchen frame) in both profiles on the saved sample frames, and prints the
seconds one look takes and how many cameras that allows at the default pace. Run it on the computer that will do the
watching: the answer is about that machine and no other.
"""
import argparse
import glob
import time

import cv2
import torch

from worker import INTERVALS, PROFILES

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--rounds", type=int, default=3)
args = parser.parse_args()
if args.threads > 0:
    torch.set_num_threads(args.threads)

dining_frames = [cv2.imread(p) for p in sorted(glob.glob("samples/dining_[0-9].jpg"))]
kitchen_frames = [cv2.imread(p) for p in sorted(glob.glob("samples/kitchen_[0-9].jpg"))]
if not dining_frames or not kitchen_frames:
    raise SystemExit("No sample frames in samples/. Save a few with a dry run first.")
print(f"threads: {torch.get_num_threads()}   frames: {len(dining_frames)} dining, {len(kitchen_frames)} kitchen")

from dining import DiningDetector
from hygiene import HygieneChecker

results = {}
for profile, spec in PROFILES.items():
    detector = DiningDetector(*spec["dining"])
    detector.detect(dining_frames[0])  # the first look loads the model; it is not counted
    started = time.time()
    people = []
    for _ in range(args.rounds):
        people = [len(detector.detect(f)["people"]) for f in dining_frames]
    dining = (time.time() - started) / (args.rounds * len(dining_frames))

    checker = HygieneChecker(pose_weights=spec["kitchen"][0], size=spec["kitchen"][1])
    checker.check(kitchen_frames[0])
    started = time.time()
    judged = []
    for _ in range(args.rounds):
        judged = [sum(1 for p in checker.check(f) if p.head_covered is not None or p.gloves is not None) for f in kitchen_frames]
    kitchen = (time.time() - started) / (args.rounds * len(kitchen_frames))
    results[profile] = (dining, kitchen, people, judged)
    print(f"\n{profile}")
    print(f"  dining look   {dining:5.2f} s   people found per frame: {people}")
    print(f"  kitchen look  {kitchen:5.2f} s   people judged per frame: {judged}")
    # One look per interval per camera, all cores busy: how many cameras before the computer cannot keep up.
    print(f"  keeps up with about {INTERVALS['dining'] / dining:4.1f} dining cameras (a look every {INTERVALS['dining']:g}s), or {INTERVALS['kitchen'] / kitchen:4.1f} kitchen cameras (every {INTERVALS['kitchen']:g}s)")
    mixed = 1 / (3 * dining / INTERVALS["dining"] + kitchen / INTERVALS["kitchen"])
    print(f"  a restaurant with 3 dining cameras + 1 kitchen camera uses {100 / mixed:3.0f}% of this computer (room for {mixed:3.1f} such restaurants)")
