"""
Kitchen hygiene: is each person's head covered, and are their hands gloved?

Asking one detector for "a glove" somewhere in a whole kitchen frame was tried first and was wrong more often than right.
What works on real footage is to look only at the piece that matters:

  1. A pose model finds every person and where their eyes, ears, shoulders, elbows and wrists are.
  2. THE CROWN — from just above the eyes to well above the head, so a tall hat is inside it — is shown to CLIP with
     one plain question:
     "the top of a head covered by a hat or cap" or "the top of a head with hair, no hat"?
  3. EACH HAND — a small square just past the wrist — is judged by its colour: a bare hand is mostly skin tone; a
     kitchen glove (blue, white, black) is not.

Two honest limits, both stated on the screen that shows the alerts:
  - a transparent or skin-coloured glove cannot be told from a bare hand by colour;
  - anything too small, blurred or half out of the frame is left unjudged. An unsure answer never raises an alarm, and
    an alarm needs the same person seen in breach several times running (see KitchenWatcher).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# One plain question. Longer lists of phrasings were tried on the sample kitchen and raised more false alarms, not fewer.
CROWN_PROMPTS = ["the top of a head covered by a hat or cap", "the top of a head with black hair, no hat"]

# COCO keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = range(11)


@dataclass
class Person:
    box: tuple  # x1, y1, x2, y2
    track: int
    crown_box: tuple | None = None
    head_covered: bool | None = None  # None = could not be judged
    head_score: float = 0.0  # how sure CLIP is that the head is covered, 0..1
    hands: list = field(default_factory=list)  # [(box, gloved | None, skin share)]

    @property
    def gloves(self):
        """False when a hand is plainly bare; True when every judged hand is gloved; None when no hand could be judged."""
        judged = [g for _, g, _ in self.hands if g is not None]
        if not judged:
            return None
        return all(judged)


def _clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def skin_share(crop) -> float:
    """How much of a piece of the picture is skin-coloured (YCrCb, which holds across light and dark skin)."""
    if crop is None or crop.size == 0:
        return 0.0
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    return float(cv2.inRange(ycc, (0, 135, 85), (255, 180, 135)).mean() / 255)


class HygieneChecker:
    def __init__(self, pose_weights="models/yolo11s-pose.pt", device="cpu", sure=0.7, bare=0.2, size=960):
        import clip  # the copy ultralytics installs

        self.device = device
        self.sure = sure
        self.size = size
        self.bare = bare
        self.pose = YOLO(pose_weights)
        self.clip, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip.eval()
        with torch.no_grad():
            text = self.clip.encode_text(clip.tokenize(CROWN_PROMPTS).to(device)).float()
            self.crown_text = text / text.norm(dim=-1, keepdim=True)

    def _covered(self, crops):
        """For each crown, how sure CLIP is that it is covered (0..1)."""
        from PIL import Image

        if not crops:
            return []
        batch = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]).to(self.device)
        with torch.no_grad():
            image = self.clip.encode_image(batch).float()
            image = image / image.norm(dim=-1, keepdim=True)
            return (100.0 * image @ self.crown_text.T).softmax(dim=-1)[:, 0].cpu().tolist()

    @staticmethod
    def _crown_box(box, kpts, conf):
        """From just above the eyes to well above the head, a little wider than it. Needs the face or an ear to be seen."""
        face = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if conf[i] > 0.4]
        if not face:
            return None
        cx = float(np.mean([kpts[i][0] for i in face]))
        eye_y = float(np.mean([kpts[i][1] for i in face]))
        shoulders = [i for i in (L_SHOULDER, R_SHOULDER) if conf[i] > 0.4]
        head = (float(np.mean([kpts[i][1] for i in shoulders])) - eye_y) if shoulders else (box[3] - box[1]) * 0.12
        if head < 30:
            return None
        # Tall enough to take in a chef's hat that sits high on the head, not only the hairline under it.
        return (cx - head * 0.7, eye_y - head * 1.7, cx + head * 0.7, eye_y - head * 0.1)

    @staticmethod
    def _hand_box(kpts, conf, elbow, wrist):
        if conf[wrist] < 0.5 or conf[elbow] < 0.4:
            return None
        ex, ey = kpts[elbow]
        wx, wy = kpts[wrist]
        forearm = float(np.hypot(wx - ex, wy - ey))
        if forearm < 60:  # a hand smaller than this on screen is a smudge, not evidence
            return None
        cx, cy = wx + (wx - ex) * 0.3, wy + (wy - ey) * 0.3
        half = forearm * 0.32
        return (cx - half, cy - half, cx + half, cy + half)

    def check(self, frame, min_person_height=0.2):
        """Everyone in the frame large enough to judge, with what could be told about their head and hands."""
        result = self.pose.track(frame, persist=True, conf=0.35, imgsz=self.size, verbose=False, tracker="bytetrack.yaml")[0]
        people = []
        if result.boxes is None or result.keypoints is None or result.keypoints.conf is None:
            return people
        ids = result.boxes.id.int().tolist() if result.boxes.id is not None else list(range(len(result.boxes)))
        crowns, owners = [], []
        for index, (box, kp, kc) in enumerate(zip(result.boxes.xyxy.tolist(), result.keypoints.xy.tolist(), result.keypoints.conf.tolist())):
            if (box[3] - box[1]) / frame.shape[0] < min_person_height:
                continue
            person = Person(box=tuple(box), track=ids[index])
            people.append(person)
            crown = self._crown_box(box, kp, kc)
            if crown:
                cb = _clip_box(crown, frame.shape)
                if cb[2] - cb[0] >= 36 and cb[3] - cb[1] >= 28:
                    person.crown_box = cb
                    crowns.append(frame[cb[1]:cb[3], cb[0]:cb[2]])
                    owners.append(person)
            for elbow, wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
                hand = self._hand_box(kp, kc, elbow, wrist)
                if not hand:
                    continue
                b = _clip_box(hand, frame.shape)
                if b[2] - b[0] < 36 or b[3] - b[1] < 36:
                    continue
                skin = skin_share(frame[b[1]:b[3], b[0]:b[2]])
                person.hands.append((b, False if skin >= 0.5 else True if skin <= 0.15 else None, skin))

        for person, covered in zip(owners, self._covered(crowns)):
            person.head_score = float(covered)
            # "Bare" must be said more firmly than "covered": it is the answer that puts a photograph in front of the manager.
            person.head_covered = True if covered >= self.sure else False if covered <= self.bare else None
        return people


def close_up(frame, person, size=420):
    """A close picture of the person's head and shoulders, for the alert."""
    x1, y1, x2, y2 = person.box
    w, h = x2 - x1, y2 - y1
    if person.crown_box:
        cx1, cy1, cx2, cy2 = person.crown_box
        cw = cx2 - cx1
        box = (cx1 - cw * 0.55, cy1 - cw * 0.25, cx2 + cw * 0.55, cy2 + cw * 1.45)
    else:
        box = (x1 - w * 0.1, y1 - h * 0.04, x2 + w * 0.1, y1 + h * 0.4)
    bx = _clip_box(box, frame.shape)
    crop = frame[bx[1]:bx[3], bx[0]:bx[2]]
    if crop.size == 0:
        return None
    scale = size / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def draw(frame, people):
    """The frame with what was decided drawn on it, for the eye to check."""
    out = frame.copy()
    GOOD, BAD, UNSURE = (70, 170, 70), (40, 40, 220), (0, 200, 230)
    for p in people:
        cv2.rectangle(out, (int(p.box[0]), int(p.box[1])), (int(p.box[2]), int(p.box[3])), (210, 210, 210), 1)
        if p.crown_box:
            colour = GOOD if p.head_covered else BAD if p.head_covered is False else UNSURE
            cv2.rectangle(out, p.crown_box[:2], p.crown_box[2:], colour, 2)
            cv2.putText(out, f"{'cap' if p.head_covered else 'NO CAP' if p.head_covered is False else 'cap?'} {p.head_score:.2f}", (p.crown_box[0], max(16, p.crown_box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        for b, gloved, skin in p.hands:
            colour = GOOD if gloved else BAD if gloved is False else UNSURE
            cv2.rectangle(out, b[:2], b[2:], colour, 2)
            cv2.putText(out, f"{'glove' if gloved else 'BARE HAND' if gloved is False else 'hand?'} {skin:.2f}", (b[0], max(16, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return out


class KitchenWatcher:
    """
    Turns a stream of per-frame judgements into alerts worth sending.

    One frame is never enough: a head turned away or a blurred hand must not put someone's photograph in front of the
    manager. An alert is raised only when the SAME tracked person has been judged in breach in `need` of their last
    `window` judged frames, and then not again for that person and that kind until `cooldown` seconds have passed.
    """

    def __init__(self, need=3, window=5, cooldown=300):
        self.need, self.window, self.cooldown = need, window, cooldown
        self.history = {}  # (track, kind) -> [bool, ...]
        self.last_alert = {}  # (track, kind) -> time

    def update(self, people, now):
        alerts = []
        for person in people:
            for kind, ok in (("no_cap", person.head_covered), ("no_gloves", person.gloves)):
                if ok is None:
                    continue
                key = (person.track, kind)
                seen = (self.history.get(key, []) + [not ok])[-self.window:]
                self.history[key] = seen
                if sum(seen) >= self.need and now - self.last_alert.get(key, 0) >= self.cooldown:
                    self.last_alert[key] = now
                    alerts.append((kind, person))
        return alerts
