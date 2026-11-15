"""
Opening a camera, whatever it speaks.

    rtsp://  rtsps://     a CCTV camera, a DVR/NVR channel, a phone running an RTSP app      -> read as video, over TCP
    http://  https://     a video stream (MJPEG, HLS .m3u8, an .mp4 feed)                    -> read as video
                          a still-picture address (…/snapshot.jpg, /shot.jpg, /image.cgi)    -> fetched again for every look
    YouTube live link     resolved to its real stream first                                  -> read as video

Which of the two an http(s) address is, is found out by asking it once and looking at what it says it is sending.
A name and password written in the address (http://user:pass@host/…) are used for both. Cameras on a local network
almost always have a self-signed certificate, so an https address is not refused for that.

A camera that drops is reopened by itself, and a YouTube address — which expires after a few hours — is resolved afresh.
"""
import os
import re
import time
from urllib.parse import urlsplit, urlunsplit

# RTSP over TCP: on Wi-Fi (and through most routers) UDP loses packets and the picture never starts. ffmpeg reads this
# when a stream is opened, so it is set before anything opens one.
# A stream that goes quiet (the network drops, YouTube stops the feed) must not leave ffmpeg waiting for ever: with no
# time limit one read can hang, and the camera then silently stops being watched. 15 s (in microseconds) is allowed.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;15000000|rw_timeout;15000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

YOUTUBE = re.compile(r"(youtube\.com|youtu\.be)", re.I)
STILL_TYPES = ("image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp")
REOPEN_AFTER = 2 * 3600  # a YouTube stream address stops working after a few hours; get a fresh one before that


def resolve(url: str, height: int = 720) -> str:
    """The address OpenCV can open. For YouTube this asks yt-dlp for the HLS stream; anything else is used as it is."""
    if not YOUTUBE.search(url or ""):
        return url
    from yt_dlp import YoutubeDL

    # `height` bounds the shorter side, so an upright phone-style camera is not cut down to a thumbnail.
    side = max(height, 720)
    long = side * 2
    pick = f"[width<={long}][height<={long}][vcodec!=none]"
    options = {"quiet": True, "no_warnings": True, "format": f"best{pick}/bestvideo{pick}/best", "format_sort": [f"res:{side}"]}
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("url") or info["formats"][-1]["url"]


def safe_login(url: str) -> str:
    """
    `rtsp://admin:P@ss#1@192.168.1.20:554/stream1` as people type it -> the same address with the name and password
    percent-encoded, which is the only form a player accepts. The LAST `@` before the path is the one that ends the
    login, so an `@` inside the password is kept as part of it. An address already encoded is left as it is.
    """
    from urllib.parse import quote, unquote

    match = re.match(r"^([a-z]+://)([^/]*)@([^/@]+)(/.*)?$", url or "", re.I)
    if not match:
        return url
    scheme, login, host, rest = match.groups()
    name, _, password = login.partition(":")
    encoded = quote(unquote(name), safe="") + (":" + quote(unquote(password), safe="") if password else "")
    return f"{scheme}{encoded}@{host}{rest or ''}"


def _split_login(url: str):
    """`http://user:pass@host/x` -> (`http://host/x`, (user, pass)). The login is sent as a header, never left in the address."""
    parts = urlsplit(url)
    if not parts.username:
        return url, None
    from urllib.parse import unquote

    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    # The address carries the login percent-encoded; the header must carry what was actually typed.
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment)), (unquote(parts.username), unquote(parts.password or ""))


def kind_of(url: str) -> str:
    """'video' or 'still'. Only an http(s) address can be a still; it is asked once what it sends."""
    if YOUTUBE.search(url or "") or not re.match(r"https?://", url or "", re.I):
        return "video"
    clean, login = _split_login(url)
    try:
        with requests.get(clean, auth=login, stream=True, timeout=8, verify=False) as reply:
            sends = reply.headers.get("content-type", "").split(";")[0].strip().lower()
            if reply.status_code == 401 and login:  # some cameras want digest, not basic
                with requests.get(clean, auth=requests.auth.HTTPDigestAuth(*login), stream=True, timeout=8, verify=False) as again:
                    sends = again.headers.get("content-type", "").split(";")[0].strip().lower()
        return "still" if sends in STILL_TYPES else "video"
    except requests.RequestException:
        return "video"  # let the video opener try, and report what it finds


class Camera:
    """A camera read one fresh picture at a time. `read()` gives a frame, or None when there is none to give right now."""

    def __init__(self, url: str, height: int = 720):
        self.url = safe_login(url)
        self.height = height
        self.kind = None  # found out on the first read
        self.capture = None
        self.opened_at = 0.0
        self.session = None
        self.login = None
        self.digest = False

    # ---- still pictures over http(s) ----
    def _fetch_still(self):
        if self.session is None:
            self.session = requests.Session()
            self.session.verify = False  # a camera on the local network signs its own certificate
            self.clean, self.login = _split_login(self.url)
        auth = requests.auth.HTTPDigestAuth(*self.login) if self.login and self.digest else self.login
        reply = self.session.get(self.clean, auth=auth, timeout=10)
        if reply.status_code == 401 and self.login and not self.digest:
            self.digest = True
            return self._fetch_still()
        if reply.status_code != 200 or not reply.content:
            return None
        return cv2.imdecode(np.frombuffer(reply.content, np.uint8), cv2.IMREAD_COLOR)

    # ---- video ----
    def _open(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(resolve(self.url, self.height), cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            if hasattr(cv2, prop):
                self.capture.set(getattr(cv2, prop), 15000)
        self.opened_at = time.time()

    def read(self, skip: int = 0):
        """The newest picture. For video, `skip` older frames are thrown away first so it is the present, not the buffer."""
        if self.kind is None:
            self.kind = kind_of(self.url)
        if self.kind == "still":
            try:
                return self._fetch_still()
            except requests.RequestException:
                return None
        if self.capture is None or not self.capture.isOpened() or time.time() - self.opened_at > REOPEN_AFTER:
            self._open()
        for _ in range(max(0, skip)):
            if not self.capture.grab():
                break
        ok, frame = self.capture.read()
        if not ok:
            self._open()
            # The first frames of a stream are often empty until a key frame arrives.
            for _ in range(20):
                ok, frame = self.capture.read()
                if ok:
                    break
        return frame if ok else None

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.session is not None:
            self.session.close()
            self.session = None


# The local cameras' own certificates would otherwise print a warning on every single look.
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover - the warning is cosmetic
    pass
