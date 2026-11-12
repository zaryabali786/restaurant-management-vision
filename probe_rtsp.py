"""
Is this RTSP address giving video?  `python probe_rtsp.py rtsp://192.168.1.230:8554 [more paths ...]`

Tries the address as given and the paths phone-camera apps usually serve on, over TCP, giving each a few seconds.
The first one that yields a picture is saved to samples/rtsp_probe.jpg and its address is printed.
"""
import os
import sys
import time

# TCP: phones on Wi-Fi drop UDP. (No "timeout" option here: for RTSP, ffmpeg reads that as "wait for a camera to call
# me", which opens at once and never shows a picture.)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2  # noqa: E402  (must come after the option above)

base = sys.argv[1].rstrip("/")
paths = sys.argv[2:] or ["", "/", "/live", "/h264_ulaw.sdp", "/h264_pcm.sdp", "/h264.sdp", "/stream", "/video", "/cam", "/unicast", "/live/main", "/mystream"]
os.makedirs("samples", exist_ok=True)

for path in paths:
    url = base + path
    started = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    frame = None
    if cap.isOpened():
        for _ in range(15):  # the first frames of a stream are often empty until a key frame arrives
            ok, got = cap.read()
            if ok and got is not None and got.size:
                frame = got
                break
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), round(cap.get(cv2.CAP_PROP_FPS), 1)) if cap.isOpened() else None
    cap.release()
    took = round(time.time() - started, 1)
    if frame is not None:
        cv2.imwrite("samples/rtsp_probe.jpg", frame)
        print(f"VIDEO  {url}   {frame.shape[1]}x{frame.shape[0]}  fps={info[2]}  ({took}s)  -> samples/rtsp_probe.jpg")
        break
    print(f"none   {url}   {'opened but no picture' if info else 'did not open'}  ({took}s)")
else:
    print("No path gave a picture.")
