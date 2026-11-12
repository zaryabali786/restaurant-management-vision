"""
Does `Camera` read every kind of camera address?   python test_streams.py [rtsp://… a real RTSP camera to try as well]

Stands up its own little camera on this computer that serves the saved sample frames four ways — a still picture over
http, the same behind a password, a still over https with a self-signed certificate, and a moving MJPEG stream — and
reads each through the same `Camera` the worker uses. RTSP needs a real camera: give its address to include it.
"""
import base64
import glob
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from streams import Camera, kind_of

FRAMES = [open(p, "rb").read() for p in sorted(glob.glob("samples/dining_[0-9].jpg"))]
if not FRAMES:
    raise SystemExit("No sample frames in samples/.")
LOGIN = "Basic " + base64.b64encode(b"admin:secret").decode()
count = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.startswith("/secure") and self.headers.get("Authorization") != LOGIN:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="camera"')
            self.end_headers()
            return
        if self.path.startswith("/mjpeg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for i in range(200):
                    jpg = FRAMES[i % len(FRAMES)]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.05)
            except OSError:
                pass
            return
        count["n"] += 1
        jpg = FRAMES[count["n"] % len(FRAMES)]
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.end_headers()
        self.wfile.write(jpg)


def serve(port, tls=False):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if tls:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "samples/_k.pem", "-out", "samples/_c.pem", "-days", "1", "-subj", "/CN=camera.local"], check=True, capture_output=True)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain("samples/_c.pem", "samples/_k.pem")
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


serve(8791)
try:
    serve(8792, tls=True)
    https_ok = True
except Exception as error:  # no openssl on this computer
    https_ok = False
    print("(https test skipped: could not make a certificate:", error, ")")

CASES = [
    ("http still picture", "http://127.0.0.1:8791/shot.jpg", "still"),
    ("http still, with a password", "http://admin:secret@127.0.0.1:8791/secure/shot.jpg", "still"),
    ("http MJPEG video", "http://127.0.0.1:8791/mjpeg", "video"),
]
if https_ok:
    CASES.append(("https still, self-signed certificate", "https://127.0.0.1:8792/shot.jpg", "still"))
    CASES.append(("https MJPEG video, self-signed", "https://127.0.0.1:8792/mjpeg", "video"))
if len(sys.argv) > 1:
    CASES.append(("the RTSP camera given", sys.argv[1], "video"))

failed = 0
for name, url, expect in CASES:
    started = time.time()
    kind = kind_of(url)
    camera = Camera(url)
    shapes = []
    for _ in range(3):
        frame = camera.read()
        shapes.append(None if frame is None else f"{frame.shape[1]}x{frame.shape[0]}")
    camera.close()
    good = kind == expect and all(shapes)
    failed += 0 if good else 1
    print(f"{'PASS' if good else 'FAIL'}  {name:<38} read as {kind:<5}  frames: {shapes}  ({time.time() - started:.1f}s)")

wrong = Camera("http://admin:nope@127.0.0.1:8791/secure/shot.jpg")
refused = wrong.read() is None
wrong.close()
failed += 0 if refused else 1
print(f"{'PASS' if refused else 'FAIL'}  a wrong password gets no picture")
print("\nall passed" if not failed else f"\n{failed} FAILED")
sys.exit(1 if failed else 0)
