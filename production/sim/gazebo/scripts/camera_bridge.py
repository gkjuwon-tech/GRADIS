#!/usr/bin/env python3
"""
Gazebo 카메라 브리지 — M4TD 짐벌 EO 카메라 → MJPEG HTTP.

gz transport 이미지 토픽을 구독해서 http://127.0.0.1:8881/cam.mjpg 로 서빙한다.
edge.agent는 Webots 시뮬 때와 '완전히 같은 URL'로 영상을 받는다 —
렌더러가 바뀌어도 프로덕션 스택은 1글자도 안 바뀐다는 원칙.

요구: Ubuntu + Gazebo Harmonic 파이썬 바인딩
  sudo apt install python3-gz-transport13 python3-gz-msgs10
실행:
  python3 camera_bridge.py [--topic /world/gradis_city/.../eo_camera/image]
토픽 확인:
  gz topic -l | grep camera
"""
import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MJPEG_PORT = 8881
DEFAULT_TOPIC = ("/world/gradis_city/model/m4td/link/gimbal_link/"
                 "sensor/eo_camera/image")

latest_jpeg = [None]
stats = {"frames": 0, "t0": time.time()}


def gz_subscriber(topic):
    try:
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image as GzImage
    except ImportError:
        raise SystemExit(
            "gz 파이썬 바인딩 없음:\n"
            "  sudo apt install python3-gz-transport13 python3-gz-msgs10")
    from PIL import Image

    def on_image(msg: GzImage):
        try:
            img = Image.frombytes("RGB", (msg.width, msg.height), msg.data)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            latest_jpeg[0] = buf.getvalue()
            stats["frames"] += 1
            if stats["frames"] % 150 == 0:
                fps = stats["frames"] / (time.time() - stats["t0"])
                print(f"[bridge] {stats['frames']} frames ({fps:.1f} fps)")
        except Exception as e:
            print(f"[bridge] frame error: {e}")

    node = Node()
    if not node.subscribe(GzImage, topic, on_image):
        raise SystemExit(f"[bridge] subscribe 실패: {topic}\n"
                         f"  gz topic -l 로 실제 토픽 확인")
    print(f"[bridge] subscribed: {topic}")
    while True:
        time.sleep(1)


class MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path not in ("/cam.mjpg", "/"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=gradisframe")
        self.end_headers()
        try:
            while True:
                buf = latest_jpeg[0]
                if buf is not None:
                    self.wfile.write(b"--gradisframe\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(buf)))
                    self.end_headers()
                    self.wfile.write(buf)
                    self.wfile.write(b"\r\n")
                time.sleep(1 / 15)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--port", type=int, default=MJPEG_PORT)
    args = ap.parse_args()

    threading.Thread(target=gz_subscriber, args=(args.topic,),
                     daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), MjpegHandler)
    srv.daemon_threads = True
    print(f"[bridge] MJPEG: http://127.0.0.1:{args.port}/cam.mjpg")
    srv.serve_forever()


if __name__ == "__main__":
    main()
