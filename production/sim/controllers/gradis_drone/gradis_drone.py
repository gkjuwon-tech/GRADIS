"""
GRADIS 디지털 트윈 — 드론 브리지 컨트롤러 (Webots Supervisor).

역할 두 가지:
  1) 포즈 미러링: ArduCopter SITL(진짜 펌웨어)의 MAVLink 텔레메트리
     (GLOBAL_POSITION_INT + ATTITUDE)를 받아 Webots 드론 노드의 위치/자세를
     매 스텝 갱신한다. 비행역학·모드·페일세이프 전부 SITL이 계산한다.
     Webots는 '세상을 렌더링하는 눈'일 뿐. → 움직임 = 실펌웨어 그 자체.
  2) 카메라 스트리밍: 드론 카메라 프레임을 MJPEG HTTP로 서빙한다.
     edge.agent가 --url http://127.0.0.1:8881/cam.mjpg 로 읽는다(cv2 호환).

MAVLink 연결: SITL serial1 = tcp:127.0.0.1:5762
  (serial0:5760은 mavlink_companion이 쓴다 — 포트당 1클라이언트)

controllerArgs: --home <lat> <lon>   (월드 원점의 GPS 좌표 = SITL --home과 일치)
"""
import io
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from controller import Supervisor  # Webots API

MJPEG_PORT = 8881
MAVLINK_URL = "tcp:127.0.0.1:5762"

# ---- args -------------------------------------------------------------------
home_lat, home_lon = 37.52860, 126.96520
argv = sys.argv[1:]
if "--home" in argv:
    i = argv.index("--home")
    home_lat, home_lon = float(argv[i + 1]), float(argv[i + 2])

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(home_lat))

# ---- 공유 상태 ----------------------------------------------------------------
state = {
    "lat": home_lat, "lon": home_lon, "alt": 0.0,     # GLOBAL_POSITION_INT
    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,            # ATTITUDE (NED rad)
    "ok": False,
}
latest_jpeg = [None]   # MJPEG 서버가 읽는 최신 프레임


# ---- MAVLink 수신 스레드 -------------------------------------------------------
def mavlink_thread():
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[bridge] pymavlink 없음: pip install pymavlink", flush=True)
        return
    while True:
        try:
            m = mavutil.mavlink_connection(MAVLINK_URL)
            m.wait_heartbeat(timeout=30)
            print(f"[bridge] SITL connected ({MAVLINK_URL})", flush=True)
            # 위치/자세 스트림 요청 (10Hz)
            m.mav.request_data_stream_send(
                m.target_system, m.target_component,
                3, 10, 1)   # MAV_DATA_STREAM_POSITION
            m.mav.request_data_stream_send(
                m.target_system, m.target_component,
                1, 10, 1)   # MAV_DATA_STREAM_EXTENDED_STATUS
            m.mav.request_data_stream_send(
                m.target_system, m.target_component,
                10, 15, 1)  # MAV_DATA_STREAM_EXTRA1 (ATTITUDE)
            while True:
                msg = m.recv_match(blocking=True, timeout=5)
                if msg is None:
                    continue
                t = msg.get_type()
                if t == "GLOBAL_POSITION_INT":
                    state["lat"] = msg.lat / 1e7
                    state["lon"] = msg.lon / 1e7
                    state["alt"] = msg.relative_alt / 1000.0
                    state["ok"] = True
                elif t == "ATTITUDE":
                    state["roll"], state["pitch"], state["yaw"] = \
                        msg.roll, msg.pitch, msg.yaw
        except Exception as e:
            print(f"[bridge] MAVLink 재연결 ({e})", flush=True)
            time.sleep(3)


# ---- MJPEG 서버 ---------------------------------------------------------------
class MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path not in ("/cam.mjpg", "/"):
            self.send_response(404); self.end_headers(); return
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


def mjpeg_thread():
    srv = ThreadingHTTPServer(("127.0.0.1", MJPEG_PORT), MjpegHandler)
    srv.daemon_threads = True
    print(f"[bridge] MJPEG: http://127.0.0.1:{MJPEG_PORT}/cam.mjpg", flush=True)
    srv.serve_forever()


# ---- NED 오일러 → ENU 액시스-앵글 ------------------------------------------------
def ned_euler_to_enu_axis_angle(roll, pitch, yaw):
    """MAVLink NED(roll,pitch,yaw) → Webots ENU(x동,y북,z상) 회전.
    ENU 오일러: yaw_enu = π/2 - yaw,  pitch_enu = -pitch,  roll_enu = roll."""
    cy, sy = math.cos((math.pi / 2 - yaw) / 2), math.sin((math.pi / 2 - yaw) / 2)
    cp, sp = math.cos(-pitch / 2), math.sin(-pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    # ZYX 순 쿼터니언 (yaw→pitch→roll)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return (0.0, 0.0, 1.0, 0.0)
    ang = 2.0 * math.atan2(n, w)
    return (x / n, y / n, z / n, ang)


# ---- 메인 (Webots 루프) ---------------------------------------------------------
def main():
    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())
    cam = sup.getDevice("camera")
    cam.enable(max(ts, 64))   # ~15fps 렌더 (MX150 배려)

    me = sup.getSelf()
    f_trans = me.getField("translation")
    f_rot = me.getField("rotation")

    threading.Thread(target=mavlink_thread, daemon=True).start()
    threading.Thread(target=mjpeg_thread, daemon=True).start()

    try:
        from PIL import Image
    except ImportError:
        print("[bridge] Pillow 없음: pip install pillow", flush=True)
        return

    n = 0
    while sup.step(ts) != -1:
        # 1) SITL 포즈 → Webots 노드
        if state["ok"]:
            x = (state["lon"] - home_lon) * M_PER_DEG_LON   # 동
            y = (state["lat"] - home_lat) * M_PER_DEG_LAT   # 북
            z = max(0.05, state["alt"])
            f_trans.setSFVec3f([x, y, z])
            f_rot.setSFRotation(list(
                ned_euler_to_enu_axis_angle(state["roll"], state["pitch"], state["yaw"])))

        # 2) 카메라 → JPEG (4프레임에 1번 — 인코딩 비용 절약, ~15fps 스트림이면 충분)
        n += 1
        if n % 2 == 0:
            raw = cam.getImage()   # BGRA bytes
            if raw:
                w, h = cam.getWidth(), cam.getHeight()
                img = Image.frombytes("RGBA", (w, h), bytes(raw), "raw", "BGRA")
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                latest_jpeg[0] = buf.getvalue()


if __name__ == "__main__":
    main()
