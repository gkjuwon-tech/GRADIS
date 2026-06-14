#!/usr/bin/env python3
"""
GRADIS Core — 관제 서버 (Ground Control)

드론(엣지)이 "야 사건 났어" 하고 던지는 Incident를 받아서 저장하고,
관제사 브라우저로 실시간(SSE) 중계한다. 그게 전부다. 멍청할 정도로 단순해야
새벽 3시에 안 죽는다.

설계 원칙:
  - 의존성 0. 파이썬 표준 라이브러리만 쓴다. (pip install 하다가 출동 놓치면 인생 끝)
  - 추론은 절대 여기서 안 한다. 드론이 이미 다 했다. 여긴 우체국이다.
  - 원본 클립은 '위험할 때만' 도착한다. 평상시엔 아무것도 안 온다 = 프라이버시.

실행:  python core/server.py            (기본 127.0.0.1:8088)
       PORT=9000 python core/server.py
"""
import json, os, sqlite3, threading, time, base64, queue, html, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Windows 콘솔(cp949)에서도 한글/em-dash 안 깨지게
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "_data")
CLIPS = os.path.join(DATA, "clips")
DB = os.path.join(DATA, "gradis.db")
os.makedirs(CLIPS, exist_ok=True)

# ----------------------------------------------------------------------------
# DB
# ----------------------------------------------------------------------------
_dblock = threading.Lock()
_db = sqlite3.connect(DB, check_same_thread=False)
# WAL: 쓰기(ingest/heartbeat)가 읽기(대시보드/next-poll)를 안 막는다.
# synchronous=NORMAL: WAL에선 안전하면서 fsync 비용 대폭 절감.
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("PRAGMA synchronous=NORMAL")
_db.execute("""CREATE TABLE IF NOT EXISTS incidents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc REAL, drone_id TEXT, kind TEXT, risk REAL, confidence REAL,
    lat REAL, lon REAL, location TEXT, clip TEXT, skeletons TEXT,
    status TEXT DEFAULT 'OPEN', acked_at REAL)""")
_db.execute("""CREATE TABLE IF NOT EXISTS fleet(
    node_id TEXT PRIMARY KEY, node_type TEXT, battery REAL,
    lat REAL, lon REAL, state TEXT, last_seen REAL)""")
_db.execute("""CREATE TABLE IF NOT EXISTS nav_commands(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc REAL, drone_id TEXT, source TEXT, action TEXT,
    lat REAL, lon REAL, alt_m REAL, route TEXT, reason TEXT,
    ttl_s REAL, status TEXT DEFAULT 'PENDING', acked_at REAL)""")
# 마이그레이션: 툴 인자 컬럼 (기존 DB 호환)
try:
    _db.execute("ALTER TABLE nav_commands ADD COLUMN params TEXT")
except sqlite3.OperationalError:
    pass   # 이미 있음
# 핫패스 인덱스: 컴패니언이 2초마다 PENDING을 폴링한다
_db.execute("""CREATE INDEX IF NOT EXISTS idx_nav_pending
               ON nav_commands(drone_id, status, id)""")
_db.execute("CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(ts_utc)")
_db.commit()

NAV_ACTIONS = {
    # 비행 (저수준 — SITL/시뮬 호환)
    "PROCEED", "HOLD", "ASCEND", "DESCEND", "YAW_LEFT", "YAW_RIGHT",
    "GOTO", "ROUTE", "RTL", "RETURN_TO_DOCK", "LAND", "ORBIT",
    # M4TD 툴 (지휘관 모드 — edge/dronetools.py와 1:1)
    "AUTOPILOT_ON", "AUTOPILOT_OFF",
    "CAM_THERMAL_ON", "CAM_THERMAL_OFF",
    "CAM_NIGHT_ON", "CAM_NIGHT_OFF",
    "CAM_ZOOM", "LASER_RANGE",
    "SPOTLIGHT_ON", "SPOTLIGHT_OFF", "SPOTLIGHT_STROBE",
    "SPEAKER",
}


def db(q, args=(), commit=False, one=False):
    with _dblock:
        cur = _db.execute(q, args)
        if commit:
            _db.commit()
            return cur.lastrowid
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
    out = [dict(zip(cols, r)) for r in rows]
    return (out[0] if out else None) if one else out


# ----------------------------------------------------------------------------
# SSE Hub — 관제사 브라우저들한테 실시간 방송
# ----------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.subs = set()
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=100)
        with self.lock:
            self.subs.add(q)
        return q

    def drop(self, q):
        with self.lock:
            self.subs.discard(q)

    def publish(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        with self.lock:
            dead = []
            for q in self.subs:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.subs.discard(q)

HUB = Hub()


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def static(name):
    with open(os.path.join(HERE, name), "rb") as f:
        return f.read()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 조용히. 사건 로그만 보고 싶다.
        pass

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        p, qs = u.path, parse_qs(u.query)

        if p == "/" or p == "/index.html":
            return self._send(200, static("admin.html"), "text/html; charset=utf-8")
        if p == "/app.js":
            return self._send(200, static("app.js"), "application/javascript")
        if p == "/api/health":
            return self._send(200, json.dumps({"ok": True, "ts": time.time()}))

        if p == "/api/incidents":
            lim = int(qs.get("limit", ["100"])[0])
            rows = db("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (lim,))
            return self._send(200, json.dumps(rows))

        if p == "/api/fleet":
            rows = db("SELECT * FROM fleet ORDER BY node_id")
            return self._send(200, json.dumps(rows))

        if p == "/api/nav_commands":
            drone_id = qs.get("drone_id", [None])[0]
            if drone_id:
                rows = db("""SELECT * FROM nav_commands
                             WHERE drone_id=? ORDER BY id DESC LIMIT 50""",
                          (drone_id,))
            else:
                rows = db("SELECT * FROM nav_commands ORDER BY id DESC LIMIT 100")
            return self._send(200, json.dumps(rows))

        if p.startswith("/api/nav_commands/") and p.endswith("/next"):
            parts = p.strip("/").split("/")
            if len(parts) == 4:
                return self._next_nav(parts[2])

        if p.startswith("/clips/"):
            fn = os.path.basename(p)
            fp = os.path.join(CLIPS, fn)
            if not os.path.exists(fp):
                return self._send(404, b'{"error":"gone"}')
            ext = fn.rsplit(".", 1)[-1].lower()
            ctype = {"gif": "image/gif", "mp4": "video/mp4"}.get(ext, "application/octet-stream")
            with open(fp, "rb") as f:
                return self._send(200, f.read(), ctype, {"Cache-Control": "max-age=86400"})

        if p == "/api/events":
            return self._sse()

        return self._send(404, b'{"error":"not found"}')

    # ---- SSE ----
    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = HUB.subscribe()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    msg = ": ping\n\n"
                self.wfile.write(msg.encode() if isinstance(msg, str) else msg)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.drop(q)

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"error":"bad json"}')

        if u.path == "/api/ingest":
            return self._ingest(body)
        if u.path == "/api/heartbeat":
            return self._heartbeat(body)
        if u.path == "/api/nav_commands":
            return self._nav_command(body)
        if u.path.startswith("/api/nav_commands/") and u.path.endswith("/ack"):
            cid = u.path.split("/")[3]
            db("UPDATE nav_commands SET status='ACK', acked_at=? WHERE id=?",
               (time.time(), cid), commit=True)
            row = db("SELECT * FROM nav_commands WHERE id=?", (cid,), one=True)
            HUB.publish("nav_command", row)
            return self._send(200, json.dumps(row or {}))
        if u.path.startswith("/api/incidents/") and u.path.endswith("/ack"):
            iid = u.path.split("/")[3]
            db("UPDATE incidents SET status='ACK', acked_at=? WHERE id=?",
               (time.time(), iid), commit=True)
            row = db("SELECT * FROM incidents WHERE id=?", (iid,), one=True)
            HUB.publish("update", row)
            return self._send(200, json.dumps(row or {}))

        return self._send(404, b'{"error":"not found"}')

    def _ingest(self, b):
        # 원본 클립(base64)을 디스크에 떨군다. 이건 '위험'이 확인됐을 때만 온다.
        clip_url = None
        clip_b64 = b.get("clip_b64")
        if clip_b64:
            ext = "gif" if b.get("clip_mime", "").endswith("gif") else "mp4"
            fn = f"{int(time.time()*1000)}_{b.get('drone_id','drone')}.{ext}"
            with open(os.path.join(CLIPS, fn), "wb") as f:
                f.write(base64.b64decode(clip_b64))
            clip_url = f"/clips/{fn}"

        iid = db("""INSERT INTO incidents
            (ts_utc,drone_id,kind,risk,confidence,lat,lon,location,clip,skeletons,status)
            VALUES (?,?,?,?,?,?,?,?,?,?, 'OPEN')""",
            (b.get("ts_utc", time.time()), b.get("drone_id", "UNKNOWN"),
             b.get("kind", "Unknown"), float(b.get("risk", 0)),
             float(b.get("confidence", 0)), b.get("lat"), b.get("lon"),
             b.get("location", ""), clip_url,
             json.dumps(b.get("skeletons", []))[:20000]), commit=True)

        row = db("SELECT * FROM incidents WHERE id=?", (iid,), one=True)
        HUB.publish("incident", row)
        print(f"[INCIDENT #{iid}] {row['kind']:18s} risk={row['risk']:.2f} "
              f"conf={row['confidence']:.2f} @ {row['location']} ({row['drone_id']})"
              f"{'  +clip' if clip_url else ''}")
        return self._send(200, json.dumps({"id": iid, "clip": clip_url}))

    def _heartbeat(self, b):
        db("""INSERT INTO fleet(node_id,node_type,battery,lat,lon,state,last_seen)
              VALUES(?,?,?,?,?,?,?)
              ON CONFLICT(node_id) DO UPDATE SET
              node_type=excluded.node_type, battery=excluded.battery,
              lat=excluded.lat, lon=excluded.lon, state=excluded.state,
              last_seen=excluded.last_seen""",
           (b.get("node_id", "node"), b.get("node_type", "drone"),
            b.get("battery"), b.get("lat"), b.get("lon"),
            b.get("state", "UNKNOWN"), time.time()), commit=True)
        row = db("SELECT * FROM fleet WHERE node_id=?", (b.get("node_id", "node"),), one=True)
        HUB.publish("fleet", row)
        return self._send(200, b'{"ok":true}')

    def _nav_command(self, b):
        action = str(b.get("action", "HOLD")).upper()
        if action not in NAV_ACTIONS:
            return self._send(400, json.dumps({"error": "bad action", "allowed": sorted(NAV_ACTIONS)}))
        drone_id = str(b.get("drone_id") or b.get("node_id") or "").strip()
        if not drone_id:
            return self._send(400, b'{"error":"drone_id required"}')

        route = b.get("route")
        if route is not None:
            route = json.dumps(route)[:20000]
        params = b.get("params")
        if params is not None:
            params = json.dumps(params)[:2000]

        cid = db("""INSERT INTO nav_commands
            (ts_utc,drone_id,source,action,lat,lon,alt_m,route,reason,ttl_s,status,params)
            VALUES (?,?,?,?,?,?,?,?,?,?, 'PENDING', ?)""",
            (time.time(), drone_id, str(b.get("source", "api"))[:80], action,
             b.get("lat"), b.get("lon"), b.get("alt_m"), route,
             str(b.get("reason", ""))[:500], float(b.get("ttl_s", 10)), params),
            commit=True)
        row = db("SELECT * FROM nav_commands WHERE id=?", (cid,), one=True)
        HUB.publish("nav_command", row)
        return self._send(200, json.dumps(row))

    def _next_nav(self, drone_id):
        now = time.time()
        row = db("""SELECT * FROM nav_commands
                    WHERE drone_id=? AND status='PENDING'
                      AND ts_utc + ttl_s >= ?
                    ORDER BY id ASC LIMIT 1""",
                 (drone_id, now), one=True)
        if not row:
            return self._send(200, b'{}')
        db("UPDATE nav_commands SET status='DISPATCHED' WHERE id=?",
           (row["id"],), commit=True)
        if row.get("route"):
            try:
                row["route"] = json.loads(row["route"])
            except Exception:
                row["route"] = []
        if row.get("params"):
            try:
                row["params"] = json.loads(row["params"])
            except Exception:
                row["params"] = {}
        HUB.publish("nav_command", row)
        return self._send(200, json.dumps(row))


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8088"))
    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    print("=" * 60)
    print(" GRADIS CORE — Ground Control")
    print(f" Admin   : http://{host}:{port}/")
    print(f" Ingest  : POST http://{host}:{port}/api/ingest")
    print(f" DB      : {DB}")
    print("=" * 60)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye.")


if __name__ == "__main__":
    main()
