"""
HTTP uplink from the edge agent to GRADIS Core.

Normal telemetry is lightweight: battery, position, and state. Raw imagery is
only sent when an incident is escalated. Navigation commands use the same Core
server so AI, operator, and planner commands flow through one auditable gate.

신뢰성 등급:
  - incident  : 절대 잃으면 안 된다 → 전용 워커 큐 + 지수 백오프 재시도.
                (5G가 3초 끊겼다고 낙상 보고가 증발하면 안 된다)
  - heartbeat : 최신 값만 의미 있다 → fire-and-forget, 실패 무시.
  - nav cmd   : TTL이 있다 → fire-and-forget (만료되면 Core가 버린다).
"""
import base64
import json
import queue
import threading
import time
import urllib.error
import urllib.request

_INCIDENT_RETRIES = 5
_RETRY_BASE_S = 1.5      # 1.5, 3, 6, 12, 24초 백오프


class Uplink:
    def __init__(self, core_url, drone_id):
        self.core = core_url.rstrip("/")
        self.drone_id = drone_id
        self._inc_q = queue.Queue(maxsize=32)
        self._worker = threading.Thread(target=self._incident_worker, daemon=True)
        self._worker.start()

    def _post(self, path, payload, timeout=8):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.core + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except (urllib.error.URLError, OSError) as e:
            print(f"[uplink] FAILED {path}: {e}")
            return None

    # --- incident: 큐 + 재시도 (유실 불가) -----------------------------------
    def _incident_worker(self):
        while True:
            payload = self._inc_q.get()
            for attempt in range(_INCIDENT_RETRIES):
                if self._post("/api/ingest", payload, timeout=15) is not None:
                    break
                wait = _RETRY_BASE_S * (2 ** attempt)
                print(f"[uplink] incident retry {attempt + 1}/{_INCIDENT_RETRIES} "
                      f"in {wait:.0f}s")
                time.sleep(wait)
            else:
                print("[uplink] !! incident DROPPED after retries "
                      f"(kind={payload.get('kind')})")

    def send_incident(
        self,
        verdict,
        gps,
        location,
        clip_bytes=None,
        clip_mime="image/gif",
        skeletons=None,
    ):
        payload = {
            "drone_id": self.drone_id,
            "ts_utc": verdict.t_utc,
            "kind": verdict.kind,
            "risk": round(verdict.risk, 3),
            "confidence": round(verdict.confidence, 3),
            "lat": gps[0],
            "lon": gps[1],
            "location": location,
            "skeletons": skeletons or [],
        }
        if clip_bytes:
            payload["clip_b64"] = base64.b64encode(clip_bytes).decode()
            payload["clip_mime"] = clip_mime
        try:
            self._inc_q.put_nowait(payload)
        except queue.Full:
            # 32건이 쌓일 정도면 링크가 죽은 것 — 가장 오래된 것부터 포기
            try:
                self._inc_q.get_nowait()
            except queue.Empty:
                pass
            self._inc_q.put_nowait(payload)
        return True

    # --- telemetry: fire-and-forget ------------------------------------------
    def heartbeat(self, battery, gps, state, node_type="drone"):
        payload = {
            "node_id": self.drone_id,
            "node_type": node_type,
            "battery": battery,
            "lat": gps[0],
            "lon": gps[1],
            "state": state,
        }
        threading.Thread(
            target=self._post,
            args=("/api/heartbeat", payload),
            daemon=True,
        ).start()

    def nav_command(
        self,
        action,
        reason="",
        lat=None,
        lon=None,
        alt_m=None,
        route=None,
        params=None,
        source="edge-ai",
        ttl_s=8,
    ):
        payload = {
            "drone_id": self.drone_id,
            "source": source,
            "action": str(action).upper(),
            "reason": reason,
            "ttl_s": ttl_s,
        }
        if lat is not None:
            payload["lat"] = lat
        if lon is not None:
            payload["lon"] = lon
        if alt_m is not None:
            payload["alt_m"] = alt_m
        if route is not None:
            payload["route"] = route
        if params is not None:
            payload["params"] = params

        threading.Thread(
            target=self._post,
            args=("/api/nav_commands", payload),
            daemon=True,
        ).start()
