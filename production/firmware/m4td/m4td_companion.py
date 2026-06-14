#!/usr/bin/env python3
"""
GRADIS M4TD Companion — DJI Matrice 4TD + Dock 3 감독(supervisory) 레이어.

[기체 결정 v4 — 2026-06-12, 시장 피벗]
  초기 타겟 시장 = 필리핀 · 남아공 · 멕시코 (미국 아님 → NDAA/Blue UAS 무관).
  따라서 가격·스펙·도크 생태계 모두 우위인 DJI Matrice 4TD 채택.
  (이 파일은 한때 "미국 진출" 사유로 소각됐다가 부활함. 모든 결정엔 시장이 먼저다.)
  MAVLink 경로(firmware/companion/)는 SITL 디지털 트윈 대역 전용.

이 레이어의 역할은 '감독'이다:
  1) Core nav_commands 게이트 폴링 → VLM/운영자 툴 명령을 DJI Cloud API로 변환
  2) 지오펜스 강제 (어떤 GOTO도 반경 밖이면 클램프 — 우리 게이트는 유지한다)
  3) 텔레메트리 → Core 하트비트 중계
  4) AUTOPILOT on/off 상태머신: on=DJI 미션 소유권, off=DRC 직접 제어

비행·장애물 회피·배터리/충전은 기체+Dock 3가 자체 수행한다:
  - 순찰 비행:   DJI Dock 3 미션 (FlightHub 2 / Cloud API)
  - 장애물 회피: 전방위 비전 + 3D IR (기체 내장)
  - 배터리:      Dock 3 자동 충전/재출격

전송 계층:
  - paho-mqtt + --mqtt: DJI Cloud API MQTT 토픽 실발행
    (thing/product/{gateway_sn}/services — DJI Cloud API v2 규격)
  - 기본: DryRun — 변환 결과를 로그로만 (개발/검증용)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.abspath(os.path.join(HERE, "..", "..", "config", "m4td.json"))


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> Any:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def distance_m(a_lat, a_lon, b_lat, b_lon) -> float:
    mean_lat = math.radians((a_lat + b_lat) / 2.0)
    dx = (b_lon - a_lon) * 111_320.0 * math.cos(mean_lat)
    dy = (b_lat - a_lat) * 110_540.0
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# DJI Cloud API 전송 계층
# ---------------------------------------------------------------------------
class DryRunTransport:
    """실기체 없이 변환 결과를 검증하는 모드. 토픽/페이로드는 실규격 그대로."""

    def __init__(self, gateway_sn: str):
        self.gateway_sn = gateway_sn

    def service(self, method: str, data: dict[str, Any]) -> None:
        topic = f"thing/product/{self.gateway_sn}/services"
        body = {"tid": str(uuid.uuid4()), "bid": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "method": method, "data": data}
        print(f"[m4td][dry-run] {topic}\n              "
              f"{json.dumps(body, ensure_ascii=False)}")


class MqttTransport(DryRunTransport):
    """DJI Cloud API MQTT 발행 (paho-mqtt 필요)."""

    def __init__(self, gateway_sn: str, host: str, port: int,
                 username: str, password: str):
        super().__init__(gateway_sn)
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(client_id=f"gradis-{gateway_sn}")
        if username:
            self.client.username_pw_set(username, password)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()
        print(f"[m4td] MQTT connected {host}:{port}")

    def service(self, method: str, data: dict[str, Any]) -> None:
        topic = f"thing/product/{self.gateway_sn}/services"
        body = {"tid": str(uuid.uuid4()), "bid": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "method": method, "data": data}
        self.client.publish(topic, json.dumps(body), qos=1)
        print(f"[m4td] -> {method} {data}")


# ---------------------------------------------------------------------------
# 감독 레이어
# ---------------------------------------------------------------------------
class M4TDCompanion:
    def __init__(self, cfg: dict[str, Any], core: str, node_id: str, transport):
        self.cfg = cfg
        self.core = core.rstrip("/")
        self.node_id = node_id
        self.t = transport

        safety = cfg["mission"]["safety"]
        self.geofence_radius_m = float(safety.get("geofence_radius_m", 300))
        home = cfg["mission"]["dock"]
        self.home = (float(home["lat"]), float(home["lon"]))
        self.payload_index = cfg["platform"].get("payload_index", "99-0-0")

        self.autopilot = True        # 기본: DJI 미션이 비행 소유
        self.state = "AUTOPILOT-PATROL"
        self._stop = threading.Event()

    # ---- 지오펜스 (우리 게이트 — DJI 지오펜스와 이중) ------------------------
    def _clamp(self, lat: float, lon: float) -> tuple[float, float]:
        d = distance_m(self.home[0], self.home[1], lat, lon)
        if d <= self.geofence_radius_m:
            return lat, lon
        f = self.geofence_radius_m / d
        c_lat = self.home[0] + (lat - self.home[0]) * f
        c_lon = self.home[1] + (lon - self.home[1]) * f
        print(f"[m4td] GEOFENCE: {d:.0f}m > {self.geofence_radius_m:.0f}m → clamp")
        return c_lat, c_lon

    # ---- 액션 → DJI Cloud API ------------------------------------------------
    def apply(self, cmd: dict[str, Any]) -> None:
        action = str(cmd.get("action", "")).upper()
        params = cmd.get("params") or {}
        print(f"[m4td] cmd id={cmd.get('id')} action={action} "
              f"src={cmd.get('source')} reason={str(cmd.get('reason', ''))[:60]}")

        if action == "AUTOPILOT_ON":
            self.t.service("drc_mode_exit", {})
            self.t.service("flighttask_recovery", {})
            self.autopilot, self.state = True, "AUTOPILOT-PATROL"

        elif action == "AUTOPILOT_OFF":
            self.t.service("flighttask_pause", {})
            self.t.service("drc_mode_enter",
                           {"mqtt_broker": {}, "osd_frequency": 10,
                            "hsi_frequency": 1})
            self.autopilot, self.state = False, "DIRECT-CONTROL"

        elif action == "GOTO":
            if self.autopilot:
                print("[m4td] GOTO 무시: autopilot on (먼저 autopilot off 필요)")
                return
            lat, lon = self._clamp(float(cmd["lat"]), float(cmd["lon"]))
            self.t.service("fly_to_point", {
                "fly_to_id": str(uuid.uuid4()),
                "max_speed": 14,
                "points": [{"latitude": lat, "longitude": lon,
                            "height": float(cmd.get("alt_m") or 40)}]})
            self.state = "DIRECT-GOTO"

        elif action == "ORBIT":
            if self.autopilot:
                print("[m4td] ORBIT 무시: autopilot on")
                return
            self.t.service("poi_mode_enter",
                           {"circle_radius": float(params.get("radius_m", 20))})
            self.state = "DIRECT-ORBIT"

        elif action in ("CAM_THERMAL_ON", "CAM_THERMAL_OFF"):
            self.t.service("camera_mode_switch", {
                "payload_index": self.payload_index,
                "camera_mode": 2 if action.endswith("ON") else 0})  # 2=IR, 0=wide

        elif action in ("CAM_NIGHT_ON", "CAM_NIGHT_OFF"):
            self.t.service("camera_exposure_mode_set", {
                "payload_index": self.payload_index,
                "exposure_mode": "night" if action.endswith("ON") else "auto"})

        elif action == "CAM_ZOOM":
            self.t.service("camera_focal_length_set", {
                "payload_index": self.payload_index,
                "camera_type": "zoom",
                "zoom_factor": float(params.get("factor", 2.0))})

        elif action == "LASER_RANGE":
            self.t.service("camera_point_focus_action", {
                "payload_index": self.payload_index,
                "x": float(params.get("cx", 0.5)),
                "y": float(params.get("cy", 0.5)),
                "laser_ranging": True})

        elif action.startswith("SPOTLIGHT"):
            mode = action.split("_")[-1].lower()      # on/off/strobe
            self.t.service("spotlight_set", {
                "payload_index": self.payload_index,
                "spotlight_mode": mode, "brightness": 100})

        elif action == "SPEAKER":
            self.t.service("speaker_tts_play", {
                "payload_index": self.payload_index,
                "text": str(cmd.get("reason", ""))[:200], "play_mode": "single"})

        elif action in ("RETURN_TO_DOCK", "RTL"):
            self.t.service("return_home", {})
            self.autopilot, self.state = True, "RETURN-TO-DOCK"

        elif action in ("HOLD", "PROCEED", "ASCEND", "DESCEND",
                        "YAW_LEFT", "YAW_RIGHT", "LAND", "ROUTE"):
            # 저수준 마이크로 명령 — M4TD에선 자율 스택이 처리. 기록만.
            print(f"[m4td] {action}: M4TD 자율 스택이 처리 — no-op")
        else:
            print(f"[m4td] unknown action {action}")

    # ---- Core 연동 -----------------------------------------------------------
    def _ack(self, cid) -> None:
        try:
            post_json(f"{self.core}/api/nav_commands/{cid}/ack", {}, timeout=2.0)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[m4td] ack failed: {e}")

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                post_json(self.core + "/api/heartbeat", {
                    "node_id": self.node_id, "node_type": "drone",
                    "battery": None,            # 실기체: OSD에서, 도크가 관리
                    "lat": self.home[0], "lon": self.home[1],
                    "state": self.state,
                })
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                print(f"[m4td] heartbeat failed: {e}")
            self._stop.wait(2.0)

    def run(self) -> None:
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print(f"[m4td] supervisory loop up — autopilot={self.autopilot} "
              f"geofence={self.geofence_radius_m:.0f}m home={self.home}")
        url = f"{self.core}/api/nav_commands/{quote(self.node_id)}/next"
        try:
            while True:
                try:
                    cmd = get_json(url, timeout=2.0)
                except (urllib.error.URLError, OSError, TimeoutError) as e:
                    print(f"[m4td] poll failed: {e}")
                    time.sleep(3)
                    continue
                if cmd and cmd.get("id"):
                    try:
                        self.apply(cmd)
                    finally:
                        self._ack(cmd["id"])
                else:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[m4td] bye.")
        finally:
            self._stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--core", default="http://127.0.0.1:8088")
    ap.add_argument("--node-id", default=None)
    ap.add_argument("--mqtt", action="store_true",
                    help="DJI Cloud API MQTT 실발행 (기본: dry-run 로그)")
    args = ap.parse_args()

    cfg = load_json(args.config)
    node_id = args.node_id or cfg["runtime"]["node_id"]
    sn = cfg["platform"]["gateway_sn"]

    if args.mqtt:
        m = cfg["cloud_api"]["mqtt"]
        transport = MqttTransport(sn, m["host"], int(m["port"]),
                                  m.get("username", ""), m.get("password", ""))
    else:
        transport = DryRunTransport(sn)

    M4TDCompanion(cfg, args.core, node_id, transport).run()


if __name__ == "__main__":
    main()
