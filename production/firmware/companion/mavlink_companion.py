#!/usr/bin/env python3
"""
GRADIS MAVLink companion — 디지털 트윈 전용 (M4TD 자율스택의 SITL 대역).

실기체(DJI Matrice 4TD)의 자율비행·페이로드는 firmware/m4td/m4td_companion.py
(DJI Cloud API)가 담당한다. DJI 스택은 비공개라 시뮬레이션이 불가능하므로,
디지털 트윈(Gazebo/Webots + ArduCopter SITL)에서는 이 컴패니언이 M4TD의
'자율 순찰 + 툴 실행' 역할을 동일한 Core 게이트 의미론으로 대역(standin)한다.
→ 시뮬에서 검증되는 것: AI 판단·툴 의사결정·게이트·지오펜스·도크 상태머신.

Responsibilities:
  - publish telemetry to GRADIS Core
  - AUTOPILOT 상태머신: on=순찰 미션 자율 수행 / off=VLM·운영자 직접 제어
  - VLM/운영자 툴 명령 실행 (GOTO/ORBIT + 짐벌·조명·스피커 페이로드 대역)
  - geofence 강제, MAVLink 링크 워치독, low-battery 도크 복귀
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
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote

try:
    from pymavlink import mavutil
except ImportError:  # py_compile and package validation should still work.
    mavutil = None


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.abspath(
    os.path.join(HERE, "..", "..", "config", "m4td_sim.json")
)


@dataclass(frozen=True)
class Waypoint:
    lat: float
    lon: float
    alt_m: float


@dataclass(frozen=True)
class Dock:
    node_id: str
    lat: float
    lon: float
    state: str
    battery: float | None = None


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read() or b"{}"
    return json.loads(raw)


def get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"[]")


def distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    mean_lat = math.radians((a_lat + b_lat) / 2.0)
    dx = (b_lon - a_lon) * 111_320.0 * math.cos(mean_lat)
    dy = (b_lat - a_lat) * 110_540.0
    return math.hypot(dx, dy)


def waypoints(items: Iterable[dict[str, Any]]) -> list[Waypoint]:
    return [Waypoint(float(x["lat"]), float(x["lon"]), float(x["alt_m"])) for x in items]


class PayloadBridge:
    """짐벌 EO/IR·스포트라이트·스피커 페이로드 명령 변환.

    표준이 있는 것(카메라 줌/모드)은 MAVLink 카메라 프로토콜로 보낸다 —
    NextVision 등 MAVLink 지원 짐벌은 그대로 반응한다. 표준이 없는
    것(스포트라이트/스피커)은 벤더 드라이버 훅(`vendor_hook`)으로 넘기고,
    훅이 없으면 로그만 남긴다(통합 전 개발 모드).
    """

    def __init__(self, mav, vendor_hook=None):
        self.m = mav
        self.vendor_hook = vendor_hook   # callable(action:str, params:dict) | None

    def _cmd(self, command, *p7):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component, command, 0, *p7)

    def handle(self, action: str, params: dict[str, Any], reason: str = "") -> bool:
        if action in ("CAM_THERMAL_ON", "CAM_THERMAL_OFF"):
            # MAV_CMD_SET_CAMERA_MODE: param2=mode (벤더 IR 모드 매핑은 짐벌 설정)
            self._cmd(mavutil.mavlink.MAV_CMD_SET_CAMERA_MODE,
                      0, 2 if action.endswith("ON") else 0, 0, 0, 0, 0, 0)
        elif action == "CAM_ZOOM":
            self._cmd(mavutil.mavlink.MAV_CMD_SET_CAMERA_ZOOM,
                      0, float(params.get("factor", 2.0)), 0, 0, 0, 0, 0)
        elif action in ("CAM_NIGHT_ON", "CAM_NIGHT_OFF", "LASER_RANGE",
                        "SPOTLIGHT_ON", "SPOTLIGHT_OFF", "SPOTLIGHT_STROBE",
                        "SPEAKER"):
            if self.vendor_hook:
                self.vendor_hook(action, dict(params, reason=reason))
            else:
                print(f"[payload] {action} params={params} reason={reason[:60]} "
                      f"(vendor hook 미장착 — 로그만)")
        else:
            return False
        return True


class Companion:
    def __init__(self, cfg: dict[str, Any], master: str, core: str, node_id: str):
        if mavutil is None:
            raise SystemExit("Missing dependency: pip install pymavlink")

        self.cfg = cfg
        self.core = core.rstrip("/")
        self.node_id = node_id
        self.m = mavutil.mavlink_connection(master)

        safety = cfg["mission"]["safety"]
        self.low_battery_pct = float(safety["low_battery_pct"])
        self.critical_battery_pct = float(safety["critical_battery_pct"])
        self.redeploy_battery_pct = float(safety["redeploy_battery_pct"])
        self.approach_alt_m = float(safety["dock_approach_alt_m"])
        self.reach_tolerance_m = float(safety["waypoint_reach_tolerance_m"])
        self.max_charge_wait_s = float(safety["max_charge_wait_s"])
        self.geofence_radius_m = float(safety.get("geofence_radius_m", 300))
        self.link_timeout_s = float(safety.get("mavlink_timeout_s", 5))

        self.patrol = waypoints(cfg["mission"]["patrol"])
        self.configured_docks = [
            Dock(d["node_id"], float(d["lat"]), float(d["lon"]), "CONFIGURED")
            for d in cfg["mission"].get("charging_docks", [])
        ]

        self.lat = 0.0
        self.lon = 0.0
        self.alt_m = 0.0
        self.battery_pct = 100.0
        self.state = "BOOT"
        self.active_dock: Dock | None = None
        self.home: tuple[float, float] | None = None   # 지오펜스 중심 (이륙 지점)
        self.last_msg_t = time.time()                  # MAVLink 링크 워치독
        self._hb_custom_mode = -1                      # 최근 하트비트 모드
        self._hb_base_mode = 0                         # 최근 하트비트 base_mode
        self.autopilot = True                          # on=순찰 자율 / off=직접 제어
        self.payload = PayloadBridge(self.m)           # 짐벌/조명/스피커 변환
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # MAVLink ---------------------------------------------------------------
    def wait_heartbeat(self) -> None:
        print("[companion] waiting for MAVLink heartbeat")
        self.m.wait_heartbeat()
        print(
            f"[companion] connected sys={self.m.target_system} "
            f"comp={self.m.target_component}"
        )
        # 텔레메트리 레이트 상향 — 위치 5Hz / 배터리 1Hz.
        # 기본 스트림은 느려서(1Hz 위치) reach 판정과 지오펜스가 수 미터씩 늦는다.
        self._request_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5.0)
        self._request_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 1.0)

    def _request_interval(self, msg_id: int, hz: float) -> None:
        try:
            self.m.mav.command_long_send(
                self.m.target_system,
                self.m.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                1e6 / hz,   # interval in microseconds
                0, 0, 0, 0, 0,
            )
        except Exception as e:  # 오토파일럿이 미지원이어도 치명적 아님
            print(f"[companion] message interval request failed ({msg_id}): {e}")

    # 모드/시동 확인은 텔레메트리 스레드가 갱신하는 공유 상태(_hb_*)를 읽는다.
    # 같은 연결에서 두 스레드가 recv_match를 하면 메시지를 서로 뺏는다(레이스).
    def set_mode(self, mode: str, confirm_s: float = 5.0) -> bool:
        """모드 전환 + 하트비트로 실제 전환 확인. '보냈으니 됐겠지'는 비행에선 금물."""
        mapping = self.m.mode_mapping()
        if mode not in mapping:
            raise RuntimeError(f"Mode {mode!r} is not available on this vehicle")
        want = mapping[mode]
        deadline = time.time() + confirm_s
        while time.time() < deadline:
            self.m.set_mode_apm(want)
            t0 = time.time()
            while time.time() - t0 < 1.0:
                with self._lock:
                    if self._hb_custom_mode == want:
                        return True
                time.sleep(0.1)
        print(f"[companion] WARNING: mode {mode} not confirmed in {confirm_s}s")
        return False

    def armed(self) -> bool:
        with self._lock:
            return bool(self._hb_base_mode
                        & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def arm(self, confirm_s: float = 15.0) -> bool:
        """시동 + 확인. EKF 미수렴 등으로 거부되면 재시도 (prearm은 FC의 권한)."""
        deadline = time.time() + confirm_s
        while time.time() < deadline:
            self.m.mav.command_long_send(
                self.m.target_system,
                self.m.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0,
            )
            t0 = time.time()
            while time.time() - t0 < 2.0:
                if self.armed():
                    return True
                time.sleep(0.2)
        print(f"[companion] WARNING: arm not confirmed in {confirm_s}s")
        return False

    def takeoff(self, alt_m: float) -> None:
        self.m.mav.command_long_send(
            self.m.target_system,
            self.m.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            alt_m,
        )

    def _clamp_geofence(self, lat: float, lon: float) -> tuple[float, float]:
        """목표 좌표를 이륙 지점 중심 지오펜스 원 안으로 강제한다.

        VLM/운영자/플래너 어디서 온 명령이든 이 게이트를 통과한다 — 치안 드론이
        잘못된 좌표 하나로 옆 동네까지 날아가는 사고를 구조적으로 차단.
        """
        if self.home is None:
            return lat, lon
        h_lat, h_lon = self.home
        d = distance_m(h_lat, h_lon, lat, lon)
        if d <= self.geofence_radius_m:
            return lat, lon
        f = self.geofence_radius_m / d
        c_lat = h_lat + (lat - h_lat) * f
        c_lon = h_lon + (lon - h_lon) * f
        print(
            f"[companion] GEOFENCE: target {d:.0f}m > {self.geofence_radius_m:.0f}m, "
            f"clamped to ({c_lat:.7f}, {c_lon:.7f})"
        )
        return c_lat, c_lon

    def goto(self, lat: float, lon: float, alt_m: float) -> None:
        lat, lon = self._clamp_geofence(lat, lon)
        self.m.mav.set_position_target_global_int_send(
            0,
            self.m.target_system,
            self.m.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            int(0b110111111000),
            int(lat * 1e7),
            int(lon * 1e7),
            alt_m,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def land(self) -> None:
        self.set_mode("LAND")

    def rtl(self) -> None:
        self.set_mode("RTL")
        self.state = "RTL-CRITICAL-BATTERY"

    def condition_yaw(self, delta_deg: float) -> None:
        self.m.mav.command_long_send(
            self.m.target_system,
            self.m.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            abs(delta_deg),
            25,
            1 if delta_deg >= 0 else -1,
            1,
            0,
            0,
            0,
        )

    # Telemetry and Core ----------------------------------------------------
    def _telemetry_loop(self) -> None:
        while not self._stop.is_set():
            msg = self.m.recv_match(blocking=True, timeout=1)
            if not msg:
                continue
            self.last_msg_t = time.time()
            t = msg.get_type()
            with self._lock:
                if t == "GLOBAL_POSITION_INT":
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt_m = msg.relative_alt / 1000.0
                elif t == "SYS_STATUS" and msg.battery_remaining >= 0:
                    self.battery_pct = float(msg.battery_remaining)
                elif t == "HEARTBEAT":
                    self._hb_custom_mode = msg.custom_mode
                    self._hb_base_mode = msg.base_mode

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                payload = {
                    "node_id": self.node_id,
                    "node_type": "drone",
                    "battery": round(self.battery_pct, 1),
                    "lat": self.lat,
                    "lon": self.lon,
                    "state": self.state,
                    "active_dock": self.active_dock.node_id if self.active_dock else None,
                }
            try:
                post_json(self.core + "/api/heartbeat", payload)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                print(f"[companion] heartbeat failed: {e}")
            self._stop.wait(2.0)

    def fleet_docks(self) -> list[Dock]:
        docks: list[Dock] = []
        try:
            rows = get_json(self.core + "/api/fleet")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[companion] fleet lookup failed: {e}")
            rows = []

        for r in rows:
            if r.get("node_type") != "dock":
                continue
            try:
                docks.append(
                    Dock(
                        str(r["node_id"]),
                        float(r["lat"]),
                        float(r["lon"]),
                        str(r.get("state", "UNKNOWN")),
                        None if r.get("battery") is None else float(r["battery"]),
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue

        known = {d.node_id for d in docks}
        docks.extend(d for d in self.configured_docks if d.node_id not in known)
        return docks

    def nearest_ready_dock(self) -> Dock | None:
        ready_states = {
            "IDLE",
            "IDLE-READY",
            "DRONE-DOCKED",
            "CHARGING",
            "CONFIGURED",
        }
        with self._lock:
            lat, lon = self.lat, self.lon
        candidates = [d for d in self.fleet_docks() if d.state in ready_states]
        if not candidates:
            return None
        return min(candidates, key=lambda d: distance_m(lat, lon, d.lat, d.lon))

    def next_nav_command(self) -> dict[str, Any] | None:
        url = f"{self.core}/api/nav_commands/{quote(self.node_id)}/next"
        try:
            cmd = get_json(url, timeout=2.0)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[companion] nav command lookup failed: {e}")
            return None
        return cmd if cmd and cmd.get("id") else None

    def ack_nav_command(self, command_id: Any) -> None:
        try:
            post_json(f"{self.core}/api/nav_commands/{command_id}/ack", {}, timeout=2.0)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[companion] nav command ack failed: {e}")

    # Mission ---------------------------------------------------------------
    def reached(self, wp: Waypoint | Dock, tolerance_m: float | None = None) -> bool:
        tol = self.reach_tolerance_m if tolerance_m is None else tolerance_m
        with self._lock:
            return distance_m(self.lat, self.lon, wp.lat, wp.lon) <= tol

    def start_vehicle(self, takeoff_alt_m: float) -> None:
        """확인 기반 이륙 시퀀스: GUIDED 확인 → ARM 확인 → 고도 도달 확인.
        고정 sleep으로 '아마 떴겠지' 하던 방식 폐기 — 단계마다 증거를 본다."""
        self.state = "TAKEOFF"
        with self._lock:
            if self.home is None and (self.lat or self.lon):
                self.home = (self.lat, self.lon)   # 지오펜스 중심 = 첫 이륙 지점
                print(f"[companion] geofence home set: {self.home} "
                      f"r={self.geofence_radius_m:.0f}m")

        if not self.set_mode("GUIDED"):
            print("[companion] GUIDED 전환 실패 — 이륙 중단")
            self.state = "TAKEOFF-ABORT"
            return
        if not self.arm(confirm_s=30.0):
            print("[companion] 시동 실패(prearm?) — 이륙 중단")
            self.state = "TAKEOFF-ABORT"
            return

        self.takeoff(takeoff_alt_m)
        # 고도 도달 확인 (90% 도달 or 45초 타임아웃)
        deadline = time.time() + 45.0
        while not self._stop.is_set() and time.time() < deadline:
            with self._lock:
                alt = self.alt_m
            if alt >= takeoff_alt_m * 0.9:
                print(f"[companion] takeoff complete alt={alt:.1f}m")
                return
            time.sleep(1)
        print(f"[companion] WARNING: takeoff alt not reached "
              f"(now {self.alt_m:.1f}m / target {takeoff_alt_m:.1f}m)")

    def apply_nav_command(self, cmd: dict[str, Any]) -> bool:
        action = str(cmd.get("action", "")).upper()
        reason = cmd.get("reason", "")
        params = cmd.get("params") or {}
        print(f"[companion] nav command id={cmd.get('id')} action={action} reason={reason}")

        with self._lock:
            lat, lon, alt = self.lat, self.lon, self.alt_m

        # ---- 지휘관 툴: 자율/직접 상태머신 ----
        if action == "AUTOPILOT_ON":
            self.autopilot = True
            self.state = "PATROL"
            self.ack_nav_command(cmd["id"])
            print("[companion] autopilot ON — 순찰 미션 재개")
            return True
        if action == "AUTOPILOT_OFF":
            self.autopilot = False
            self.state = "DIRECT-HOLD"
            self.goto(lat, lon, max(alt, 8.0))   # 현위치 호버, 지시 대기
            self.ack_nav_command(cmd["id"])
            print("[companion] autopilot OFF — 직접 제어 대기")
            return True
        if action == "ORBIT":
            if self.autopilot:
                print("[companion] ORBIT 무시: autopilot on")
                self.ack_nav_command(cmd["id"])
                return False
            self.state = "DIRECT-ORBIT"
            radius = float(params.get("radius_m", 20))
            nan = float("nan")
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavutil.mavlink.MAV_CMD_DO_ORBIT, 0,
                radius, 3.0, 0, 0, nan, nan, nan)   # 현위치 중심 궤도
            self.ack_nav_command(cmd["id"])
            return True

        # ---- 페이로드 툴 (짐벌/조명/스피커) — 비행은 안 멈춘다 ----
        if self.payload.handle(action, params, reason):
            self.ack_nav_command(cmd["id"])
            return False

        if action == "PROCEED":
            self.ack_nav_command(cmd["id"])
            return False
        if action == "HOLD":
            self.state = "AI-HOLD"
            self.goto(lat, lon, max(alt, 8.0))
        elif action == "ASCEND":
            self.state = "AI-ASCEND"
            self.goto(lat, lon, alt + 5.0)
        elif action == "DESCEND":
            self.state = "AI-DESCEND"
            self.goto(lat, lon, max(8.0, alt - 5.0))
        elif action == "YAW_LEFT":
            self.state = "AI-YAW-LEFT"
            self.condition_yaw(-25)
        elif action == "YAW_RIGHT":
            self.state = "AI-YAW-RIGHT"
            self.condition_yaw(25)
        elif action == "GOTO":
            if cmd.get("lat") is None or cmd.get("lon") is None:
                print("[companion] GOTO ignored: missing lat/lon")
                self.ack_nav_command(cmd["id"])
                return False
            if self.autopilot:
                print("[companion] GOTO 무시: autopilot on (먼저 autopilot off)")
                self.ack_nav_command(cmd["id"])
                return False
            self.state = "DIRECT-GOTO"
            self.goto(float(cmd["lat"]), float(cmd["lon"]), float(cmd.get("alt_m") or alt))
        elif action == "ROUTE":
            route = cmd.get("route") or []
            self.state = "AI-ROUTE"
            for item in route:
                with self._lock:
                    if self.battery_pct <= self.low_battery_pct:
                        break
                wp = Waypoint(float(item["lat"]), float(item["lon"]), float(item.get("alt_m", alt)))
                self.goto(wp.lat, wp.lon, wp.alt_m)
                t0 = time.time()
                while not self._stop.is_set() and time.time() - t0 < 30:
                    if self.reached(wp):
                        break
                    time.sleep(1)
        elif action == "RTL":
            self.state = "AI-RTL"
            self.set_mode("RTL")
        elif action == "RETURN_TO_DOCK":
            self.ack_nav_command(cmd["id"])
            self.return_to_charge()
            return True
        elif action == "LAND":
            self.state = "AI-LAND"
            self.land()
        else:
            print(f"[companion] unknown nav command action={action}")
            self.ack_nav_command(cmd["id"])
            return False

        self.ack_nav_command(cmd["id"])
        time.sleep(2)
        return True

    def orbit_patrol(self) -> None:
        i = 0
        while not self._stop.is_set():
            with self._lock:
                battery = self.battery_pct

            # MAVLink 링크 워치독 — 텔레메트리가 끊기면 우리는 장님이다.
            # FC 자체 failsafe(RC/GPS rescue)가 최후의 보루지만, 컴패니언도
            # 새 명령 주입을 멈추고 RTL을 시도하는 게 안전하다.
            silent = time.time() - self.last_msg_t
            if silent > self.link_timeout_s:
                print(f"[companion] MAVLink silent {silent:.1f}s — LINK-LOSS, RTL 시도")
                self.state = "LINK-LOSS"
                try:
                    self.set_mode("RTL")
                except Exception as e:
                    print(f"[companion] RTL set failed during link loss: {e}")
                self._stop.wait(2.0)
                continue

            if battery <= self.critical_battery_pct:
                print(f"[companion] critical battery {battery:.0f}%, RTL")
                self.rtl()
                return
            if battery <= self.low_battery_pct:
                self.return_to_charge()
                i = 0
                continue

            cmd = self.next_nav_command()
            if cmd and self.apply_nav_command(cmd):
                continue

            # 직접 제어 모드: 웨이포인트 진행 정지, 명령 폴링만 유지
            if not self.autopilot:
                time.sleep(1)
                continue

            wp = self.patrol[i % len(self.patrol)]
            self.state = "PATROL"
            self.goto(wp.lat, wp.lon, wp.alt_m)
            print(
                f"[companion] patrol wp={i % len(self.patrol)} "
                f"lat={wp.lat:.7f} lon={wp.lon:.7f} alt={wp.alt_m:.1f}m "
                f"battery={battery:.0f}%"
            )
            t0 = time.time()
            while not self._stop.is_set() and time.time() - t0 < 35:
                if self.reached(wp):
                    break
                with self._lock:
                    if self.battery_pct <= self.low_battery_pct:
                        break
                time.sleep(1)
            i += 1

    def return_to_charge(self) -> None:
        dock = self.nearest_ready_dock()
        if dock is None:
            print("[companion] no dock found, falling back to RTL")
            self.rtl()
            return

        self.active_dock = dock
        self.state = "RETURN-TO-DOCK"
        print(
            f"[companion] low battery, returning to dock={dock.node_id} "
            f"lat={dock.lat:.7f} lon={dock.lon:.7f}"
        )

        self.goto(dock.lat, dock.lon, self.approach_alt_m)
        t0 = time.time()
        while not self._stop.is_set() and time.time() - t0 < 120:
            with self._lock:
                if self.battery_pct <= self.critical_battery_pct:
                    self.rtl()
                    return
            if self.reached(dock, tolerance_m=3.0):
                break
            time.sleep(1)

        self.state = "LANDING-ON-DOCK"
        print(f"[companion] landing on dock={dock.node_id}")
        self.land()
        time.sleep(20)
        self.wait_for_charge()

    def wait_for_charge(self) -> None:
        self.state = "DOCKED-CHARGING"
        t0 = time.time()
        while not self._stop.is_set():
            with self._lock:
                battery = self.battery_pct
            dock_full = False
            if self.active_dock:
                for dock in self.fleet_docks():
                    if dock.node_id == self.active_dock.node_id:
                        dock_full = (dock.battery or 0) >= self.redeploy_battery_pct
                        break

            if battery >= self.redeploy_battery_pct or dock_full:
                print(
                    f"[companion] charge complete battery={battery:.0f}%, redeploying"
                )
                self.state = "REDEPLOY"
                self.active_dock = None
                self.start_vehicle(float(self.cfg["mission"]["takeoff_alt_m"]))
                return

            if time.time() - t0 > self.max_charge_wait_s:
                print("[companion] charge wait timed out, staying docked")
                self.state = "DOCKED-WAITING"
                return

            print(
                f"[companion] charging battery={battery:.0f}% "
                f"target={self.redeploy_battery_pct:.0f}%"
            )
            time.sleep(30)

    def run(self, takeoff_alt_m: float) -> None:
        self.wait_heartbeat()
        threading.Thread(target=self._telemetry_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        self.start_vehicle(takeoff_alt_m)
        try:
            self.orbit_patrol()
        finally:
            self._stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--master", default="udp:127.0.0.1:14550")
    ap.add_argument("--core", default="http://127.0.0.1:8088")
    ap.add_argument("--node-id", default="GRADIS-M4TD-01")
    ap.add_argument("--takeoff-alt", type=float, default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    takeoff_alt = args.takeoff_alt or float(cfg["mission"]["takeoff_alt_m"])
    Companion(cfg, args.master, args.core, args.node_id).run(takeoff_alt)


if __name__ == "__main__":
    main()
