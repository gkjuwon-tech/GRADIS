#!/usr/bin/env python3
"""
GRADIS Litchi Companion - DJI Mini 2 PoC bridge.

This process intentionally does not pretend that Litchi exposes a local flight
control API. Litchi owns flight execution on the Android controller; GRADIS owns
video AI, alerting, audit logs, and operator-ready mission artifacts.

Responsibilities:
  - Send heartbeat rows to GRADIS Core so the Mini 2 appears in fleet status.
  - Poll /api/nav_commands/{node_id}/next and acknowledge every command.
  - Convert GOTO/ROUTE/ORBIT/RETURN_TO_DOCK intents into Litchi-compatible CSV
    mission files and short operator cards in poc/outbox/.
  - Generate a seed patrol mission CSV before the demo.

The operator imports the CSV into Litchi Mission Hub or opens it on the Android
device, reviews altitude/geofence/finish action, then flies it in Litchi.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "config" / "litchi_mini2_poc.json"
DEFAULT_OUTBOX = HERE / "outbox"

ACTION_COLUMNS = []
for i in range(1, 16):
    ACTION_COLUMNS.extend([f"actiontype{i}", f"actionparam{i}"])

LITCHI_COLUMNS = [
    "latitude",
    "longitude",
    "altitude(m)",
    "heading(deg)",
    "curvesize(m)",
    "rotationdir",
    "gimbalmode",
    "gimbalpitchangle",
    *ACTION_COLUMNS,
    "altitudemode",
    "speed(m/s)",
    "poi_latitude",
    "poi_longitude",
    "poi_altitude(m)",
    "poi_altitudemode",
    "photo_timeinterval",
    "photo_distinterval",
]


def post_json(url: str, payload: dict, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw or b"{}")


def get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def meters_to_latlon(home_lat: float, home_lon: float, east_m: float, north_m: float):
    lat = home_lat + north_m / 110_540.0
    lon = home_lon + east_m / (111_320.0 * math.cos(math.radians(home_lat)))
    return lat, lon


class LitchiMissionWriter:
    def __init__(self, cfg: dict, outbox: Path):
        self.cfg = cfg
        self.outbox = outbox
        self.outbox.mkdir(parents=True, exist_ok=True)

        home = cfg["mission"]["home"]
        self.home_lat = float(home["lat"])
        self.home_lon = float(home["lon"])
        self.default_alt = float(cfg["mission"]["default_alt_m"])
        self.default_speed = float(cfg["mission"]["default_speed_m_s"])
        self.gimbal_pitch = float(cfg["mission"].get("gimbal_pitch_deg", -45))
        self.altitude_mode = int(cfg["mission"].get("altitude_mode", 0))
        self.geofence_radius_m = float(cfg["mission"]["safety"]["geofence_radius_m"])

    def _row(self, lat: float, lon: float, alt_m=None, speed=None, heading=0.0):
        row = {
            "latitude": f"{lat:.8f}",
            "longitude": f"{lon:.8f}",
            "altitude(m)": f"{float(alt_m if alt_m is not None else self.default_alt):.1f}",
            "heading(deg)": f"{float(heading):.0f}",
            "curvesize(m)": "0",
            "rotationdir": "0",
            "gimbalmode": "2",
            "gimbalpitchangle": f"{self.gimbal_pitch:.0f}",
            "altitudemode": str(self.altitude_mode),
            "speed(m/s)": f"{float(speed if speed is not None else self.default_speed):.1f}",
            "poi_latitude": f"{lat:.8f}",
            "poi_longitude": f"{lon:.8f}",
            "poi_altitude(m)": f"{float(alt_m if alt_m is not None else self.default_alt):.1f}",
            "poi_altitudemode": str(self.altitude_mode),
            "photo_timeinterval": "0",
            "photo_distinterval": "0",
        }
        for col in ACTION_COLUMNS:
            row[col] = "0"
        return row

    def _write_csv(self, name: str, rows: list[dict]) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = self.outbox / f"{int(time.time())}_{safe}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LITCHI_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_card(self, name: str, command: dict, csv_path: Path | None, notes: list[str]):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = self.outbox / f"{int(time.time())}_{safe}.md"
        lines = [
            f"# GRADIS Litchi Operator Card - {name}",
            "",
            f"- command_id: {command.get('id', 'seed')}",
            f"- action: {command.get('action', name)}",
            f"- source: {command.get('source', 'local')}",
            f"- reason: {command.get('reason', '')}",
            f"- csv: {csv_path.name if csv_path else 'none'}",
            "",
            "## Operator Steps",
            "1. Open Litchi Mission Hub or Litchi on the Android controller.",
            "2. Import the CSV if one was generated.",
            "3. Verify Mini 2, altitude, RTH altitude, geofence, finish action, and battery.",
            "4. Keep RC link active for the entire mission. Mini 2 Litchi waypoint flight uses virtual-stick style control, so link loss must trigger RTH.",
            "5. Start or reject the action as pilot in command.",
            "",
            "## Notes",
        ]
        lines.extend(f"- {note}" for note in notes)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _clamp(self, lat: float, lon: float):
        east = (lon - self.home_lon) * 111_320.0 * math.cos(math.radians(self.home_lat))
        north = (lat - self.home_lat) * 110_540.0
        dist = math.hypot(east, north)
        if dist <= self.geofence_radius_m:
            return lat, lon, False
        scale = self.geofence_radius_m / dist
        return meters_to_latlon(self.home_lat, self.home_lon, east * scale, north * scale) + (True,)

    def seed_patrol(self):
        box = float(self.cfg["mission"]["patrol_box_m"])
        half = box / 2
        offsets = [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
            (-half, -half),
        ]
        rows = []
        for east, north in offsets:
            lat, lon = meters_to_latlon(self.home_lat, self.home_lon, east, north)
            rows.append(self._row(lat, lon))
        csv_path = self._write_csv("mini2_seed_patrol", rows)
        card_path = self._write_card(
            "mini2_seed_patrol",
            {"action": "SEED_PATROL"},
            csv_path,
            [
                "This is the default demo patrol around the configured home point.",
                "CSV stores waypoint rows only; set Litchi global mission settings manually after import.",
            ],
        )
        return csv_path, card_path

    def from_command(self, command: dict):
        action = str(command.get("action", "")).upper()
        notes = []
        csv_path = None

        if action == "GOTO" and command.get("lat") is not None and command.get("lon") is not None:
            lat, lon, clamped = self._clamp(float(command["lat"]), float(command["lon"]))
            if clamped:
                notes.append("Target was outside the PoC geofence and was clamped to the allowed radius.")
            csv_path = self._write_csv(
                f"goto_{command.get('id')}",
                [self._row(self.home_lat, self.home_lon), self._row(lat, lon, command.get("alt_m"))],
            )
            notes.append("Use this as a reviewed Litchi micro-mission, not as an automatic command.")

        elif action == "ROUTE" and command.get("route"):
            rows = []
            for point in command["route"]:
                lat, lon, clamped = self._clamp(float(point["lat"]), float(point["lon"]))
                if clamped:
                    notes.append("At least one route point was clamped to the PoC geofence.")
                rows.append(self._row(lat, lon, point.get("alt_m")))
            csv_path = self._write_csv(f"route_{command.get('id')}", rows)

        elif action == "ORBIT":
            radius = float((command.get("params") or {}).get("radius_m") or 12)
            rows = []
            for deg in range(0, 360, 60):
                east = math.cos(math.radians(deg)) * radius
                north = math.sin(math.radians(deg)) * radius
                lat, lon = meters_to_latlon(self.home_lat, self.home_lon, east, north)
                lat, lon, _ = self._clamp(lat, lon)
                rows.append(self._row(lat, lon, heading=(deg + 180) % 360))
            csv_path = self._write_csv(f"orbit_{command.get('id')}", rows)
            notes.append("Orbit is approximated as a six-point Litchi mission around home/POI.")

        elif action in {"RETURN_TO_DOCK", "RTL"}:
            csv_path = self._write_csv(
                f"return_home_{command.get('id')}",
                [self._row(self.home_lat, self.home_lon, self.default_alt)],
            )
            notes.append("Prefer Litchi/DJI RTH button in an urgent case; CSV is for planned recovery only.")

        elif action in {"LAND", "HOLD", "AUTOPILOT_ON", "AUTOPILOT_OFF"}:
            notes.append("No CSV generated. This is an operator instruction for Litchi/DJI Fly state.")

        else:
            notes.append("Payload or unsupported flight command logged for audit. Mini 2 has no thermal/spotlight/speaker payload.")

        card_path = self._write_card(action.lower() or "command", command, csv_path, notes)
        return csv_path, card_path


class LitchiCompanion:
    def __init__(self, cfg: dict, core_url: str, node_id: str, outbox: Path):
        self.cfg = cfg
        self.core = core_url.rstrip("/")
        self.node_id = node_id
        self.writer = LitchiMissionWriter(cfg, outbox)
        self.home = cfg["mission"]["home"]
        self.state = "LITCHI_READY_OPERATOR_REVIEW"
        self.last_artifact = None

    def heartbeat(self):
        payload = {
            "node_id": self.node_id,
            "node_type": "dji-mini2-litchi",
            "battery": None,
            "lat": float(self.home["lat"]),
            "lon": float(self.home["lon"]),
            "state": self.state,
        }
        post_json(f"{self.core}/api/heartbeat", payload, timeout=2.0)

    def next_cmd(self):
        try:
            return get_json(f"{self.core}/api/nav_commands/{quote(self.node_id)}/next", timeout=2.0)
        except (urllib.error.URLError, OSError, TimeoutError):
            return None

    def ack(self, cid):
        try:
            post_json(f"{self.core}/api/nav_commands/{cid}/ack", {}, timeout=2.0)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"[litchi] ack failed: {exc}")

    def handle(self, command: dict):
        csv_path, card_path = self.writer.from_command(command)
        self.last_artifact = str(card_path)
        self.state = f"OPERATOR_REVIEW_REQUIRED:{command.get('action')}"
        print(f"[litchi] command {command.get('id')} {command.get('action')} -> {card_path}")
        if csv_path:
            print(f"[litchi] mission csv -> {csv_path}")

    def run(self, seed_only=False):
        csv_path, card_path = self.writer.seed_patrol()
        print(f"[litchi] seed patrol csv -> {csv_path}")
        print(f"[litchi] seed patrol card -> {card_path}")
        if seed_only:
            return

        print(f"[litchi] bridge online node={self.node_id} core={self.core}")
        while True:
            try:
                self.heartbeat()
                command = self.next_cmd()
                if command and command.get("id"):
                    try:
                        self.handle(command)
                    finally:
                        self.ack(command["id"])
                time.sleep(2.0)
            except KeyboardInterrupt:
                print("\n[litchi] bye")
                return
            except Exception as exc:
                print(f"[litchi] loop warning: {exc}")
                time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--core", default=None)
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--outbox", default=str(DEFAULT_OUTBOX))
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    node_id = args.node_id or cfg["runtime"]["node_id"]
    core = args.core or cfg["runtime"]["core_url"]
    LitchiCompanion(cfg, core, node_id, Path(args.outbox)).run(seed_only=args.seed_only)


if __name__ == "__main__":
    main()
