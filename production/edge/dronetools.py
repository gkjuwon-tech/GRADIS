"""
M4TD tool-control adapter for VLM decisions.

The aircraft normally flies its patrol mission through the platform autopilot.
The VLM should intervene only when scene context requires closer inspection or
response, then hand control back. Every tool call is converted to an auditable
Core nav_command with TTL and operator-visible reason text.
"""

TOOLS = {
    "autopilot": (
        "AUTOPILOT",
        {"mode": (str, ("on", "off"))},
        "Toggle autonomous patrol. on=aircraft flies its patrol mission itself; "
        "off=take direct control before using goto/orbit.",
    ),
    "goto": (
        "GOTO",
        {"lat": (float, None), "lon": (float, None), "alt_m": (float, None)},
        "Fly directly to a location. Use after autopilot off when approaching "
        "an incident spotted in the camera view.",
    ),
    "orbit": (
        "ORBIT",
        {"radius_m": (float, None)},
        "Circle the current point of interest to maintain visual coverage.",
    ),
    "thermal": (
        "CAM_THERMAL",
        {"mode": (str, ("on", "off"))},
        "Switch radiometric thermal view on/off. Use at night, in smoke, or to "
        "check a collapsed person's heat signature.",
    ),
    "night_mode": (
        "CAM_NIGHT",
        {"mode": (str, ("on", "off"))},
        "Enable or disable low-light enhanced imaging.",
    ),
    "zoom": (
        "CAM_ZOOM",
        {"factor": (float, None)},
        "Zoom in to inspect a distant subject without flying closer.",
    ),
    "rangefinder": (
        "LASER_RANGE",
        {"cx": (float, None), "cy": (float, None)},
        "Laser-range the object at normalized image position cx,cy.",
    ),
    "spotlight": (
        "SPOTLIGHT",
        {"mode": (str, ("on", "off", "strobe"))},
        "Control spotlight payload. strobe can be used as a visual warning.",
    ),
    "speaker": (
        "SPEAKER",
        {"message": (str, None)},
        "Broadcast a TTS warning through the speaker payload.",
    ),
    "rth": (
        "RETURN_TO_DOCK",
        {},
        "Abort and return to the dock or launch point for charging/standby.",
    ),
}


class DroneToolbox:
    def __init__(self, uplink, enabled=None):
        self.uplink = uplink
        self.enabled = set(enabled) if enabled else set(TOOLS)

    def spec_lines(self):
        out = []
        for name in sorted(self.enabled):
            _, schema, desc = TOOLS[name]
            args = ", ".join(
                f"{k}:{typ.__name__}" + (f"{list(allow)}" if allow else "")
                for k, (typ, allow) in schema.items()
            )
            out.append(f"- {name}({args}): {desc}")
        return out

    def names(self):
        return sorted(self.enabled)

    def call(self, name, args=None, source="vlm-tool", reason=""):
        args = args or {}
        if name not in self.enabled:
            return False, f"unknown/disabled tool: {name}"
        action, schema, _ = TOOLS[name]

        clean = {}
        for key, (typ, allowed) in schema.items():
            if key not in args:
                return False, f"{name}: missing arg '{key}'"
            try:
                value = typ(args[key])
            except (TypeError, ValueError):
                return False, f"{name}: bad arg {key}={args[key]!r}"
            if allowed and value not in allowed:
                return False, f"{name}: {key} must be one of {allowed}"
            clean[key] = value

        mode = clean.pop("mode", None)
        final_action = f"{action}_{mode.upper()}" if mode else action
        message = clean.pop("message", "")

        self.uplink.nav_command(
            final_action,
            reason=reason or message,
            lat=clean.pop("lat", None),
            lon=clean.pop("lon", None),
            alt_m=clean.pop("alt_m", None),
            params=clean or None,
            source=source,
            ttl_s=10,
        )
        return True, f"{name} -> {final_action}"

