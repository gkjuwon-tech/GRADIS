# Defender 25 Firmware Notes

Target platform: iFlight Defender 25 O4 4S HD.

Official product data checked on 2026-06-10:

- Flight electronics: BLITZ D25 F7 AIO.
- FC firmware target: `IFLIGHT_BLITZ_F722_X1`.
- Video: DJI O4 Air Unit Pro.
- GNSS option: GPS+SBAS+Galileo+QZSS+Glonass.
- Aircraft weight: `170 +/- 5g`.
- MTOM with 550mAh battery: `240 +/- 5g`.
- MTOM with 900mAh battery: `280 +/- 5g`.
- Stock battery: 4S 550mAh Li-Po.

Important integration boundary:

- The vendor product page states that iFlight products do not support
  autonomous flight. Treat the stock Defender 25 as a Betaflight FPV platform,
  not as an SDK-native autonomous drone.
- Do not overwrite factory Betaflight settings until a CLI dump is captured.
- GRADIS autonomy belongs in the companion/ground-control layer. Keep FC-level
  failsafes, receiver failsafe, GPS rescue/RTL behavior, and battery alarms
  configured in Betaflight or the MAVLink autopilot layer.

Recommended field setup:

1. Capture the factory CLI dump in Betaflight Configurator.
2. Confirm target is `IFLIGHT_BLITZ_F722_X1`.
3. Confirm GPS UART and receiver UART wiring against the current wiring PDF.
4. Confirm battery voltage scale with a multimeter.
5. Set conservative low-voltage warnings before enabling GRADIS missions.
6. Run tethered/props-off tests for companion heartbeat, video, YOLO, and dock
   state transitions.

GRADIS package contents:

- `firmware/companion/mavlink_companion.py`: MAVLink mission, low-battery dock
  return, charge wait, and redeploy state machine.
- `firmware/dock_esp32/dock_esp32.ino`: charging dock heartbeat and relay
  controller sketch.
- `config/defender25.json`: hardware, video, model, prompt, dock, and safety
  config.
- `edge/`: YOLO pose, privacy redaction, VLM adapter, incident uplink.
- `core/`: ground-control API and admin UI.

Sources:

- https://shop.iflight.com/Defender-25-O4-4S-HD-Pro2329
- https://shop.iflight.com/Defender-25-F7-AIO-Pro1929
- https://shop.iflight.com/Defender-25-Battery-Pro1925
