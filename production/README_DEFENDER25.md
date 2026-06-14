# GRADIS Defender 25 Production Package

This folder is now organized so a Defender 25 build can be moved as one
package after purchase.

Architecture (v4): the Python engine extracts anonymous skeletons with a
split top-view pipeline: YOLO person detector/tracker -> MMPose RTMPose or
ViTPose -> temporal SkeletonEngine (GRADIS-25: COCO-17 + 8 derived joints).
All situation and navigation judgment is done by the VLM on privacy-redacted
or mannequin frames. There is no rule-based detector anymore. Hardware BOM:
`docs/BOM_DEFENDER25.md`.

Avatar layer (`--avatar`): skeleton coordinates drive a rigged faceless gray
mannequin (default `assets3d/Xbot.glb` — Mixamo X Bot, 67-bone full rig, stored
offline; `RiggedFigure.glb` kept as a low-poly fallback). Per-bone 2D retarget
+ CPU linear-blend skinning + vertex-cluster decimation + cv2 painter
rasterizer, composited over the blurred person, so the VLM sees a human
*shape* in the exact observed pose instead of dots-and-lines. Privacy is
unchanged — the mannequin is generated from coordinates, not pixels.
Preview the result:
`python tools/avatar_on_video.py _media/wide_45881.mp4 --model yolo11x.pt --pose-preset rtmpose-m`
uses YOLO only for person boxes/tracks and MMPose for keypoints. For quality
checks, switch to `--pose-preset vitpose-s`.

Main commands from `production/`:

```powershell
.\scripts\build_defender25_package.ps1
python core\server.py
python -m edge.agent --source rtsp --url rtsp://127.0.0.1:8554/defender25 --detector-model yolo11x.pt --pose-preset rtmpose-m --vlm ollama --vlm-model moondream
python firmware\companion\mavlink_companion.py --config config\defender25.json --master udp:127.0.0.1:14550
```

The build script creates `dist\gradis_defender25_package.gradispack`, a zip
compatible bundle with a project-specific extension.

AI navigation flow:

```text
edge VLM / planner / operator
  -> POST /api/nav_commands
  -> Core stores auditable command
  -> companion polls /api/nav_commands/{drone_id}/next
  -> MAVLink GUIDED command moves the aircraft
```

Example manual command:

```powershell
$body = @{
  drone_id = "GRADIS-DEFENDER25-01"
  source = "operator"
  action = "GOTO"
  lat = 37.52860
  lon = 126.96520
  alt_m = 20
  reason = "inspect incident zone"
  ttl_s = 20
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8088/api/nav_commands -ContentType application/json -Body $body
```

Safety boundary:

- Defender 25 is a Betaflight FPV craft. Official product copy does not expose
  an autonomous-flight SDK.
- Low-battery dock return is implemented in the companion layer. It needs a
  MAVLink-capable autopilot/bridge and surveyed dock coordinates before outdoor
  flight.
- The first real-world run must be props-off, then tethered, then low-altitude
  geofenced.
