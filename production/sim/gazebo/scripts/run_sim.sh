#!/usr/bin/env bash
# GRADIS 디지털 트윈 (Gazebo) — 원클릭 기동. Ubuntu 22.04/24.04 + RTX 권장.
#
# 부팅 순서:
#   1. GRADIS Core(관제)            : :8088
#   2. Gazebo (도시+M4TD+Dock3+액터) : ArduPilotPlugin이 :9002에서 SITL 대기
#   3. ArduCopter SITL (실펌웨어)    : --model JSON → Gazebo 물리와 락스텝
#   4. camera_bridge (gz→MJPEG)     : :8881
#   5. edge.agent (YOLO detector+MMPose+아바타+VLM)  : 프로덕션 스택 그대로
#   6. mavlink_companion (펌웨어층)  : tcp:5760, autopilot 순찰 시작
#
# 사전 1회: ./fetch_assets.sh  (액터 애니메이션)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
GZSIM="$HERE/.."
PROD="$(cd "$GZSIM/../.." && pwd)"

VLM_MODEL="${VLM_MODEL:-qwen2.5vl:3b}"
ARDUPILOT="${ARDUPILOT:-$HOME/ardupilot}"   # ArduPilot 소스 (sim_vehicle용)
WORLD="${WORLD:-gradis_city_night}"          # 기본 = 야간 엣지케이스 월드 (주간: gradis_city)

export GZ_SIM_RESOURCE_PATH="$GZSIM/models:$GZ_SIM_RESOURCE_PATH"
# ardupilot_gazebo 플러그인 (빌드 위치 다르면 수정)
export GZ_SIM_SYSTEM_PLUGIN_PATH="${ARDUPILOT_GAZEBO:-$HOME/ardupilot_gazebo/build}:$GZ_SIM_SYSTEM_PLUGIN_PATH"

echo "[sim] 1/6 GRADIS Core"
(cd "$PROD" && python3 core/server.py) & CORE_PID=$!
sleep 2

echo "[sim] 2/6 Gazebo (월드: $WORLD)"
gz sim -r "$GZSIM/worlds/$WORLD.sdf" & GZ_PID=$!
sleep 8

echo "[sim] 3/6 ArduCopter SITL (JSON FDM, home=DOCK3-A) — M4TD 자율스택 대역"
(cd "$ARDUPILOT" && Tools/autotest/sim_vehicle.py -v ArduCopter \
    -f JSON --model JSON \
    --add-param-file="$GZSIM/sitl/m4td.parm" \
    -l 37.52860,126.96520,38,0 \
    --no-mavproxy) & SITL_PID=$!
sleep 10

echo "[sim] 4/6 camera bridge (gz → MJPEG :8881)"
python3 "$HERE/camera_bridge.py" & BRIDGE_PID=$!
sleep 2

echo "[sim] 5/6 edge.agent (프로덕션 스택: detector → MMPose → 아바타 → VLM 툴 컨트롤)"
(cd "$PROD" && python3 -m edge.agent --source rtsp \
    --url http://127.0.0.1:8881/cam.mjpg \
    --drone-id GRADIS-M4TD-01 --zone "Riverside Park Area 2 (GZ-SIM)" \
    --lat 37.52860 --lon 126.96520 \
    --detector-model yolo11x.pt --pose-preset rtmpose-m --imgsz 1280 --tiles 1 --avatar \
    --vlm ollama --vlm-model "$VLM_MODEL" --vlm-interval 1.5) & AGENT_PID=$!

echo "[sim] 6/6 SITL EKF 수렴 대기(40s) 후 companion (autopilot 순찰)"
sleep 40
(cd "$PROD" && python3 firmware/companion/mavlink_companion.py \
    --master tcp:127.0.0.1:5760 --config config/m4td_sim.json \
    --node-id GRADIS-M4TD-01) & COMP_PID=$!

cat <<EOF

==========================================================
 GRADIS DIGITAL TWIN (GAZEBO) UP
  관제 대시보드 : http://127.0.0.1:8088/
  드론 카메라   : http://127.0.0.1:8881/cam.mjpg
  시나리오     : (15,12) 몸싸움 2인 / (12,9) 쓰러진 사람 /
                 (-12,-9) 행인(대조군)
  기대 동작    : autopilot 순찰 → VLM이 광장에서 fight/medical 판정
                 → autopilot off + goto + thermal 툴 콜 → ESCALATE
  종료: Ctrl-C (전 프로세스 정리)
==========================================================
EOF

trap 'kill $CORE_PID $GZ_PID $SITL_PID $BRIDGE_PID $AGENT_PID $COMP_PID 2>/dev/null' EXIT
wait
