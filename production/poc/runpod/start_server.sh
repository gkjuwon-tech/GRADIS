#!/usr/bin/env bash
# GRADIS Mini 2 + Litchi PoC server launcher for RunPod.
#
# Laptop tunnel:
#   ssh -L 8088:127.0.0.1:8088 -L 0.0.0.0:1935:127.0.0.1:1935 <pod>
#
# Litchi pushes:
#   rtmp://<laptop-lan-ip>:1935/gradis-mini2
#
# mediamtx receives RTMP and exposes:
#   rtsp://127.0.0.1:8554/gradis-mini2
set -e

GRADIS="${GRADIS:-/workspace/gradis}"
VLM_MODEL="${VLM_MODEL:-qwen2.5vl:7b}"
export MMPOSE_ROOT="${MMPOSE_ROOT:-/workspace/mmpose}"

echo "[server] 1/4 ollama"
pgrep -f "ollama serve" >/dev/null || (ollama serve >/workspace/ollama.log 2>&1 &)
sleep 2

echo "[server] 2/4 mediamtx (RTMP :1935, RTSP :8554)"
pgrep -f mediamtx >/dev/null || (/workspace/mediamtx >/workspace/mediamtx.log 2>&1 &)
sleep 1

echo "[server] 3/4 GRADIS Core (:8088)"
pgrep -f "core/server.py" >/dev/null || \
  (cd "$GRADIS" && python3 core/server.py >/workspace/core.log 2>&1 &)
sleep 2

echo "[server] 4/4 edge.agent (Mini 2 Litchi stream)"
pgrep -f "edge.agent.*gradis-mini2" >/dev/null || \
  (cd "$GRADIS" && python3 -m edge.agent --source rtsp \
    --url rtsp://127.0.0.1:8554/gradis-mini2 \
    --drone-id GRADIS-MINI2-LITCHI-POC --zone "Mini 2 Litchi PoC Zone" \
    --lat 37.52860 --lon 126.96520 \
    --detector-model yolo11x.pt --pose-preset rtmpose-m --imgsz 1280 --tiles 1 --avatar \
    --vlm ollama --vlm-model "$VLM_MODEL" --vlm-interval 1.0 \
    >/workspace/agent.log 2>&1 &)

echo ""
echo "=========================================================="
echo " GRADIS MINI 2 + LITCHI PoC SERVER UP (RunPod)"
echo "  Core   : localhost:8088"
echo "  RTMP   : localhost:1935/gradis-mini2"
echo "  RTSP   : localhost:8554/gradis-mini2"
echo "  Logs   : tail -f /workspace/agent.log"
echo "=========================================================="
