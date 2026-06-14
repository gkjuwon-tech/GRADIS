#!/usr/bin/env bash
# One-time setup for the GRADIS Mini 2 + Litchi PoC on a RunPod GPU instance.
#
# Recommended pod: RTX 3090/4090 class GPU, 30GB+ disk, SSH enabled.
# Copy production/ to /workspace/gradis before running this script.
set -e
cd /workspace

echo "[setup] 1/4 system packages"
apt-get update -qq
apt-get install -y -qq ffmpeg python3-pip curl git

echo "[setup] 2/4 Python dependencies"
pip3 install -q -U pip openmim
pip3 install -q ultralytics pillow opencv-python-headless pygltflib numpy scipy
python3 -m mim install -q "mmengine>=0.10.0"
python3 -m mim install -q "mmcv>=2.1.0"
python3 -m mim install -q "mmdet>=3.2.0"
if [ ! -d /workspace/mmpose/.git ]; then
  git clone --depth 1 https://github.com/open-mmlab/mmpose.git /workspace/mmpose
fi
pip3 install -q -e /workspace/mmpose
echo 'export MMPOSE_ROOT=/workspace/mmpose' >> /etc/profile.d/gradis_mmpose.sh

echo "[setup] 3/4 mediamtx (RTMP ingest + RTSP relay)"
if [ ! -f mediamtx ]; then
  curl -sL -o mtx.tar.gz \
    https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz
  tar xzf mtx.tar.gz mediamtx && rm mtx.tar.gz
fi

echo "[setup] 4/4 ollama + VLM model"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
(ollama serve >/dev/null 2>&1 &) ; sleep 3
ollama pull qwen2.5vl:7b

echo "[setup] done. Next: bash /workspace/gradis/poc/runpod/start_server.sh"
