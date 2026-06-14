#!/usr/bin/env bash
# 액터 애니메이션 DAE 다운로드 (Gazebo Fuel 표준 액터 — walk/run)
# 고품질 업그레이드: Mixamo 격투/낙상 FBX → Blender로 DAE 변환 후 교체.
set -e
cd "$(dirname "$0")/../assets/animations"

BASE="https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes"
for f in walk.dae run.dae moonwalk.dae; do
  if [ ! -f "$f" ]; then
    echo "[fetch] $f"
    curl -sfL -o "$f" "$BASE/$f"
  else
    echo "[fetch] $f 있음 — 스킵"
  fi
done
echo "[fetch] done: $(ls *.dae 2>/dev/null | tr '\n' ' ')"
