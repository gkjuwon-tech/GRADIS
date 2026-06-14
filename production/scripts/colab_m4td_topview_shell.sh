#!/usr/bin/env bash
# GRADIS M4TD Colab shell.
#
# Run after unzipping the M4TD Colab bundle:
#   cd /content/gradis_m4td/production
#   bash scripts/colab_m4td_topview_shell.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAMPLE="${1:-_media/real_world_hard/caviar_walk_by_shop_front.mpg}"
MAX_FRAMES="${MAX_FRAMES:-160}"
POSE_PRESET="${POSE_PRESET:-rtmpose-m}"
DETECTOR_MODEL="${DETECTOR_MODEL:-yolo11x.pt}"

# setup.sh runs as a separate bash process, so its `export MMPOSE_ROOT` does not
# reach the secondary tools below. Set it here too so standalone runs of
# extract_on_video can resolve the MMPose config (projects/rtmpose/...).
export MMPOSE_ROOT="${MMPOSE_ROOT:-/content/mmpose}"

echo "[m4td-colab] setup + mannequin pass (this produces avatar_*.mp4 / .gif)"
bash scripts/colab_topview_pose_setup.sh "$SAMPLE" \
  --model "$DETECTOR_MODEL" \
  --pose-preset "$POSE_PRESET" \
  --max-frames "$MAX_FRAMES"

# --- Secondary diagnostics. The mannequin video is ALREADY produced above, so
# --- these are best-effort: a failure here must never bury the prize.
echo "[m4td-colab] detector recall check (optional)"
python tools/recall_compare.py "$SAMPLE" \
  --model "$DETECTOR_MODEL" \
  --imgsz 1280 \
  --conf 0.25 \
  --tiles 2 \
  --n 12 || echo "[m4td-colab] recall check skipped (non-fatal)"

echo "[m4td-colab] extraction overlay check (optional)"
python tools/extract_on_video.py "$SAMPLE" \
  --model "$DETECTOR_MODEL" \
  --pose-preset "$POSE_PRESET" \
  --imgsz 1280 \
  --conf 0.25 \
  --tiles 1 \
  --aerial \
  --max-frames "$MAX_FRAMES" || echo "[m4td-colab] extraction overlay skipped (non-fatal)"

cat <<'EOF'

[m4td-colab] done

Quality pass:
  POSE_PRESET=vitpose-s bash scripts/colab_m4td_topview_shell.sh _media/real_world_hard/caviar_walk_by_shop_front.mpg

More top-view samples:
  bash scripts/colab_m4td_topview_shell.sh _media/real_world_hard/caviar_walk_by_shop_cor.mpg
  bash scripts/colab_m4td_topview_shell.sh _media/real_world_hard/caviar_meet_crowd.mpg
  bash scripts/colab_m4td_topview_shell.sh _media/real_world_hard/caviar_fall_on_floor.mpg
EOF

