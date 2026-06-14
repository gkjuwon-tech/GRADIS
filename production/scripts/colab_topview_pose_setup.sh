#!/usr/bin/env bash
# GRADIS Colab bootstrap for the top-view pose pipeline.
#
# Why this script exists in its current form:
#   Colab ships Python 3.12 + a very new torch (2.11+cu128). OpenMMLab does NOT
#   publish prebuilt mmcv wheels for that torch, so `mim install mmcv` falls back
#   to a from-source CUDA build. On Python 3.12 that build trips over the old
#   pkg_resources/`pkgutil.ImpImporter` path and the setuptools version dance.
#   That is the entire source of the "upload zip -> wait -> crash -> repeat" loop.
#
#   The fix is to stop gambling on a source build: we pin torch to a version that
#   HAS prebuilt mmcv wheels and install mmcv from the OpenMMLab wheel index. No
#   compiler, no mim, no ImpImporter, deterministic every run.
#
# Usage in Colab (run this as the FIRST cell, after a runtime restart):
#   %cd /content
#   from google.colab import files; files.upload()          # pick the bundle zip
#   !rm -rf /content/gradis_m4td
#   !unzip -q /content/colab_m4td_topview_pose_bundle_v1.zip -d /content/gradis_m4td
#   %cd /content/gradis_m4td/production
#   !bash scripts/colab_topview_pose_setup.sh _media/real_world_hard/caviar_walk_by_shop_front.mpg --max-frames 160
#
# Re-running without re-uploading is safe: every step is idempotent.

set -euo pipefail

VIDEO="${1:-}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pinned, prebuilt-wheel combo. The only cp312 mmcv wheel OpenMMLab ships is
# 2.2.0, built against the torch 2.4.0 series — so we pin torch to 2.4.1+cu121
# (same minor => ABI-compatible) and pull mmcv from the matching wheel index.
TORCH_VER="${TORCH_VER:-2.4.1}"
TV_VER="${TV_VER:-0.19.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"
MMCV_VER="${MMCV_VER:-2.2.0}"
# The mmcv wheel index is keyed by torch SERIES (major.minor.0), e.g. torch2.4.0.
TORCH_SERIES="${TORCH_VER%.*}.0"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_SERIES}/index.html"

# One source of truth for the "everything else" constraint set, reapplied after
# the heavy installs so torch/mmcv can't silently drag these out of range again.
PINS=(
  "requests==2.32.4"
  "tqdm>=4.67,<5"
  "rich>=13.8,<14"
  "urllib3>=2,<3"
  "filelock>=3.15"
  "pillow<11"
  "numpy==1.26.4"
  "pandas==2.2.2"
  "jedi>=0.16"
)
# OpenMMLab (mmcv 2.2.0 / mmpose / xtcocotools) is a numpy<2 world. Colab's stock
# opencv 4.13 demands numpy>=2, which would force a numpy2 cv2 onto a numpy1
# runtime -> "numpy.dtype size changed" ABI crash. Pin opencv to its last numpy1
# build so the whole stack agrees on numpy 1.26.4.
OPENCV_PIN="opencv-python-headless==4.10.0.84"

echo "[colab] 0/6 preflight"
python - <<'PY'
import os, sys, platform
print("  python :", sys.version.split()[0], "| cwd:", os.getcwd())
try:
    import torch
    print("  torch  :", torch.__version__, "| cuda avail:", torch.cuda.is_available())
except Exception as e:
    print("  torch  : (not importable yet)", e)
PY

echo "[colab] 1/6 system packages"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq ffmpeg git || true
fi

echo "[colab] 2/6 toolchain (setuptools that is sane on py3.12)"
# setuptools 69.x has pkg_resources WITHOUT the dead pkgutil.ImpImporter reference,
# so nothing in the OpenMMLab chain can resurrect that crash.
python -m pip install -q -U pip wheel
python -m pip install -q -U "setuptools==69.5.1"

echo "[colab] 3/6 pin torch ${TORCH_VER}+${CUDA_TAG} (gives us prebuilt mmcv)"
python -m pip install -q \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "[colab] 4/6 OpenMMLab stack from prebuilt wheels (no source build, no mim)"
python -m pip install -q "mmengine>=0.10.4"
# mmcv: prebuilt CUDA wheel matched to the pinned torch series. This is the step
# that used to compile for 20 minutes and die — now it's a wheel download.
python -m pip install -q "mmcv==${MMCV_VER}" -f "${MMCV_INDEX}"
# mmdet is required EVEN for pure top-down: importing mmpose.apis eagerly loads
# all heads, and the RTMO hybrid head does `from mmdet.utils import ...`. It is a
# pure-python wheel (no compile), so this is cheap.
python -m pip install -q "mmdet==3.3.0"
#
# mmpose also lists `chumpy` as a dep — that's SMPL/3D-body only, and its build
# is broken on Python 3.12 ("getting requirements to build wheel" dies). We only
# do 2D top-down, so install mmpose --no-deps and add just the inference deps it
# actually needs (json-tricks/munkres are pure-python wheels; xtcocotools is a
# tiny cython source build that compiles fine, unlike chumpy).
python -m pip install -q --no-deps "mmpose==1.3.2"
python -m pip install -q "json-tricks" "munkres"
# NOTE: xtcocotools is built later (step 5) — it must compile against the FINAL
# pinned numpy, so it cannot go before the numpy lock.

# mmpose 1.3.2 hard-asserts mmcv < 2.2.0 at import time, but the only prebuilt
# cp312 wheel is exactly 2.2.0 (API-compatible). Lift the ceiling in the installed
# package. find_spec locates the file WITHOUT importing (so the assert never runs).
python - <<'PY'
import importlib.util, re
for pkg in ("mmpose", "mmdet"):
    spec = importlib.util.find_spec(pkg)
    if not spec or not spec.origin:
        continue
    src = open(spec.origin, encoding="utf-8").read()
    new = re.sub(r"(mm\w+_maximum_version\s*=\s*)['\"][\d.]+['\"]",
                 r"\g<1>'9.9.9'", src)
    if new != src:
        open(spec.origin, "w", encoding="utf-8").write(new)
        print(f"  lifted version ceiling in {spec.origin}")
PY

echo "[colab] 5/6 app deps + re-pin the constraint set + numpy ABI lock"
# Remove every opencv variant first so a single numpy1-built cv2 wins (Colab
# stacks opencv-python + contrib + headless, any of which can shadow cv2).
python -m pip uninstall -y -q opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless || true
python -m pip install -q -U ultralytics pygltflib scipy
python -m pip install -q --no-deps "${OPENCV_PIN}"
python -m pip install -q -U "${PINS[@]}"
# Final, clean numpy lock. Uninstall FIRST (possibly twice, in case a stacked
# numpy2 install left a mtrand.so behind) then install fresh with no cache — a
# mixed numpy1/numpy2 tree is exactly what throws "numpy.dtype size changed".
python -m pip uninstall -y -q numpy || true
python -m pip uninstall -y -q numpy || true
python -m pip install -q --no-cache-dir --force-reinstall --no-deps "numpy==1.26.4"
# xtcocotools last, compiled against the now-final numpy 1.26.4.
python -m pip install -q "cython>=0.29"
python -m pip install -q --force-reinstall --no-deps --no-build-isolation "xtcocotools>=1.12"

# MMPose configs (e.g. projects/rtmpose/...) live in the git tree, not the wheel.
# Clone for configs only — do NOT `pip install -e` it (that triggers a build).
MMPOSE_ROOT="${MMPOSE_ROOT:-/content/mmpose}"
if [ ! -d "$MMPOSE_ROOT/.git" ]; then
  git clone --depth 1 https://github.com/open-mmlab/mmpose.git "$MMPOSE_ROOT"
fi
export MMPOSE_ROOT

echo "[colab] 6/6 HARD verification gate (fail loud BEFORE touching video)"
python - <<'PY'
import importlib, os, sys
fails = []
for mod in ("torch", "torchvision", "mmengine", "mmcv", "mmpose",
            "ultralytics", "cv2", "pygltflib", "scipy"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok  {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        fails.append((mod, repr(e)))
        print(f"  FAIL {mod:13s} {e!r}")

# numpy ABI smoke test: a numpy1 runtime with a numpy2-built extension throws
# "numpy.dtype size changed" exactly here. Exercise the paths that tripped it
# (numpy.random + cv2 array round-trip) so it fails in the gate, not mid-render.
try:
    import numpy as _np
    _np.random.default_rng(0).random(8)
    import cv2 as _cv2
    _cv2.cvtColor(_np.zeros((4, 4, 3), _np.uint8), _cv2.COLOR_BGR2RGB)
    print(f"  ok  numpy ABI     (numpy {_np.__version__}, cv2 {_cv2.__version__})")
except Exception as e:
    fails.append(("numpy-abi", repr(e)))
    print(f"  FAIL numpy ABI    {e!r}")

# Prove mmcv's compiled ops actually loaded (this is what a bad build breaks).
try:
    from mmcv.ops import nms  # noqa: F401
    print("  ok  mmcv.ops      (compiled ops load)")
except Exception as e:
    fails.append(("mmcv.ops", repr(e)))
    print(f"  FAIL mmcv.ops     {e!r}")

# The real failure surface: backends.py does `from mmpose.apis import ...`, which
# eagerly loads every head (RTMO -> mmdet). Importing the package alone is NOT
# enough to prove this works, so exercise the exact import path here.
try:
    from mmpose.apis import inference_topdown, init_model  # noqa: F401
    print("  ok  mmpose.apis   (heads + mmdet resolve)")
except Exception as e:
    fails.append(("mmpose.apis", repr(e)))
    print(f"  FAIL mmpose.apis  {e!r}")

# Prove the RTMPose config our backend asks for is resolvable.
try:
    from edge.pose.backends import RTMPOSE_M_CONFIG, _resolve_mmpose_config
    path = _resolve_mmpose_config(RTMPOSE_M_CONFIG)
    print("  resolved RTMPose config:", path)
    if not os.path.exists(path):
        fails.append(("rtmpose-config", path))
        print("  FAIL config path does not exist")
    else:
        print("  ok  rtmpose config on disk")
except Exception as e:
    fails.append(("config-resolve", repr(e)))
    print(f"  FAIL config-resolve {e!r}")

if fails:
    print("\n[colab] VERIFICATION FAILED — stopping before the video so you don't")
    print("        burn time on a half-installed stack. Failures:")
    for name, why in fails:
        print(f"          - {name}: {why}")
    sys.exit(1)
print("\n[colab] verification passed — environment is good.")
PY

if [ -z "$VIDEO" ]; then
  echo "[colab] setup complete, no video argument supplied."
  echo "[colab] Example:"
  echo "  python tools/avatar_on_video.py _media/real_world_hard/caviar_walk_by_shop_front.mpg --model yolo11x.pt --pose-preset rtmpose-m --max-frames 160"
  exit 0
fi

# Aerial/top-view detector defaults (env-overridable):
#   CONF=0.2 TILES=2: small overhead people score low and are tiny -> lower conf +
#                     sliced inference can recover them.
echo "[colab] running avatar (mannequin) pipeline on: $VIDEO"
python tools/avatar_on_video.py "$VIDEO" \
  --model yolo11x.pt \
  --pose-preset rtmpose-m \
  --imgsz "${IMGSZ:-1280}" \
  --conf "${CONF:-0.2}" \
  --tiles "${TILES:-2}" \
  --max-frames "${MAX_FRAMES:-180}" \
  "$@"

echo
echo "[colab] DONE. Mannequin outputs are next to the source video:"
echo "  $(dirname "$VIDEO")/avatar_$(basename "${VIDEO%.*}").mp4"
echo "  $(dirname "$VIDEO")/avatar_$(basename "${VIDEO%.*}").gif"
