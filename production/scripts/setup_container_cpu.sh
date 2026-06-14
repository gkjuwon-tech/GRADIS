#!/usr/bin/env bash
# GRADIS CPU container bootstrap for the top-view pose pipeline.
#
# This is the CPU/Python-3.11 sibling of scripts/colab_topview_pose_setup.sh.
# The Colab script targets py3.12 + CUDA; a headless CPU dev container needs a
# different pin set, but the SAME idea: never gamble on a from-source mmcv build,
# pin torch to a series that HAS prebuilt mmcv wheels, and pull mmcv from the
# matching wheel index. No compiler, no mim, deterministic every run.
#
# Usage (from production/):
#   bash scripts/setup_container_cpu.sh
#
# Re-running is safe: every step is idempotent. Intended to run inside a venv.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# CPU pins. The cp311 mmcv 2.2.0 wheel is built against the torch 2.4.0 series,
# so pin torch to 2.4.1+cpu (same minor => ABI-compatible) and pull mmcv from the
# cpu wheel index keyed by torch SERIES (major.minor.0).
TORCH_VER="${TORCH_VER:-2.4.1}"
TV_VER="${TV_VER:-0.19.1}"
CPU_TAG="cpu"
MMCV_VER="${MMCV_VER:-2.2.0}"
TORCH_SERIES="${TORCH_VER%.*}.0"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/${CPU_TAG}/torch${TORCH_SERIES}/index.html"

# numpy<2 world (OpenMMLab + xtcocotools). Keep the whole stack on numpy 1.26.4
# and the last numpy1 opencv build so nothing throws "numpy.dtype size changed".
PINS=(
  "requests==2.32.4"
  "tqdm>=4.67,<5"
  "rich>=13.8,<14"
  "urllib3>=2,<3"
  "filelock>=3.15"
  "pillow<11"
  "numpy==1.26.4"
)
OPENCV_PIN="opencv-python-headless==4.10.0.84"

echo "[setup] 0/6 preflight"
python - <<'PY'
import os, sys
print("  python :", sys.version.split()[0], "| cwd:", os.getcwd())
PY

echo "[setup] 1/6 system packages (best effort, no sudo assumed)"
if command -v apt-get >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq ffmpeg git) >/dev/null 2>&1 \
    || echo "[setup] apt unavailable; will rely on imageio-ffmpeg if needed"
fi

echo "[setup] 2/6 toolchain"
python -m pip install -q -U pip wheel
python -m pip install -q -U "setuptools<70"

echo "[setup] 3/6 pin torch ${TORCH_VER}+${CPU_TAG} (gives us prebuilt mmcv)"
python -m pip install -q \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
  --index-url "https://download.pytorch.org/whl/${CPU_TAG}"

echo "[setup] 4/6 OpenMMLab stack from prebuilt wheels (no source build, no mim)"
python -m pip install -q "mmengine>=0.10.4"
python -m pip install -q "mmcv==${MMCV_VER}" -f "${MMCV_INDEX}"
python -m pip install -q "mmdet==3.3.0"
# mmpose pulls chumpy (SMPL/3D only, build broken on new py) -> --no-deps + the
# tiny pure-python inference deps it actually needs.
python -m pip install -q --no-deps "mmpose==1.3.2"
python -m pip install -q "json-tricks" "munkres"

# mmpose 1.3.2 hard-asserts mmcv < 2.2.0 at import; the only matching wheel is
# exactly 2.2.0 (API-compatible). Lift the ceiling without importing the package.
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

echo "[setup] 5/6 app deps + numpy ABI lock"
python -m pip uninstall -y -q opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless || true
python -m pip install -q -U ultralytics pygltflib scipy onnxruntime imageio-ffmpeg
python -m pip install -q --no-deps "${OPENCV_PIN}"
python -m pip install -q -U "${PINS[@]}"
python -m pip uninstall -y -q numpy || true
python -m pip uninstall -y -q numpy || true
python -m pip install -q --no-cache-dir --force-reinstall --no-deps "numpy==1.26.4"
python -m pip install -q "cython>=0.29"
python -m pip install -q --force-reinstall --no-deps --no-build-isolation "xtcocotools>=1.12"

# MMPose configs (projects/rtmpose/...) live in the git tree, not the wheel.
MMPOSE_ROOT="${MMPOSE_ROOT:-$ROOT/_external/mmpose}"
mkdir -p "$(dirname "$MMPOSE_ROOT")"
if [ ! -d "$MMPOSE_ROOT/.git" ]; then
  git clone --depth 1 https://github.com/open-mmlab/mmpose.git "$MMPOSE_ROOT"
fi
export MMPOSE_ROOT

echo "[setup] 6/6 HARD verification gate"
MMPOSE_ROOT="$MMPOSE_ROOT" python - <<'PY'
import importlib, os, sys
fails = []
for mod in ("torch", "torchvision", "mmengine", "mmcv", "mmpose",
            "ultralytics", "cv2", "pygltflib", "scipy", "onnxruntime"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok  {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        fails.append((mod, repr(e)))
        print(f"  FAIL {mod:13s} {e!r}")
try:
    import numpy as _np
    _np.random.default_rng(0).random(8)
    import cv2 as _cv2
    _cv2.cvtColor(_np.zeros((4, 4, 3), _np.uint8), _cv2.COLOR_BGR2RGB)
    print(f"  ok  numpy ABI     (numpy {_np.__version__}, cv2 {_cv2.__version__})")
except Exception as e:
    fails.append(("numpy-abi", repr(e)))
    print(f"  FAIL numpy ABI    {e!r}")
try:
    from mmcv.ops import nms  # noqa: F401
    print("  ok  mmcv.ops      (compiled ops load)")
except Exception as e:
    fails.append(("mmcv.ops", repr(e)))
    print(f"  FAIL mmcv.ops     {e!r}")
try:
    from mmpose.apis import inference_topdown, init_model  # noqa: F401
    print("  ok  mmpose.apis   (heads + mmdet resolve)")
except Exception as e:
    fails.append(("mmpose.apis", repr(e)))
    print(f"  FAIL mmpose.apis  {e!r}")
sys.path.insert(0, os.getcwd())
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
    print("\n[setup] VERIFICATION FAILED:")
    for name, why in fails:
        print(f"          - {name}: {why}")
    sys.exit(1)
print("\n[setup] verification passed — environment is good.")
PY

echo "[setup] complete. Remember: export MMPOSE_ROOT=$MMPOSE_ROOT"
