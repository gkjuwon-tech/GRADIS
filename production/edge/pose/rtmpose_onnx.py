"""
RTMPose-ONNX backend — the no-OpenMMLab, container-runnable pose path.

Why this exists
---------------
The production top-view path (``backends.TopViewPoseBackend``) needs MMPose,
which on Python 3.12 / new torch drags in an mmcv *source* build. That build is
exactly the "upload zip -> wait 20 min -> ImpImporter crash -> repeat" loop that
burned a whole night. This backend gets the SAME RTMPose weights through
``rtmlib`` + ``onnxruntime`` instead: a wheel-only install, no compiler, no mim,
no CUDA required. It runs on a bare CPU container and downloads its ONNX weights
on first use, so verification needs zero manual uploads.

It returns the identical GRADIS detection format as ``backends`` so it is a drop
-in for ``SkeletonEngine``/``MannequinRenderer``:

    list[dict] with
      kp:       np.ndarray[17, 3]  normalized COCO keypoints (x, y, confidence)
      bbox:     np.ndarray[4]      normalized xyxy person box
      score:    float              detection confidence
      track_id: None               (SkeletonEngine assigns stable ids)

Top-view limb gating is shared with ``backends`` so the behavior signal is the
same regardless of which pose engine produced it.
"""

from __future__ import annotations

import numpy as np

# COCO-17 limb indices (elbows, wrists, knees, ankles). Reused for the top-view
# confidence gate so this backend and the MMPose one agree on what a "limb" is.
_LIMB_IDS = np.array([7, 8, 9, 10, 13, 14, 15, 16])


def _bbox_from_kp(kp_px: np.ndarray, scores: np.ndarray, conf: float, pad: float):
    """Derive a person box from visible keypoints (rtmlib gives no box back).

    Uses only confident joints so a single hallucinated point cannot blow the
    box up. Falls back to the full keypoint extent if nothing is confident.
    """
    vis = scores >= conf
    pts = kp_px[vis] if vis.any() else kp_px
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return np.array(
        [x0 - bw * pad, y0 - bh * pad, x1 + bw * pad, y1 + bh * pad],
        dtype=np.float32,
    )


class RTMPoseOnnxBackend:
    """Detector + RTMPose top-down via rtmlib/onnxruntime, GRADIS-format output."""

    def __init__(
        self,
        mode: str = "balanced",
        device: str | None = None,
        backend: str = "onnxruntime",
        det_conf: float = 0.20,
        kpt_conf: float = 0.25,
        topview_limb_conf: float = 0.35,
        bbox_pad: float = 0.12,
    ):
        try:
            from rtmlib import Body
        except ImportError as e:
            raise RuntimeError(
                "Missing rtmlib. Install the container pose path with: "
                "pip install rtmlib onnxruntime"
            ) from e
        if device is None:
            device = _auto_device()
        # rtmlib.Body = person detector (YOLOX) + RTMPose top-down. Weights are
        # fetched to ~/.cache/rtmlib on first call.
        self.body = Body(mode=mode, backend=backend, device=device)
        self.det_conf = float(det_conf)
        self.kpt_conf = float(kpt_conf)
        self.topview_limb_conf = float(topview_limb_conf)
        self.bbox_pad = float(bbox_pad)

    def __call__(self, frame_bgr) -> list[dict]:
        height, width = frame_bgr.shape[:2]
        keypoints, scores = self.body(frame_bgr)
        if keypoints is None or len(keypoints) == 0:
            return []
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        dets: list[dict] = []
        for kp_px, sc in zip(keypoints, scores):
            if kp_px.shape[0] < 17:
                continue
            kp_px, sc = kp_px[:17], sc[:17]
            # Detection confidence from the strongest joints, not the mean of all
            # 17. The mean is dragged below any useful threshold by occluded/gated
            # limbs, which made every real person look like noise to the tracker.
            # Top-8 separates true people (~0.5-0.7) from car/clutter FPs (~0.15).
            det_score = float(np.mean(np.sort(sc)[-8:]))
            if det_score < self.det_conf:
                continue
            kp = np.zeros((17, 3), np.float32)
            kp[:, 0] = np.clip(kp_px[:, 0] / width, 0.0, 1.0)
            kp[:, 1] = np.clip(kp_px[:, 1] / height, 0.0, 1.0)
            kp[:, 2] = np.clip(sc, 0.0, 1.0)
            kp = self._gate_topview_limbs(kp)
            box = _bbox_from_kp(kp_px, sc, self.kpt_conf, self.bbox_pad)
            box[[0, 2]] = np.clip(box[[0, 2]] / width, 0.0, 1.0)
            box[[1, 3]] = np.clip(box[[1, 3]] / height, 0.0, 1.0)
            dets.append(
                {"kp": kp, "bbox": box, "score": det_score, "track_id": None}
            )
        return dets

    def _gate_topview_limbs(self, kp: np.ndarray) -> np.ndarray:
        """Zero out low-confidence limbs so the tracker coasts instead of
        snapping the mannequin onto a hallucinated side-view leg."""
        gated = kp.copy()
        low = gated[_LIMB_IDS, 2] < self.topview_limb_conf
        gated[_LIMB_IDS[low], 2] = 0.0
        return gated


def _auto_device() -> str:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda"
    except Exception:
        pass
    return "cpu"
