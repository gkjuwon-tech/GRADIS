"""
Pose backends for the GRADIS anonymization boundary.

The production path no longer uses YOLO's pose head. YOLO is only a person detector
and tracker; body keypoints are estimated by a separate MMPose top-down model
such as RTMPose or ViTPose. This split is important for drone/top-view video,
where one-stage COCO pose heads often hallucinate side-view walking limbs.

Return format:
    list[dict] where each dict contains:
      kp: np.ndarray[17, 3] normalized COCO keypoints (x, y, confidence)
      bbox: np.ndarray[4] normalized xyxy person box
      score: float detector confidence
      track_id: optional int from BoT-SORT/ByteTrack
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


COCO_PERSON_CLASS_ID = 0

RTMPOSE_M_CONFIG = (
    "projects/rtmpose/rtmpose/body_2d_keypoint/"
    "rtmpose-m_8xb256-420e_coco-256x192.py"
)
RTMPOSE_M_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
)
VITPOSE_S_CONFIG = (
    "configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192.py"
)
VITPOSE_S_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192-4c101a76_20230314.pth"
)


def _as_float_array(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _clip_box_xyxy(box, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = np.clip(x0, 0, max(0, width - 1))
    x1 = np.clip(x1, 0, max(0, width - 1))
    y0 = np.clip(y0, 0, max(0, height - 1))
    y1 = np.clip(y1, 0, max(0, height - 1))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _resolve_mmpose_config(path_or_name: str) -> str:
    """Resolve MMPose config paths for both pip installs and git clones."""
    if os.path.exists(path_or_name) or path_or_name.startswith(("http://", "https://")):
        return path_or_name

    search_roots = []
    env_root = os.environ.get("MMPOSE_ROOT")
    if env_root:
        search_roots.append(env_root)
    try:
        import mmpose

        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(mmpose.__file__)))
        search_roots.append(pkg_root)
    except Exception:
        pass
    search_roots.extend([os.getcwd(), os.path.join(os.getcwd(), "mmpose")])

    for root in search_roots:
        candidate = os.path.join(root, path_or_name)
        if os.path.exists(candidate):
            return candidate
    return path_or_name


@dataclass
class PersonBox:
    xyxy: np.ndarray
    score: float
    track_id: int | None = None

    def normalized(self, width: int, height: int) -> np.ndarray:
        x0, y0, x1, y1 = self.xyxy
        return np.array([x0 / width, y0 / height, x1 / width, y1 / height], np.float32)


class YoloPersonDetector:
    """Ultralytics YOLO detector/tracker limited to class 0 (person)."""

    def __init__(
        self,
        model: str = "yolo11x.pt",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.5,
        tracker: str = "botsort.yaml",
        device: str | None = None,
        half: bool = True,
        tiles: int = 1,
        tile_overlap: float = 0.2,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Missing detector dependency. Install with: pip install ultralytics"
            ) from e
        if model.endswith("-pose.pt"):
            raise ValueError(
                "YOLO pose-head weights are not allowed in the top-view pipeline. "
                "Use detector weights such as yolo11x.pt, yolo11l.pt, or a "
                "person/aerial fine-tuned detection model."
            )
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.device = device
        self.half = half
        self.tiles = max(1, int(tiles))
        self.tile_overlap = float(tile_overlap)

    def __call__(self, frame_bgr) -> list[PersonBox]:
        boxes = self._detect_full_frame(frame_bgr)
        if self.tiles > 1:
            boxes.extend(self._detect_tiles(frame_bgr))
        return _nms_boxes(boxes, iou_thr=0.55)

    def _predict(self, imgs, *, track: bool):
        common = dict(
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[COCO_PERSON_CLASS_ID],
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if track and self.tracker:
            return self.model.track(
                imgs,
                persist=True,
                tracker=self.tracker,
                **common,
            )
        return self.model.predict(imgs, **common)

    def _detect_full_frame(self, frame_bgr) -> list[PersonBox]:
        height, width = frame_bgr.shape[:2]
        results = self._predict(frame_bgr, track=True)
        return _ultralytics_results_to_boxes(results, width, height)

    def _detect_tiles(self, frame_bgr) -> list[PersonBox]:
        height, width = frame_bgr.shape[:2]
        imgs, offsets = [], []
        n = self.tiles
        tw, th = width / n, height / n
        for r in range(n):
            for c in range(n):
                x0 = int(max(0, c * tw - self.tile_overlap * tw))
                x1 = int(min(width, (c + 1) * tw + self.tile_overlap * tw))
                y0 = int(max(0, r * th - self.tile_overlap * th))
                y1 = int(min(height, (r + 1) * th + self.tile_overlap * th))
                if x1 > x0 and y1 > y0:
                    imgs.append(frame_bgr[y0:y1, x0:x1])
                    offsets.append((x0, y0))
        if not imgs:
            return []
        out = []
        for result, (ox, oy), img in zip(self._predict(imgs, track=False), offsets, imgs):
            local = _ultralytics_results_to_boxes([result], img.shape[1], img.shape[0])
            for box in local:
                shifted = box.xyxy.copy()
                shifted[[0, 2]] += ox
                shifted[[1, 3]] += oy
                shifted = _clip_box_xyxy(shifted, width, height)
                out.append(PersonBox(shifted, box.score, None))
        return out


def _ultralytics_results_to_boxes(results: Iterable, width: int, height: int) -> list[PersonBox]:
    out: list[PersonBox] = []
    for result in results:
        if result.boxes is None:
            continue
        xyxy = _as_float_array(result.boxes.xyxy)
        conf = _as_float_array(result.boxes.conf)
        cls = _as_float_array(result.boxes.cls).astype(np.int32)
        ids = None
        if getattr(result.boxes, "id", None) is not None:
            ids = _as_float_array(result.boxes.id).astype(np.int32)
        for i, box in enumerate(xyxy):
            if int(cls[i]) != COCO_PERSON_CLASS_ID:
                continue
            track_id = int(ids[i]) if ids is not None and i < len(ids) else None
            out.append(
                PersonBox(
                    xyxy=_clip_box_xyxy(box, width, height),
                    score=float(conf[i]),
                    track_id=track_id,
                )
            )
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = aa + bb - inter
    return inter / denom if denom > 1e-6 else 0.0


def _nms_boxes(boxes: list[PersonBox], iou_thr: float = 0.55) -> list[PersonBox]:
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: list[PersonBox] = []
    for box in boxes:
        if all(_iou(box.xyxy, old.xyxy) <= iou_thr for old in kept):
            kept.append(box)
    return kept


class MMPoseTopDownPose:
    """Top-down keypoint estimator using MMPose Python APIs."""

    def __init__(
        self,
        config: str | None = None,
        checkpoint: str | None = None,
        preset: str = "rtmpose-m",
        device: str | None = None,
    ):
        if config is None or checkpoint is None:
            config, checkpoint = preset_paths(preset)
        self.config = _resolve_mmpose_config(config)
        self.checkpoint = checkpoint
        self.device = device or _auto_device()
        try:
            from mmpose.apis import inference_topdown, init_model
        except ImportError as e:
            raise RuntimeError(
                "Missing MMPose. In Colab run scripts/colab_topview_pose_setup.sh, "
                "or install OpenMMLab packages manually."
            ) from e
        self._inference_topdown = inference_topdown
        self.model = init_model(self.config, self.checkpoint, device=self.device)

    def __call__(self, frame_bgr, boxes: list[PersonBox]) -> list[dict]:
        if not boxes:
            return []
        height, width = frame_bgr.shape[:2]
        bboxes = np.stack([b.xyxy for b in boxes]).astype(np.float32)
        try:
            samples = self._inference_topdown(self.model, frame_bgr, bboxes=bboxes)
        except TypeError:
            samples = self._inference_topdown(self.model, frame_bgr, bboxes)

        dets = []
        for i, sample in enumerate(samples):
            pred = sample.pred_instances
            keypoints = _as_float_array(pred.keypoints)
            raw_scores = getattr(pred, "keypoint_scores", None)
            scores = None if raw_scores is None else _as_float_array(raw_scores)
            if keypoints.ndim == 3:
                keypoints = keypoints[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]
            if keypoints.shape[0] < 17:
                continue
            kp = np.zeros((17, 3), np.float32)
            kp[:, 0] = np.clip(keypoints[:17, 0] / width, 0.0, 1.0)
            kp[:, 1] = np.clip(keypoints[:17, 1] / height, 0.0, 1.0)
            kp[:, 2] = np.clip(scores[:17] if scores is not None else 1.0, 0.0, 1.0)
            box = boxes[min(i, len(boxes) - 1)]
            dets.append(
                {
                    "kp": kp,
                    "bbox": box.normalized(width, height),
                    "score": box.score,
                    "track_id": box.track_id,
                }
            )
        return dets


def preset_paths(preset: str) -> tuple[str, str]:
    preset = (preset or "rtmpose-m").lower()
    if preset in {"rtmpose", "rtmpose-m", "rtm"}:
        return RTMPOSE_M_CONFIG, RTMPOSE_M_CHECKPOINT
    if preset in {"vitpose", "vitpose-s", "vit"}:
        return VITPOSE_S_CONFIG, VITPOSE_S_CHECKPOINT
    raise ValueError(f"unknown pose preset: {preset}")


def _auto_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TopViewPoseBackend:
    """Detector/tracker + MMPose pose estimator + top-view confidence gate."""

    def __init__(
        self,
        detector_model: str = "yolo11x.pt",
        model: str | None = None,
        pose_config: str | None = None,
        pose_checkpoint: str | None = None,
        pose_preset: str = "rtmpose-m",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.5,
        tracker: str = "botsort.yaml",
        device: str | None = None,
        half: bool = True,
        tiles: int = 1,
        topview_limb_conf: float = 0.35,
    ):
        detector_model = model or detector_model
        self.detector = YoloPersonDetector(
            model=detector_model,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            tracker=tracker,
            device=device,
            half=half,
            tiles=tiles,
        )
        self.pose = MMPoseTopDownPose(
            config=pose_config,
            checkpoint=pose_checkpoint,
            preset=pose_preset,
            device=device,
        )
        self.topview_limb_conf = float(topview_limb_conf)

    def __call__(self, frame_bgr) -> list[dict]:
        boxes = self.detector(frame_bgr)
        detections = self.pose(frame_bgr, boxes)
        for det in detections:
            det["kp"] = self._gate_topview_limbs(det["kp"])
        return detections

    def _gate_topview_limbs(self, kp: np.ndarray) -> np.ndarray:
        """Reduce hallucinated limb influence in top-view frames.

        We keep head/torso cues whenever possible and require higher confidence
        for elbows, wrists, knees, and ankles. The downstream tracker can then
        coast/recover instead of snapping the mannequin to a side-view fantasy.
        """
        gated = kp.copy()
        limb_ids = [7, 8, 9, 10, 13, 14, 15, 16]
        low = gated[limb_ids, 2] < self.topview_limb_conf
        gated[np.array(limb_ids)[low], 2] = 0.0
        return gated


def make_backend(name: str = "topview", **kw):
    key = (name or "topview").lower()
    if key in {"rtmpose-onnx", "onnx", "rtmlib", "rtmpose_onnx"}:
        # No-OpenMMLab, container-runnable path (onnxruntime only).
        from .rtmpose_onnx import RTMPoseOnnxBackend

        allowed = {
            "mode", "device", "backend", "det_conf",
            "kpt_conf", "topview_limb_conf", "bbox_pad",
        }
        return RTMPoseOnnxBackend(**{k: v for k, v in kw.items() if k in allowed})
    if key in {"topview", "mmpose", "rtmpose", "vitpose"}:
        if key == "vitpose" and "pose_preset" not in kw:
            kw["pose_preset"] = "vitpose-s"
        if key == "rtmpose" and "pose_preset" not in kw:
            kw["pose_preset"] = "rtmpose-m"
        return TopViewPoseBackend(**kw)
    if key in {"yolo", "yolo-pose", "yolopose"}:
        raise ValueError(
            "The old YOLO pose-head backend was removed. Use backend='topview' or "
            "backend='mmpose' with YOLO detector weights plus MMPose pose weights."
        )
    raise ValueError(f"unknown pose backend: {name}")
