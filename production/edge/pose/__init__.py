"""
GRADIS pose extraction package.

Production pose extraction is a split pipeline:
    YOLO detector/tracker -> MMPose RTMPose/ViTPose -> SkeletonEngine

Raw camera frames stay local. Only normalized keypoints and anonymous
mannequin/redacted frames cross the privacy boundary.
"""

import time

from . import skeleton as topo
from .backends import (
    MMPoseTopDownPose,
    TopViewPoseBackend,
    YoloPersonDetector,
    make_backend,
)
from .engine import Person, SkeletonEngine


class PoseEstimator:
    def __init__(self, backend="topview", **kwargs):
        self.backend = make_backend(backend, **kwargs)
        self.engine = SkeletonEngine()
        self.t0 = time.time()

    def __call__(self, frame_bgr, t=None):
        if t is None:
            t = time.time() - self.t0
        raw_dets = self.backend(frame_bgr)
        people = self.engine.process(raw_dets, t)
        return [p.as_tuple() for p in people]


__all__ = [
    "MMPoseTopDownPose",
    "Person",
    "PoseEstimator",
    "SkeletonEngine",
    "TopViewPoseBackend",
    "YoloPersonDetector",
    "make_backend",
    "topo",
]

