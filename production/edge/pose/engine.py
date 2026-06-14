"""
SkeletonEngine — 추출 파이프라인 전체를 묶는 오케스트레이터.

  원시검출 → 해부학게이트 → 추적(ID) → 폐색복원 → 적응평활 → 정규화
                                  └──────── Track 내부에서 ────────┘

입력: list[dict{"kp": np[17,3] 정규화}]  (백엔드 또는 합성 GT)
출력: list[Person]
  Person.id        : 안정적 추적 ID (가해자/피해자 안 섞임)
  Person.kp        : np[17,3] 평활/복원된 키포인트 (이미지좌표)
  Person.ext       : np[25,3] GRADIS-25 확장 골격 (파생 8관절 포함, 0..16은 kp와 동일)
  Person.inferred  : np[17] 복원된(추정) 관절 표시
  Person.quality   : 0..1 이 스켈레톤을 얼마나 믿을지
  Person.norm      : np[25,2] 시점불변 정규화 좌표 (ML/학습용, 얼굴없음)

다운스트림(redact/렌더/업링크)은 (id, kp) 튜플만 쓰면 되도록 .as_tuple() 제공
(kp는 25관절 — 0..16이 COCO-17 그대로라 기존 코드와 호환, 17..24는 정밀 파생 관절).
"""
import numpy as np
from .tracker import MultiPersonTracker
from .skeleton import (anatomical_validity, normalize_pose, extend_keypoints,
                       N_KPT, AERIAL_KPT_WEIGHT)


class Person:
    __slots__ = ("id", "kp", "ext", "inferred", "quality", "norm")

    def __init__(self, id, kp, ext, inferred, quality, norm):
        self.id = id
        self.kp = kp
        self.ext = ext
        self.inferred = inferred
        self.quality = quality
        self.norm = norm

    def as_tuple(self):
        return (self.id, self.ext)


class SkeletonEngine:
    def __init__(self, max_age=12, gate=0.14, aerial=True, default_fps=12):
        self.tracker = MultiPersonTracker(max_age=max_age, gate=gate)
        self.aerial = aerial
        self.dt_default = 1.0 / default_fps
        self._t_prev = None

    def process(self, detections, t):
        dt = self.dt_default if self._t_prev is None else max(1e-3, t - self._t_prev)
        self._t_prev = t

        # 항공 시야: 얼굴 키포인트 신뢰도 하향(작고 불안정) → 추적/평활 안정화
        if self.aerial:
            dets = []
            for d in detections:
                kp = d["kp"].copy()
                kp[:, 2] = kp[:, 2] * AERIAL_KPT_WEIGHT
                dd = dict(d); dd["kp"] = kp
                dets.append(dd)
            detections = dets

        tracks = self.tracker.update(detections, t, dt)

        out = []
        for tr in tracks:
            q = anatomical_validity(tr.kp)
            # 복원된 관절 비율이 높으면 품질 감점(추정이 많을수록 덜 신뢰)
            inf_ratio = float(tr.inferred.mean())
            q = max(0.0, q * (1.0 - 0.4 * inf_ratio))
            ext = extend_keypoints(tr.kp)
            out.append(Person(tr.id, tr.kp.copy(), ext, tr.inferred.copy(),
                              q, normalize_pose(ext)))
        return out
