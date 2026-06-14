"""
시점-강건 기하 안정화 (View-robust geometry).

탑뷰/드론 시점에서 단일 프레임 2D 해부학 검증은 원리적으로 약하다:
투영은 뼈를 '짧게'만 만들 수 있으므로(카메라 쪽을 향한 사지), 짧은 뼈가
'단축(foreshortening)'인지 '환각'인지 한 프레임만으론 구분 불가능하다.

그러나 한 가지 사영 불변식은 남는다:
    투영된 뼈 길이는 그 사람의 실제(3D) 뼈 길이를 절대 넘을 수 없다.

이걸 시간축으로 쓴다. 트랙별로 각 뼈의 '여태 관측된 최대 투영 길이'(사람
크기로 정규화)를 학습해두면, 어느 프레임에서 그 한계를 크게 넘는 사지는
물리적으로 불가능 → 환각이다. 그 distal 관절을 뼈 방향으로 한계 안쪽까지
끌어당긴다. 각도 하드코딩 없이, 그 사람 자신의 관측 이력만으로 환각을 잡는다.

scale은 시점 방향에 덜 민감한 bbox 대각선을 쓴다(torso는 탑뷰에서 같이 단축됨).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .skeleton import BONES, N_KPT

# distal(자식) -> proximal(부모). distal을 부모 기준으로 끌어당긴다.
_PARENT = {}
for _p, _c in BONES:
    _PARENT.setdefault(_c, _p)

# 끌어당겨도 되는(말단 쪽) 관절만. 어깨폭/엉덩이폭/몸통 같은 구조 뼈는 건드리지 않는다.
_DISTAL = {7, 8, 9, 10, 13, 14, 15, 16}  # elbows, wrists, knees, ankles

# 한계 = 관측 분포의 이 분위수 × slack. max가 아니라 분위수인 게 핵심:
# 환각은 가장 긴 관측이므로 max를 쓰면 한계가 환각에 끌려가 영영 안 걸린다.
# 투영 뼈길이 분포는 [0, 실제길이]이고 그 상단 분위수가 곧 실제길이 근사다.
_PCTL = 80
_SLACK = 1.2
_WIN = 64  # 분위수 추정 윈도우(프레임)
_MIN_OBS = 8  # 이만큼 모이기 전엔 클램프 안 함(통계 신뢰 전)


def _bbox_diag(bbox):
    if bbox is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    d = float(np.hypot(max(1e-4, x1 - x0), max(1e-4, y1 - y0)))
    return d if d > 1e-4 else None


def update_and_clamp(kp, bbox, bone_hist, conf_thresh=0.3):
    """
    kp: (>=17,3) 정규화 좌표. bbox: (4,) 정규화 xyxy or None.
    bone_hist: dict[(a,b)] -> deque of 정규화 길이(뼈길이/bbox대각). in-place 갱신.
    반환: (kp_clamped, n_clamped)
    """
    diag = _bbox_diag(bbox)
    if diag is None:
        return kp, 0
    out = kp.copy()
    n_clamped = 0
    for (a, b) in BONES:
        if out[a, 2] < conf_thresh or out[b, 2] < conf_thresh:
            continue
        L = float(np.hypot(*(out[a, :2] - out[b, :2])))
        r = L / diag
        dq = bone_hist.get((a, b))
        if dq is None:
            dq = bone_hist[(a, b)] = deque(maxlen=_WIN)
        if len(dq) < _MIN_OBS:
            dq.append(r)
            continue
        limit = float(np.percentile(dq, _PCTL)) * _SLACK * diag
        dq.append(r)  # 이번 관측도 기록(클램프 판정 후)
        # distal 관절만 끌어당긴다. a,b 중 어느 쪽이 distal인지 판정.
        if L > limit and limit > 1e-6:
            if b in _DISTAL and _PARENT.get(b) == a:
                anchor, mover = a, b
            elif a in _DISTAL and _PARENT.get(a) == b:
                anchor, mover = b, a
            else:
                continue
            d = out[mover, :2] - out[anchor, :2]
            nrm = float(np.hypot(*d))
            if nrm > 1e-6:
                out[mover, :2] = out[anchor, :2] + d / nrm * limit
                out[mover, 2] *= 0.6  # 보정됨 → 신뢰도 감점
                n_clamped += 1
    return out, n_clamped
