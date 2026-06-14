"""
폐색 복원 (Occlusion Recovery).

탑다운 드론 시야에선 사람이 사람을, 나무가 다리를 가린다. pose 모델은 가려진
관절의 신뢰도를 떨어뜨리거나 엉뚱한 데 찍는다. 이게 그대로 VLM에 가면:
  - 낙상 중 다리가 사라진 스켈레톤 → VLM이 자세를 오해 (= 사람 안 구함)
  - 손목이 튀어 → 가짜 난투처럼 보임 → 헛출동

복원 전략(보이는 것으로 안 보이는 것 추정):
  1) 강체 이동: 직전 프레임의 관절을 '사람 중심 이동량'만큼 평행이동
  2) 골격 길이 제약: 부모 관절이 보이면, 자식을 부모에서 EMA 뼈길이만큼 떨어뜨림
  3) 좌우 대칭 보강: 한쪽이 보이면 반대쪽 길이로 보정
복원된 관절은 confidence를 낮춰 표시(inferred) → 다운스트림이 가중치 조절 가능.
"""
import numpy as np
from .skeleton import BONES, N_KPT

# 자식 -> 부모 (복원 시 부모 기준으로 자식 배치)
_PARENT = {}
for _p, _c in BONES:
    _PARENT.setdefault(_c, _p)


def recover(kp, prev_kp, center_shift, bone_len, conf_thresh=0.35):
    """
    kp: (17,3) 현재 관측(x,y,conf). prev_kp: 직전 평활 (17,3) or None.
    center_shift: (2,) 사람 중심 이동량. bone_len: dict[(a,b)]->길이(px,정규화).
    반환: (kp_filled[17,3], inferred[17] bool)
    """
    out = kp.copy()
    inferred = np.zeros(N_KPT, bool)
    if prev_kp is None:
        return out, inferred

    for j in range(N_KPT):
        if kp[j, 2] >= conf_thresh:
            continue  # 잘 보임 — 복원 불필요
        cand = None

        # 1) 강체 이동 예측 (직전 위치 + 중심 이동)
        if prev_kp[j, 2] > 0.1 or np.any(prev_kp[j, :2]):
            cand = prev_kp[j, :2] + center_shift

        # 2) 골격 길이 제약 (부모가 현재 보이면 더 정확)
        par = _PARENT.get(j)
        if par is not None and out[par, 2] >= conf_thresh:
            key = (par, j) if (par, j) in bone_len else (j, par)
            L = bone_len.get(key)
            if L is not None and cand is not None:
                d = cand - out[par, :2]
                n = np.hypot(*d)
                if n > 1e-5:
                    cand = out[par, :2] + d / n * L   # 방향 유지, 길이 보정
            elif L is not None and prev_kp[par, 2] > 0.1:
                # 직전 방향을 재사용
                d = prev_kp[j, :2] - prev_kp[par, :2]
                n = np.hypot(*d)
                if n > 1e-5:
                    cand = out[par, :2] + d / n * L

        if cand is not None:
            out[j, :2] = cand
            out[j, 2] = max(out[j, 2], 0.25)  # 복원됨: 낮은 신뢰도로 표시
            inferred[j] = True

    return out, inferred


def update_bone_lengths(bone_len, kp, alpha=0.2, conf_thresh=0.5):
    """양 끝이 잘 보이는 뼈의 길이를 EMA로 학습(사람마다 체형이 다르니까)."""
    for (a, b) in BONES:
        if kp[a, 2] >= conf_thresh and kp[b, 2] >= conf_thresh:
            L = float(np.hypot(*(kp[a, :2] - kp[b, :2])))
            if 1e-4 < L < 1.0:
                prev = bone_len.get((a, b))
                bone_len[(a, b)] = L if prev is None else (1 - alpha) * prev + alpha * L
    return bone_len
