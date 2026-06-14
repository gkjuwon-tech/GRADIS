"""
GRADIS-25 골격 토폴로지 + 해부학 제약 + 시점 불변 정규화.

여긴 '사람의 몸은 이렇게 생겼다'는 사전지식을 코드로 박는 곳이다.
모델이 가끔 뱉는 해부학적으로 불가능한 포즈(무릎이 어깨 위 등)를 걸러내고,
폐색 복원에 쓸 뼈 연결관계/대칭쌍을 정의한다.

GRADIS-25 = COCO-17(검출) + 파생 8관절(기하 유도):
  17 neck      어깨 중점 — 몸통각/목 분석의 안정 기준점
  18 pelvis    엉덩이 중점 — 무게중심 추적의 정식 원점
  19 spine     neck↔pelvis 중점 — 몸통 굽힘(웅크림/구부림) 감지
  20 head      코+눈+귀 가중평균 — nose 단독보다 훨씬 안정적인 머리 위치
  21 l_hand    손목을 팔꿈치→손목 방향으로 연장 — 손에 든 것/타격점 추정
  22 r_hand
  23 l_foot    발목을 무릎→발목 방향으로 연장 — 지면 접점(낙상/난간 분석)
  24 r_foot
파생 관절은 검출 관절에서 결정론적으로 계산되며 신뢰도는 부모 관절의 최소값을
따른다. 다운스트림(렌더/redact/ML)은 인덱스 0..16이 기존 COCO-17과 완전 호환.
"""
import numpy as np

# COCO-17 인덱스
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16
N_KPT = 17

# GRADIS-25 파생 인덱스 (17..24)
NECK, PELVIS, SPINE, HEAD = 17, 18, 19, 20
LHAND, RHAND, LFOOT, RFOOT = 21, 22, 23, 24
N_KPT_EXT = 25

KPT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho",
             "l_elb", "r_elb", "l_wri", "r_wri", "l_hip", "r_hip",
             "l_kne", "r_kne", "l_ank", "r_ank",
             "neck", "pelvis", "spine", "head",
             "l_hand", "r_hand", "l_foot", "r_foot"]

# 골격 (부모, 자식) — 폐색 복원 시 부모로부터 자식을 추정한다.
BONES = [
    (LSHO, LELB), (LELB, LWRI),      # 왼팔
    (RSHO, RELB), (RELB, RWRI),      # 오른팔
    (LHIP, LKNE), (LKNE, LANK),      # 왼다리
    (RHIP, RKNE), (RKNE, RANK),      # 오른다리
    (LSHO, RSHO), (LHIP, RHIP),      # 어깨/엉덩이 폭
    (LSHO, LHIP), (RSHO, RHIP),      # 몸통 양옆
    (NOSE, LSHO), (NOSE, RSHO),      # 목
]

# 좌우 대칭쌍 — 한쪽이 가려지면 반대쪽 길이로 보강
SYMMETRY = [(LSHO, RSHO), (LELB, RELB), (LWRI, RWRI),
            (LHIP, RHIP), (LKNE, RKNE), (LANK, RANK)]

# 항공(탑다운) 시야에서 얼굴 키포인트는 작고 불안정 → 신뢰도 가중치 하향.
# 몸통/사지를 신뢰한다. 이게 '드론 전용' 튜닝.
AERIAL_KPT_WEIGHT = np.array([
    0.6, 0.3, 0.3, 0.3, 0.3,   # nose, eyes, ears (낮게)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 어깨/팔꿈치/손목
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 엉덩이/무릎/발목
], np.float32)

# 사람 키 대비 뼈 길이 비율(정규화). 해부학 검증/복원의 기준값.
# (성인 평균 비율, 사람 키=1.0 기준 근사)
BONE_RATIO = {
    (LSHO, LELB): 0.16, (LELB, LWRI): 0.14,
    (RSHO, RELB): 0.16, (RELB, RWRI): 0.14,
    (LHIP, LKNE): 0.24, (LKNE, LANK): 0.23,
    (RHIP, RKNE): 0.24, (RKNE, RANK): 0.23,
    (LSHO, RSHO): 0.20, (LHIP, RHIP): 0.12,
    (LSHO, LHIP): 0.30, (RSHO, RHIP): 0.30,
    (NOSE, LSHO): 0.18, (NOSE, RSHO): 0.18,
}


def torso_size(kp):
    """어깨중심–엉덩이중심 거리 = 사람 스케일 기준자(시점/거리 정규화용)."""
    sh = _mid(kp, LSHO, RSHO)
    hp = _mid(kp, LHIP, RHIP)
    if sh is None or hp is None:
        # 폴백: 보이는 키포인트 바운딩 대각선
        v = kp[kp[:, 2] > 0.2]
        if len(v) < 2:
            return None
        return float(np.hypot(np.ptp(v[:, 0]), np.ptp(v[:, 1]))) * 0.5
    return float(np.hypot(*(sh - hp)))


def _mid(kp, a, b):
    if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
        return (kp[a, :2] + kp[b, :2]) / 2
    if kp[a, 2] > 0.2:
        return kp[a, :2].copy()
    if kp[b, 2] > 0.2:
        return kp[b, :2].copy()
    return None


def anatomical_validity(kp):
    """
    해부학적으로 말이 되는 포즈인지 0..1 점수. 낮으면 쓰레기 검출(걸러냄).
    - 뼈 길이가 사람 스케일 대비 비정상(너무 길거나 0)인 비율
    - 신뢰 키포인트 수
    """
    scale = torso_size(kp)
    if not scale or scale < 1e-4:
        return 0.0
    ok, bad = 0, 0
    for (a, b), ratio in BONE_RATIO.items():
        if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
            L = np.hypot(*(kp[a, :2] - kp[b, :2])) / scale
            exp = ratio / 0.30  # 몸통(0.30)을 1.0 스케일로 환산
            if 0.4 * exp <= L <= 2.2 * exp:
                ok += 1
            else:
                bad += 1
    visible = int((kp[:, 2] > 0.3).sum())
    if ok + bad == 0:
        return 0.2 * min(1.0, visible / 6)
    bone_score = ok / (ok + bad)
    vis_score = min(1.0, visible / 10)
    return float(0.65 * bone_score + 0.35 * vis_score)


def normalize_pose(kp):
    """
    시점/거리/위치 불변 표현. 엉덩이중심 원점, 몸통크기로 스케일.
    다운스트림 ML 모델(ST-GCN 등) 입력용. (검출/감지는 이미지좌표를 쓰고,
    이건 '학습용 좌표 텐서' — 얼굴 없음, 숫자만.)
    반환: np[K,2] 정규화 좌표 (보이지 않는 곳은 0). K는 입력 관절 수(17 또는 25).
    """
    k = kp.shape[0]
    out = np.zeros((k, 2), np.float32)
    hip = _mid(kp, LHIP, RHIP)
    scale = torso_size(kp)
    if hip is None or not scale or scale < 1e-4:
        return out
    vis = kp[:, 2] > 0.2
    out[vis] = (kp[vis, :2] - hip) / scale
    return out


# ----------------------------------------------------------------------------
# GRADIS-25 확장 — 검출 17관절에서 파생 8관절을 기하적으로 유도
# ----------------------------------------------------------------------------

# 사지 말단 연장 비율: hand = wrist + r*(wrist-elbow), foot = ankle + r*(ankle-knee)
_HAND_EXT = 0.35   # 손바닥 중심 근사 (전완 길이 대비)
_FOOT_EXT = 0.30   # 발끝/지면 접점 근사 (정강이 길이 대비)


def _derive_mid(kp, out, dst, a, b):
    if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
        out[dst, :2] = (kp[a, :2] + kp[b, :2]) / 2
        out[dst, 2] = min(kp[a, 2], kp[b, 2])


def _derive_ext(kp, out, dst, near, far, ratio):
    """far(말단)를 near→far 방향으로 ratio만큼 연장. 예: 팔꿈치→손목→손."""
    if kp[near, 2] > 0.2 and kp[far, 2] > 0.2:
        d = kp[far, :2] - kp[near, :2]
        if float(np.hypot(*d)) > 1e-5:
            out[dst, :2] = kp[far, :2] + d * ratio
            out[dst, 2] = min(kp[near, 2], kp[far, 2]) * 0.9  # 파생 = 약간 감점


def extend_keypoints(kp17):
    """
    kp17: np[17,3] → np[25,3] GRADIS-25. 0..16은 그대로 복사(하위호환).
    파생 관절은 부모가 안 보이면 conf=0으로 남는다.
    """
    out = np.zeros((N_KPT_EXT, 3), np.float32)
    out[:N_KPT] = kp17

    _derive_mid(kp17, out, NECK, LSHO, RSHO)
    _derive_mid(kp17, out, PELVIS, LHIP, RHIP)
    if out[NECK, 2] > 0.2 and out[PELVIS, 2] > 0.2:
        out[SPINE, :2] = (out[NECK, :2] + out[PELVIS, :2]) / 2
        out[SPINE, 2] = min(out[NECK, 2], out[PELVIS, 2])

    # head: 보이는 머리 키포인트(코/눈/귀)의 신뢰도 가중 평균
    head_idx = [NOSE, LEYE, REYE, LEAR, REAR]
    w = kp17[head_idx, 2].copy()
    vis = w > 0.2
    if vis.any():
        pts = kp17[head_idx][vis]
        ww = w[vis] / w[vis].sum()
        out[HEAD, :2] = (pts[:, :2] * ww[:, None]).sum(0)
        out[HEAD, 2] = float(w[vis].max())

    _derive_ext(kp17, out, LHAND, LELB, LWRI, _HAND_EXT)
    _derive_ext(kp17, out, RHAND, RELB, RWRI, _HAND_EXT)
    _derive_ext(kp17, out, LFOOT, LKNE, LANK, _FOOT_EXT)
    _derive_ext(kp17, out, RFOOT, RKNE, RANK, _FOOT_EXT)
    return out
