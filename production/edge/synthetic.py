"""
합성 스켈레톤 생성기 — 카메라/드론 없이 파이프라인 전체를 '진짜로' 돌리기 위한 소스.

여기서 만드는 건 가짜 알림이 아니다. 진짜 COCO-17 좌표 시퀀스를 생성해서
redact 렌더링을 거쳐 진짜 VLM(Cognition)에 흘려보낸다. VLM이 실제로
낙상/몸싸움을 판단해야만 알림이 뜬다. 즉 '판단은 진짜, 입력만 합성'.

실드론에선 이 파일 대신 pose.PoseEstimator(웹캠/RTSP)가 좌표를 공급한다.
타임라인(약 11초):
   0.0–3.0s  보행자 A 한 명 정상 보행        (아무 일 없음 → 영상 폐기되어야 함)
   3.0–3.6s  A 낙상                          (FALL 떠야 함)
   3.6–6.0s  A 바닥에 누워 미동 없음          (MEDICAL 떠도 정상)
   6.0–10.5s 보행자 B, C 등장 → 몸싸움        (FIGHT 떠야 함)
"""
import numpy as np
from .render import blank_frame, draw_skeleton

# 서있는 자세의 로컬 템플릿 (엉덩이중심 원점, 위=-y). 대략 키 0.62.
_TEMPLATE = np.array([
    [0.00, -0.34],  # 0 nose
    [-.02, -0.36], [.02, -0.36], [-.04, -0.35], [.04, -0.35],  # eyes/ears
    [-.07, -0.24], [.07, -0.24],   # shoulders 5,6
    [-.10, -0.12], [.10, -0.12],   # elbows 7,8
    [-.11, -0.01], [.11, -0.01],   # wrists 9,10
    [-.05, 0.00], [.05, 0.00],     # hips 11,12
    [-.05, 0.16], [.05, 0.16],     # knees 13,14
    [-.05, 0.32], [.05, 0.32],     # ankles 15,16
], np.float32)


def _person(cx, cy, scale=0.9, tilt_deg=0.0, jitter=None):
    """템플릿을 기울이고(tilt) 옮겨서(cx,cy) 정규화 좌표 kp[17,3] 반환."""
    th = np.radians(tilt_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    pts = (_TEMPLATE * scale) @ R.T
    if jitter is not None:
        pts = pts + jitter
    kp = np.ones((17, 3), np.float32)
    kp[:, 0] = pts[:, 0] + cx
    kp[:, 1] = pts[:, 1] + cy
    return kp


def frames(fps=12):
    """제너레이터: (rel_t, frame_PIL, persons) 를 시간순으로 내보낸다."""
    dt = 1.0 / fps
    T = 10.5
    n = int(T * fps)
    for i in range(n):
        t = i * dt
        persons = []   # (id, kp)

        # --- 보행자 A: 걷다가 3.0s에 낙상 ---
        if t < 6.0:
            ax = 0.30 + 0.03 * t                     # 천천히 이동
            if t < 3.0:                              # 정상 보행
                swing = 0.015 * np.sin(t * 6)
                j = np.zeros((17, 2), np.float32); j[15, 0] += swing; j[16, 0] -= swing
                persons.append((1, _person(ax, 0.50, tilt_deg=2 * np.sin(t * 3), jitter=j)))
            else:                                    # 낙상 (0.6초간 수직→수평 + 하강)
                p = min(1.0, (t - 3.0) / 0.6)
                tilt = 90 * p
                cy = 0.50 + 0.26 * p                 # 무게중심 급강하
                persons.append((1, _person(ax, cy, tilt_deg=tilt)))

        # --- 보행자 B, C: 6.0s부터 몸싸움 ---
        # 현실적인 빠른(2Hz) 사지 진동 + 몸통 흔들림. 텔레포트 아님(부드러움).
        # 사람 팔은 빠르되 연속적이다 → 평활 후에도 충분히 빠른 속도가 남아야 한다.
        if t >= 6.0:
            f = t - 6.0
            w = 2.0 * np.pi * 2.0     # 2 Hz
            ampA, ampL = 0.07, 0.05
            jB = np.zeros((17, 2), np.float32)
            jC = np.zeros((17, 2), np.float32)
            for k, idx in enumerate((9, 10, 7, 8)):   # 손목/팔꿈치
                ph = k * 1.1
                jB[idx, 0] += ampA * np.sin(w * f + ph)
                jB[idx, 1] += ampL * np.cos(w * f + ph)
                jC[idx, 0] += ampA * np.sin(w * f + ph + np.pi)   # 반대 위상
                jC[idx, 1] += ampL * np.cos(w * f + ph + np.pi)
            swayB = 0.02 * np.sin(w * 0.5 * f)
            swayC = 0.02 * np.sin(w * 0.5 * f + np.pi)
            persons.append((2, _person(0.47 + swayB, 0.50, jitter=jB)))
            persons.append((3, _person(0.55 + swayC, 0.50, jitter=jC)))

        # 원본 '프레임'(렌더된 장면) — 실드론이면 진짜 카메라 픽셀
        frame = draw_skeleton(blank_frame(), persons)
        yield t, frame, persons
