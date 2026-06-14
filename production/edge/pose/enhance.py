"""
입력 영상 보정 — 극한 조건(저조도/저대비/저화질)에서 pose 검출률을 끌어올린다.

드론 치안 현장은 야간·역광·안개·먼 거리가 기본이다. 원본을 그대로 모델에 넣으면
어두운 곳의 사람을 놓친다 = 사람을 못 구한다. 그래서 추론 전에:
  1) 저조도면 감마 보정으로 밝히고
  2) CLAHE(국소 히스토그램 평활)로 그림자 속 디테일을 살리고
  3) 약한 디노이즈로 저화질 노이즈가 가짜 키포인트를 만들지 않게 한다.
모두 추론 '입력'에만 적용하고, 경찰에게 가는 원본 클립은 손대지 않는다.
"""
import numpy as np
import cv2


def mean_luma(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def enhance(bgr, auto=True, clahe_clip=2.5, denoise=False, force=False):
    """
    bgr: np.uint8[H,W,3]. 반환: 보정된 bgr.
    auto=True면 밝기에 따라 적응적으로 적용(밝은 영상은 거의 그대로).
    """
    luma = mean_luma(bgr)
    out = bgr

    # 1) 저조도 감마 부스트 (어두울수록 강하게)
    if force or (auto and luma < 110):
        # luma 40 -> gamma~0.5, luma 110 -> gamma~0.85
        gamma = float(np.clip(0.5 + (luma / 110.0) * 0.4, 0.45, 0.95))
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)

    # 2) CLAHE — L 채널 국소 대비 향상 (그림자 디테일)
    if force or (auto and luma < 140):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3) 저화질 노이즈 억제 (옵션 — 비용 큼)
    if denoise:
        out = cv2.bilateralFilter(out, 5, 40, 40)

    return out
