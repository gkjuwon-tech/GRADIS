"""
Privacy Compositor — VLM에게 '환경은 보여주되 사람은 신원 제거' 화면을 만든다.

마스킹 방식: **강한 블러**(검정 실루엣 X).
  - 검정 실루엣은 사람 형태/맥락(자세 윤곽, 들고 있는 물건)까지 지워 VLM 판단을 방해한다.
  - 강한 블러는 얼굴/신원은 완전히 파괴하면서 '사람이 거기 있고 대략 이런 덩어리'라는
    맥락은 남겨, VLM이 환경과 함께 상황을 더 잘 이해한다.
  - 블러 강도는 사람 크기에 비례(가까울수록 더 세게) → 거리와 무관하게 얼굴 식별 불가.
그 위에 스켈레톤(노란 막대기)을 그려 자세 정보를 명시적으로 전달한다.

원본 프레임은 이 함수에 안 들어간 채 따로 RAM 링버퍼에만 있다가 위험 확정 시에만
경찰에게 간다. 어떤 '모델'도 생얼굴을 학습/추론에 쓰지 않는다.
"""
import numpy as np
import cv2
from PIL import Image
from .render import EDGES, EDGES_EXT

SKELETON = (70, 210, 250)     # BGR 옐로우(막대기)
JOINT = (240, 240, 240)
HEAD = 0


def _bbox_px(kp, W, H):
    v = kp[kp[:, 2] > 0.2]
    if len(v) == 0:
        return None
    xs, ys = v[:, 0] * W, v[:, 1] * H
    return xs.min(), ys.min(), xs.max(), ys.max()


def _odd(n):
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1


def redact_people(frame, persons, draw_skeleton=True, blur_strength=0.9):
    """
    frame: PIL.Image (원본). persons: list[(id, kp[17,3])].
    blur_strength: 0..1, 클수록 더 강하게 블러.
    반환: 사람이 강블러+스켈레톤으로 치환된 PIL.Image (VLM 입력용).
    """
    rgb = np.asarray(frame.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]

    mask = np.zeros((H, W), np.uint8)
    max_ph = 0.0
    pts_all = []

    for pid, kp in persons:
        bb = _bbox_px(kp, W, H)
        if bb is None:
            pts_all.append(None)
            continue
        x0, y0, x1, y1 = bb
        ph = max(8.0, y1 - y0)
        max_ph = max(max_ph, ph)
        limb_w = max(8, int(ph * 0.18))
        pts = [(int(x * W), int(y * H), c) for (x, y, c) in kp]
        pts_all.append(pts)
        # 사람 영역 마스크: 골격 캡슐 + 머리 원 + 몸통 (GRADIS-25면 손/발/척추까지)
        edges = EDGES + (EDGES_EXT if len(pts) >= 25 else [])
        for a, b in edges:
            if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
                cv2.line(mask, pts[a][:2], pts[b][:2], 255, limb_w, cv2.LINE_AA)
        if kp[HEAD, 2] > 0.2:
            cv2.circle(mask, pts[HEAD][:2], int(limb_w * 1.1), 255, -1)
        torso = [5, 6, 12, 11]
        tp = np.array([pts[i][:2] for i in torso if kp[i, 2] > 0.2], np.int32)
        if len(tp) >= 3:
            cv2.fillPoly(mask, [tp], 255)

    if max_ph > 0:
        # 마스크 확장(여유) + 강한 블러
        dil = _odd(max_ph * 0.10)
        mask = cv2.dilate(mask, np.ones((dil, dil), np.uint8))
        mask = cv2.GaussianBlur(mask, (_odd(dil), _odd(dil)), 0)  # 가장자리 페더링

        k = _odd(max_ph * (0.35 + 0.5 * blur_strength))           # 사람 크기 비례 강블러
        blurred = cv2.GaussianBlur(bgr, (k, k), 0)
        blurred = cv2.GaussianBlur(blurred, (_odd(k * 0.6), _odd(k * 0.6)), 0)  # 2패스 = 더 강함
        # 추가로 픽셀화(모자이크)로 신원 완전 파괴
        small = cv2.resize(bgr, (max(1, W // 24), max(1, H // 24)),
                           interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        blurred = cv2.addWeighted(blurred, 0.55, pix, 0.45, 0)

        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        bgr = (blurred * alpha + bgr * (1 - alpha)).astype(np.uint8)

    # 스켈레톤 오버레이
    if draw_skeleton:
        for pts in pts_all:
            if pts is None:
                continue
            edges = EDGES + (EDGES_EXT if len(pts) >= 25 else [])
            for a, b in edges:
                if pts[a][2] > 0.2 and pts[b][2] > 0.2:
                    cv2.line(bgr, pts[a][:2], pts[b][:2], SKELETON, 3, cv2.LINE_AA)
            for (x, y, c) in pts:
                if c > 0.2:
                    cv2.circle(bgr, (x, y), 3, JOINT, -1, cv2.LINE_AA)

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)
