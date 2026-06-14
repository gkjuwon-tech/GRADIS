"""
스켈레톤 렌더링 + 클립 인코딩.

- draw_skeleton: 프레임 위에 COCO-17 스켈레톤(+추적 박스)을 그린다. 시그니처 옐로우.
- encode_gif: rescue된 프레임들을 애니메이션 GIF(bytes)로. (추가 의존성 0 — PIL만)
  실배포에선 H.264 mp4로 바꾸면 됨(imageio/ffmpeg). GIF는 '진짜 돌아감'의 최소 증명.
"""
import io
from PIL import Image, ImageDraw

YELLOW = (220, 192, 74)
WHITE = (235, 235, 235)
DIM = (90, 90, 95)

# COCO-17 골격 연결
EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),      # 팔
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리
    (5, 6), (11, 12), (5, 11), (6, 12),   # 몸통
    (0, 5), (0, 6),                        # 목
]

# GRADIS-25 파생 골격 (kp가 25관절일 때만): 척추 라인 + 손/발 말단
EDGES_EXT = [
    (20, 17), (17, 19), (19, 18),          # head→neck→spine→pelvis
    (9, 21), (10, 22),                     # 손목→손
    (15, 23), (16, 24),                    # 발목→발
]


def draw_skeleton(img, persons, boxed_ids=None):
    """img: PIL.Image (in-place로 안 그리고 복사본 반환).
    persons: list of (id, kp[17,3] 또는 kp[25,3] GRADIS-25)."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    W, H = out.size
    boxed_ids = boxed_ids or set()

    for pid, kp in persons:
        pts = [(x * W, y * H, c) for (x, y, c) in kp]
        edges = EDGES + (EDGES_EXT if len(pts) >= 25 else [])
        for a, b in edges:
            if pts[a][2] > 0.2 and pts[b][2] > 0.2:
                d.line([pts[a][:2], pts[b][:2]], fill=YELLOW, width=2)
        for (x, y, c) in pts:
            if c > 0.2:
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=WHITE)
        # 추적 박스 (위험 대상)
        xs = [p[0] for p in pts if p[2] > 0.2]
        ys = [p[1] for p in pts if p[2] > 0.2]
        if xs and ys:
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            pad = 10
            col = YELLOW if pid in boxed_ids else DIM
            _corners(d, x0 - pad, y0 - pad, x1 + pad, y1 + pad, col)
    return out


def _corners(d, x0, y0, x1, y1, col, n=14):
    for (cx, cy, sx, sy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        d.line([(cx, cy), (cx + sx * n, cy)], fill=col, width=2)
        d.line([(cx, cy), (cx, cy + sy * n)], fill=col, width=2)


def blank_frame(w=640, h=360, bg=(14, 14, 16)):
    return Image.new("RGB", (w, h), bg)


def encode_gif(frames, fps=10):
    """PIL.Image 리스트 -> GIF bytes. 비면 None."""
    if not frames:
        return None
    frames = [f.convert("RGB") for f in frames]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return buf.getvalue()
