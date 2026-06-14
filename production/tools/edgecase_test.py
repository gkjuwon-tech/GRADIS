#!/usr/bin/env python3
"""
극한 엣지케이스 추출 검증 (엄밀판) — 멀거나/어둡거나/저화질에서도 잡히는가.

이전 결함 수정:
  - 연속 프레임으로 처리(트래커가 확정되도록). 불연속 샘플링 금지.
  - raw 백엔드 검출이 아니라 'SkeletonEngine 최종 출력'(해부학게이트+추적 후)을 측정.
  - 열화를 충분히 극한으로(baseline이 실제로 실패하도록).

비교:
  baseline : 보정 OFF, 단일스케일
  robust   : 저조도보정(CLAHE+감마) ON + 멀티스케일 타일

  python tools/edgecase_test.py production/_media/wide_45881.mp4 --degrade dark
"""
import sys, os, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.pose import SkeletonEngine, make_backend
from edge.pose.enhance import enhance


def degrade(frame, mode):
    H, W = frame.shape[:2]
    if mode == "dark":      # 야간 모사: 매우 어둡게 + 약한 노이즈
        d = np.clip(frame.astype(np.float32) * 0.10, 0, 255)
        d = np.clip(d + np.random.normal(0, 4, d.shape), 0, 255)
        return d.astype(np.uint8)
    if mode == "far":       # 원거리: 18%로 축소해 중앙 배치(사람이 아주 작아짐)
        s = 0.18
        small = cv2.resize(frame, (int(W*s), int(H*s)))
        canvas = np.zeros_like(frame)
        y0 = (H-small.shape[0])//2; x0 = (W-small.shape[1])//2
        canvas[y0:y0+small.shape[0], x0:x0+small.shape[1]] = small
        return canvas
    if mode == "lowq":      # 심한 저화질: 1/6 해상도 + 강한 JPEG
        small = cv2.resize(frame, (W//6, H//6))
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        ok, enc = cv2.imencode(".jpg", up, [cv2.IMWRITE_JPEG_QUALITY, 12])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return frame


def run(video, mode, model, pose_preset, pose_config, pose_checkpoint, imgsz,
        conf, seg_len, start_frac):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    start = int(total * start_frac)

    base_be = make_backend("topview", detector_model=model, pose_preset=pose_preset,
                           pose_config=pose_config, pose_checkpoint=pose_checkpoint,
                           imgsz=imgsz, conf=conf, tiles=1, half=False)
    rob_be = make_backend("topview", detector_model=model, pose_preset=pose_preset,
                          pose_config=pose_config, pose_checkpoint=pose_checkpoint,
                          imgsz=imgsz, conf=conf, tiles=2, half=False)
    base_eng = SkeletonEngine(aerial=True, default_fps=fps)
    rob_eng = SkeletonEngine(aerial=True, default_fps=fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    bp, bc, rp, rc = [], [], [], []
    sample_saved = False
    for k in range(seg_len):
        ok, frame = cap.read()
        if not ok:
            break
        deg = degrade(frame, mode)
        t = k / fps
        robust_frame = enhance(deg, force=(mode == "dark"))
        bpeople = base_eng.process(base_be(deg), t)   # 엔진 최종 출력
        rpeople = rob_eng.process(rob_be(robust_frame), t)
        if k >= seg_len * 0.4:                          # 트래커 확정 후 구간만 집계
            bp.append(len(bpeople)); rp.append(len(rpeople))
            bc += [p.quality for p in bpeople]
            rc += [p.quality for p in rpeople]
        if not sample_saved and k == int(seg_len*0.7):
            _save_sample(deg, bpeople, rpeople, mode, video)
            sample_saved = True
    cap.release()

    def mm(a): return float(np.mean(a)) if a else 0.0
    print("=" * 60)
    print(f" EDGE-CASE [{mode.upper()}] — {os.path.basename(video)}  ({seg_len} 연속프레임)")
    print(f" (실제 인원 2명. 엔진 최종 출력 = 해부학게이트+추적 후)")
    print("-" * 60)
    print(f" baseline : {mm(bp):.2f} persons/frame, quality {mm(bc):.3f}")
    print(f" robust   : {mm(rp):.2f} persons/frame, quality {mm(rc):.3f}")
    if mm(bp) > 0:
        print(f"  -> 검출 {(mm(rp)/mm(bp)-1)*100:+.0f}%")
    else:
        print(f"  -> baseline 0명(완전실패) → robust {mm(rp):.2f}명 회수")
    print("=" * 60)


def _save_sample(deg, b, r, mode, video):
    from edge.render import EDGES
    def draw(img, ppl, col):
        o = img.copy(); H, W = o.shape[:2]
        for p in ppl:
            kp = p.kp; pts = [(int(x*W), int(y*H), c) for x, y, c in kp]
            for a, bb in EDGES:
                if kp[a, 2] > 0.2 and kp[bb, 2] > 0.2:
                    cv2.line(o, pts[a][:2], pts[bb][:2], col, 2, cv2.LINE_AA)
        return o
    left = draw(deg, b, (60, 60, 255)); cv2.putText(left, f"baseline n={len(b)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(60,60,255),2)
    right = draw(deg, r, (60,220,60)); cv2.putText(right, f"robust n={len(r)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(60,220,60),2)
    out = os.path.join(os.path.dirname(video), f"edge_{mode}.png")
    cv2.imwrite(out, np.hstack([left, right]))
    print(f" sample -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--degrade", default="dark", choices=["dark", "far", "lowq"])
    ap.add_argument("--model", "--detector-model", dest="detector_model",
                    default="yolo11x.pt")
    ap.add_argument("--pose-preset", default="rtmpose-m",
                    choices=["rtmpose-m", "vitpose-s"])
    ap.add_argument("--pose-config", default=None)
    ap.add_argument("--pose-checkpoint", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seg", type=int, default=60)
    ap.add_argument("--start", type=float, default=0.3)
    args = ap.parse_args()
    run(args.video, args.degrade, args.detector_model, args.pose_preset,
        args.pose_config, args.pose_checkpoint, args.imgsz, args.conf,
        args.seg, args.start)


if __name__ == "__main__":
    main()
