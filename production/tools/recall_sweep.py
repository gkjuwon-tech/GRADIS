#!/usr/bin/env python3
"""
Detector recall SWEEP — find an operating point that actually sees small people.

recall_compare.py answers "tiles vs no tiles at one setting". This answers the
prior question for top-view/drone footage: *at what conf / imgsz / tile count
does YOLO start detecting these people at all, and how tall are they in pixels?*
If even the most aggressive setting yields ~0, that is the evidence that a
COCO-trained detector is the wrong tool and an aerial-finetuned one is needed —
not a reason to pick an easier video.

Run on GPU (Colab):
  python tools/recall_sweep.py _media/real_world_hard/caviar_walk_by_shop_front.mpg
  python tools/recall_sweep.py <video> --model yolo11x.pt --n 12 \
      --conf 0.03 0.05 0.10 0.25 --imgsz 1280 1920 --tiles 1 2 3

Outputs a table (persons/frame + median person height in px) and, for the single
best setting, an overlay PNG next to the video so you can SEE the boxes.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tile_offsets(w, h, n, overlap):
    tw, th = w / n, h / n
    for r in range(n):
        for c in range(n):
            x0 = int(max(0, c * tw - overlap * tw))
            x1 = int(min(w, (c + 1) * tw + overlap * tw))
            y0 = int(max(0, r * th - overlap * th))
            y1 = int(min(h, (r + 1) * th + overlap * th))
            if x1 > x0 and y1 > y0:
                yield x0, y0, x1, y1


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    d = aa + bb - inter
    return inter / d if d > 0 else 0.0


def _nms(boxes, thr=0.55):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) <= thr for k in kept):
            kept.append(b)
    return kept


def detect(model, frame, conf, imgsz, tiles, augment):
    """Return list of [x0,y0,x1,y1,score] person boxes, with optional tiling."""
    h, w = frame.shape[:2]
    out = []
    r = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], iou=0.5,
                      augment=augment, verbose=False, half=False)[0]
    if r.boxes is not None and r.boxes.xyxy is not None:
        xy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        out.extend([[*b, float(c)] for b, c in zip(xy, cf)])
    if tiles > 1:
        crops, offs = [], []
        for x0, y0, x1, y1 in _tile_offsets(w, h, tiles, 0.2):
            crops.append(frame[y0:y1, x0:x1])
            offs.append((x0, y0))
        for res, (ox, oy) in zip(
            model.predict(crops, imgsz=imgsz, conf=conf, classes=[0], iou=0.5,
                          augment=augment, verbose=False, half=False), offs):
            if res.boxes is None or res.boxes.xyxy is None:
                continue
            xy = res.boxes.xyxy.cpu().numpy()
            cf = res.boxes.conf.cpu().numpy()
            for b, c in zip(xy, cf):
                out.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, float(c)])
    return _nms(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", "--detector-model", dest="model", default="yolo11x.pt")
    ap.add_argument("--n", type=int, default=12, help="frames sampled across the clip")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.03, 0.05, 0.10, 0.25])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1920])
    ap.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--augment", action="store_true", help="test-time augmentation (slower)")
    args = ap.parse_args()

    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.linspace(total * 0.15, total * 0.85, args.n).astype(int)
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            frames.append((int(fi), f))
    cap.release()

    model = YOLO(args.model)
    print("=" * 72)
    print(f" RECALL SWEEP — {os.path.basename(args.video)}  {W}x{H}  "
          f"{len(frames)} frames  model={os.path.basename(args.model)}"
          f"{'  +TTA' if args.augment else ''}")
    print("-" * 72)
    print(f" {'conf':>5} {'imgsz':>6} {'tiles':>6} | {'persons/frame':>14} {'med_h(px)':>10} {'p90_h':>7}")
    print("-" * 72)

    best = None  # (persons_per_frame, conf, imgsz, tiles)
    for imgsz in args.imgsz:
        for tiles in args.tiles:
            for conf in args.conf:
                counts, heights = [], []
                for _, f in frames:
                    boxes = detect(model, f, conf, imgsz, tiles, args.augment)
                    counts.append(len(boxes))
                    heights.extend([b[3] - b[1] for b in boxes])
                ppf = float(np.mean(counts)) if counts else 0.0
                medh = float(np.median(heights)) if heights else 0.0
                p90 = float(np.percentile(heights, 90)) if heights else 0.0
                print(f" {conf:>5} {imgsz:>6} {tiles:>6} | {ppf:>14.2f} {medh:>10.0f} {p90:>7.0f}")
                # Prefer real recall but penalise absurd over-detection (noise) lightly.
                if best is None or ppf > best[0]:
                    best = (ppf, conf, imgsz, tiles)
    print("-" * 72)

    if best and best[0] > 0:
        _, conf, imgsz, tiles = best
        print(f" best: conf={conf} imgsz={imgsz} tiles={tiles}  ({best[0]:.2f} persons/frame)")
        mid = frames[len(frames) // 2][1].copy()
        for b in detect(model, mid, conf, imgsz, tiles, args.augment):
            x0, y0, x1, y1, sc = b
            cv2.rectangle(mid, (int(x0), int(y0)), (int(x1), int(y1)), (60, 220, 60), 1)
            cv2.putText(mid, f"{sc:.2f}", (int(x0), int(y0) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 220, 60), 1)
        out = os.path.join(os.path.dirname(args.video) or ".", "recall_sweep_best.png")
        cv2.imwrite(out, mid)
        print(f" overlay -> {out}")
    else:
        print(" EVERY setting detected ~0 people. This is the signal that a")
        print(" COCO-trained detector cannot see these bodies — switch to an")
        print(" aerial/VisDrone-finetuned detector rather than tuning conf further.")
    print("=" * 72)


if __name__ == "__main__":
    main()
