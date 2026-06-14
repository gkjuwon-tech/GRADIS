#!/usr/bin/env python3
"""
Footage probe — gate a clip BEFORE spending a pipeline run on it.

The top-view mission has two hard preconditions a clip must satisfy
(HANDOFF §4-A) before any eye-verification is meaningful:

  1. the people are big enough to actually see a mannequin on them
     (rule of thumb: tallest person bbox >= ~150 px),
  2. the shot is genuinely top-down / aerial, not a side/oblique view that
     ordinary COCO pose already solves.

(2) can't be decided by a number, so this tool just makes (1) measurable and
dumps an annotated contact sheet of evenly-spaced frames (person boxes +
heights drawn) so a human can eye-confirm the viewpoint in one glance.

Usage:
  python tools/footage_probe.py _media/topdown/clip.mp4 --samples 9
  python tools/footage_probe.py CLIP --model yolo11x.pt --imgsz 1280 --conf 0.25
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _detect(model, frame, imgsz, conf):
    res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0],
                        verbose=False)
    boxes = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes.xyxy.cpu().numpy():
            boxes.append(b.astype(float))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--samples", type=int, default=9)
    ap.add_argument("--out", default=None, help="contact-sheet PNG path")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    idxs = np.linspace(0, max(0, total - 1), args.samples).astype(int)
    tiles, all_heights = [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = _detect(model, frame, args.imgsz, args.conf)
        hs = []
        for (x0, y0, x1, y1) in boxes:
            h = y1 - y0
            hs.append(h)
            all_heights.append(h)
            color = (60, 220, 60) if h >= 150 else (60, 160, 245)
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
            cv2.putText(frame, f"{int(h)}px", (int(x0), max(14, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        tallest = max(hs) if hs else 0
        cv2.putText(frame, f"f{int(fi)} n={len(boxes)} tall={int(tallest)}px",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"f{int(fi)} n={len(boxes)} tall={int(tallest)}px",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(frame)

    if not tiles:
        sys.exit("no frames read")

    # contact sheet
    cols = int(math.ceil(math.sqrt(len(tiles))))
    rows = int(math.ceil(len(tiles) / cols))
    th = 360
    tw = int(W * th / H)
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(t, (tw, th))

    out = args.out or os.path.join(
        os.path.dirname(args.video) or ".",
        f"probe_{os.path.splitext(os.path.basename(args.video))[0]}.png")
    cv2.imwrite(out, sheet)

    a = np.array(all_heights) if all_heights else np.array([0.0])
    print(f"clip      : {os.path.basename(args.video)}  {W}x{H}@{fps:.0f} {total}f")
    print(f"detections: {len(all_heights)} person boxes over {len(tiles)} frames")
    print(f"height px : max={a.max():.0f} median={np.median(a):.0f} min={a.min():.0f}")
    print(f"GATE >=150: {'PASS' if a.max() >= 150 else 'FAIL'} (tallest={a.max():.0f}px)")
    print(f"contact   : {out}")


if __name__ == "__main__":
    main()
