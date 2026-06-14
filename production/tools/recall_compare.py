#!/usr/bin/env python3
"""
Small-person detector recall comparison.

This tool intentionally measures YOLO person detection only. Pose extraction is
handled later by MMPose, so aerial/top-view recall should be diagnosed at the
person-box stage first.

Usage:
  python tools/recall_compare.py _media/aerial_4456.mp4 --model yolo11x.pt --tiles 3 --n 16
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.pose.backends import YoloPersonDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", "--detector-model", dest="detector_model",
                    default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = YoloPersonDetector(model=args.detector_model, imgsz=args.imgsz,
                              conf=args.conf, tiles=1, half=False)
    tiled = YoloPersonDetector(model=args.detector_model, imgsz=args.imgsz,
                               conf=args.conf, tiles=args.tiles, half=False)

    idxs = np.linspace(total * 0.2, total * 0.8, args.n).astype(int)
    base_counts, tiled_counts = [], []
    saved = False
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        b = base(frame)
        r = tiled(frame)
        base_counts.append(len(b))
        tiled_counts.append(len(r))
        if not saved and k == len(idxs) // 2:
            _sample(frame, b, r, args.video)
            saved = True
    cap.release()

    mb = float(np.mean(base_counts)) if base_counts else 0.0
    mr = float(np.mean(tiled_counts)) if tiled_counts else 0.0
    print("=" * 60)
    print(f" DETECTOR RECALL - {os.path.basename(args.video)} ({args.n} frames, tiles={args.tiles})")
    print("-" * 60)
    print(f" baseline single-scale : {mb:.1f} persons/frame")
    print(f" tiled detector        : {mr:.1f} persons/frame")
    print(f"  -> small-person recall {(mr / mb - 1) * 100:+.0f}%" if mb > 0 else "  -> baseline 0")
    print("=" * 60)


def _sample(frame, base_boxes, tiled_boxes, video):
    def draw(boxes, color):
        out = frame.copy()
        for box in boxes:
            x0, y0, x1, y1 = box.xyxy.astype(int)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        return out

    left = draw(base_boxes, (60, 60, 255))
    cv2.putText(left, f"single n={len(base_boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2)
    right = draw(tiled_boxes, (60, 220, 60))
    cv2.putText(right, f"tiled n={len(tiled_boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 60), 2)
    out = os.path.join(os.path.dirname(video), "recall_compare.png")
    cv2.imwrite(out, np.hstack([left, right]))
    print(f" sample -> {out}")


if __name__ == "__main__":
    main()

