#!/usr/bin/env python3
"""
Pose + mask fusion preview.

Pose is still the behavior signal. Segmentation is used as an AI consistency
check that lowers confidence for hallucinated joints outside the visible person
shape. The output compares raw pose/mannequin against fused pose/mannequin.

Usage:
  python tools/avatar_pose_mask_fusion.py _media/real_world_hard/caviar_meet_crowd.mpg \
      --pose-model ../yolov8s-pose.pt --mask-source fastsam --max-frames 80
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.avatar import MannequinRenderer
from edge.pose import SkeletonEngine
from edge.pose.mask_fusion import fuse_pose_with_mask, match_masks_to_poses
from edge.privacy import redact_people
from edge.render import EDGES, encode_gif


def _as_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _pose_result(result, W, H):
    out = []
    if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
        return out
    xyxy = _as_np(result.boxes.xyxy).astype(np.float32)
    conf = _as_np(result.boxes.conf).astype(np.float32)
    cls = _as_np(result.boxes.cls).astype(np.int32)
    ids = None
    if getattr(result.boxes, "id", None) is not None:
        ids = _as_np(result.boxes.id).astype(np.int32)
    kxy = _as_np(result.keypoints.xy).astype(np.float32)
    kcf = _as_np(result.keypoints.conf).astype(np.float32)
    for i in range(min(len(xyxy), len(kxy))):
        if int(cls[i]) != 0:
            continue
        kp = np.zeros((17, 3), np.float32)
        kp[:, 0] = np.clip(kxy[i, :, 0] / W, 0, 1)
        kp[:, 1] = np.clip(kxy[i, :, 1] / H, 0, 1)
        kp[:, 2] = np.clip(kcf[i], 0, 1)
        tid = int(ids[i]) if ids is not None and i < len(ids) else None
        out.append({"kp": kp, "bbox_px": xyxy[i], "score": float(conf[i]), "track_id": tid})
    return out


def _mask_result(result, W, H, person_only=False):
    masks, boxes = [], []
    if result.boxes is None or result.masks is None:
        return masks, boxes
    xyxy = _as_np(result.boxes.xyxy).astype(np.float32)
    cls = _as_np(result.boxes.cls).astype(np.int32) if result.boxes.cls is not None else None
    mdata = _as_np(result.masks.data).astype(np.float32)
    for i in range(min(len(xyxy), len(mdata))):
        if person_only and cls is not None and int(cls[i]) != 0:
            continue
        masks.append(cv2.resize(mdata[i], (W, H), interpolation=cv2.INTER_LINEAR))
        boxes.append(xyxy[i])
    return masks, boxes


def _expand_box(box, W, H, pad=0.15):
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return [
        int(np.clip(x0 - bw * pad, 0, W - 1)),
        int(np.clip(y0 - bh * pad, 0, H - 1)),
        int(np.clip(x1 + bw * pad, 0, W - 1)),
        int(np.clip(y1 + bh * pad, 0, H - 1)),
    ]


def _run_mask_model(mask_model, frame, poses, args, W, H):
    if args.mask_source == "none" or not poses:
        return [], []
    if args.mask_source == "yolo-seg":
        res = mask_model.track(frame, persist=True, tracker=args.tracker,
                               imgsz=args.imgsz, conf=args.seg_conf,
                               classes=[0], verbose=False)[0]
        return _mask_result(res, W, H, person_only=True)

    boxes = [_expand_box(p["bbox_px"], W, H, pad=args.sam_box_pad) for p in poses]
    res = mask_model.predict(frame, imgsz=args.sam_imgsz, conf=args.sam_conf,
                             iou=args.sam_iou, bboxes=boxes, verbose=False)[0]
    return _mask_result(res, W, H, person_only=False)


def _norm_box(box, W, H):
    x0, y0, x1, y1 = box
    return np.array([x0 / W, y0 / H, x1 / W, y1 / H], np.float32)


def _draw_pose(frame, persons, title):
    out = frame.copy()
    H, W = out.shape[:2]
    for pid, kp in persons:
        pts = [(int(x * W), int(y * H), c) for x, y, c in kp[:17]]
        for a, b in EDGES:
            if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
                cv2.line(out, pts[a][:2], pts[b][:2], (60, 210, 245), 2, cv2.LINE_AA)
        for x, y, c in pts:
            if c > 0.2:
                cv2.circle(out, (x, y), 3, (245, 245, 245), -1, cv2.LINE_AA)
        xs = [p[0] for p in pts if p[2] > 0.2]
        ys = [p[1] for p in pts if p[2] > 0.2]
        if xs:
            cv2.putText(out, f"ID{pid}", (min(xs), max(12, min(ys) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 210, 245), 1, cv2.LINE_AA)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 4, cv2.LINE_AA)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _draw_masks(frame, masks):
    out = frame.copy()
    for mask in masks:
        solid = (mask > 0.35).astype(np.uint8)
        contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (80, 240, 100), 1, cv2.LINE_AA)
    return out


def _avatar_panel(frame_bgr, avatar, persons, label):
    rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    red = redact_people(rgb, persons, draw_skeleton=False)
    comp = avatar.compose(red, persons, supersample=1)
    out = cv2.cvtColor(np.asarray(comp), cv2.COLOR_RGB2BGR)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 4, cv2.LINE_AA)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _contact_sheet(frames, out_path, cols=2):
    if not frames:
        return
    imgs = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((980, 420))
        imgs.append(im.copy())
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    rows = int(np.ceil(len(imgs) / cols))
    sheet = Image.new("RGB", (w * cols, h * rows), (16, 16, 18))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pose-model", default=os.path.join("..", "yolov8s-pose.pt"))
    ap.add_argument("--mask-source", default="fastsam",
                    choices=["fastsam", "yolo-seg", "none"])
    ap.add_argument("--seg-model", default="yolov8n-seg.pt")
    ap.add_argument("--sam-model", default="FastSAM-s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--sam-imgsz", type=int, default=640)
    ap.add_argument("--pose-conf", type=float, default=0.15)
    ap.add_argument("--seg-conf", type=float, default=0.12)
    ap.add_argument("--sam-conf", type=float, default=0.25)
    ap.add_argument("--sam-iou", type=float, default=0.90)
    ap.add_argument("--sam-box-pad", type=float, default=0.15)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--glb", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "assets3d", "Xbot.glb"))
    args = ap.parse_args()

    from ultralytics import FastSAM, YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    pose_model = YOLO(args.pose_model)
    if args.mask_source == "fastsam":
        mask_model = FastSAM(args.sam_model)
        mask_name = args.sam_model
    elif args.mask_source == "yolo-seg":
        mask_model = YOLO(args.seg_model)
        mask_name = args.seg_model
    else:
        mask_model = None
        mask_name = "none"
    raw_engine = SkeletonEngine(aerial=True, default_fps=fps)
    fused_engine = SkeletonEngine(aerial=True, default_fps=fps)
    avatar = MannequinRenderer(args.glb)

    base = os.path.splitext(os.path.basename(args.video))[0]
    outdir = os.path.dirname(args.video) or "."
    tag = (f"fusion_{base}_p{os.path.splitext(os.path.basename(args.pose_model))[0]}"
           f"_{args.mask_source}_{os.path.splitext(os.path.basename(mask_name))[0]}")
    out_mp4 = os.path.join(outdir, f"{tag}.mp4")
    out_gif = os.path.join(outdir, f"{tag}.gif")
    out_sheet = os.path.join(outdir, f"{tag}_contact.png")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H * 2))

    print(f"[fusion] {args.video} {W}x{H}@{fps:.0f} total={total}")
    print(f"[fusion] pose={args.pose_model} imgsz={args.imgsz}")
    print(f"[fusion] mask={args.mask_source}:{mask_name}")

    frames_for_gif = []
    frames_for_sheet = []
    stats = []
    raw_counts, fused_counts = [], []
    t0 = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (args.start + n) / fps
        pose_res = pose_model.track(frame, persist=True, tracker=args.tracker,
                                    imgsz=args.imgsz, conf=args.pose_conf,
                                    classes=[0], verbose=False)[0]
        poses = _pose_result(pose_res, W, H)
        masks, mask_boxes = _run_mask_model(mask_model, frame, poses, args, W, H)
        matches = match_masks_to_poses([p["bbox_px"] for p in poses], masks, mask_boxes)

        raw_dets, fused_dets = [], []
        frame_stats = {"outside": 0, "bad_limbs": 0, "matched": 0, "poses": len(poses)}
        for p, mi in zip(poses, matches):
            raw_dets.append({
                "kp": p["kp"],
                "bbox": _norm_box(p["bbox_px"], W, H),
                "score": p["score"],
                "track_id": p["track_id"],
            })
            kp = p["kp"]
            if mi is not None:
                kp, info = fuse_pose_with_mask(kp, masks[mi])
                frame_stats["outside"] += info["outside_points"]
                frame_stats["bad_limbs"] += info["bad_limbs"]
                frame_stats["matched"] += 1
            fused_dets.append({
                "kp": kp,
                "bbox": _norm_box(p["bbox_px"], W, H),
                "score": p["score"],
                "track_id": p["track_id"],
            })
        stats.append(frame_stats)

        raw_people = raw_engine.process(raw_dets, t)
        fused_people = fused_engine.process(fused_dets, t)
        raw_persons = [p.as_tuple() for p in raw_people]
        fused_persons = [p.as_tuple() for p in fused_people]
        raw_counts.append(len(raw_persons))
        fused_counts.append(len(fused_persons))

        original = _draw_masks(frame, masks)
        raw_pose = _draw_pose(frame, raw_persons, "raw pose")
        fused_pose = _draw_pose(frame, fused_persons, "mask-fused pose")
        fused_avatar = _avatar_panel(frame, avatar, fused_persons, "fused mannequin")
        grid = np.vstack([np.hstack([original, raw_pose]), np.hstack([fused_pose, fused_avatar])])
        writer.write(grid)
        if n % 3 == 0:
            im = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
            im.thumbnail((760, 520))
            frames_for_gif.append(im.copy())
        if len(frames_for_sheet) < 8 and n % max(1, args.max_frames // 8) == 0:
            frames_for_sheet.append(grid.copy())

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 20 == 0:
            print(f"   ...{n} frames raw={np.mean(raw_counts):.2f}/fused={np.mean(fused_counts):.2f} "
                  f"matched={np.mean([s['matched'] for s in stats]):.2f} "
                  f"bad_limbs={np.mean([s['bad_limbs'] for s in stats]):.2f}")

    cap.release()
    writer.release()
    if frames_for_gif:
        with open(out_gif, "wb") as f:
            f.write(encode_gif(frames_for_gif, fps=max(1, fps / 3)))
    _contact_sheet(frames_for_sheet, out_sheet)

    dur = time.time() - t0
    mean = lambda k: float(np.mean([s[k] for s in stats])) if stats else 0.0
    print("=" * 72)
    print(f" POSE+MASK FUSION REPORT - {base}")
    print(f" frames          : {n} ({n / max(dur, 1e-6):.2f} fps)")
    print(f" raw persons/fr  : {np.mean(raw_counts) if raw_counts else 0:.2f}")
    print(f" fused persons/fr: {np.mean(fused_counts) if fused_counts else 0:.2f}")
    print(f" matched masks/fr: {mean('matched'):.2f} / poses/fr {mean('poses'):.2f}")
    print(f" outside kpts/fr : {mean('outside'):.2f}")
    print(f" bad limbs/fr    : {mean('bad_limbs'):.2f}")
    print(f" mp4             : {out_mp4}")
    print(f" gif             : {out_gif}")
    print(f" contact sheet   : {out_sheet}")
    print("=" * 72)


if __name__ == "__main__":
    main()
