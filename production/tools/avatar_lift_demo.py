#!/usr/bin/env python3
"""
마네킹 before/after 데모 — 3D 리프트 정제가 마네킹 오버레이를 얼마나 안정화하는지.

2패스(오프라인):
  1) detector+pose+engine으로 전 프레임 person 추출
  2) 트랙별 VideoPose3D 리프트+재투영으로 정제
  각 프레임을 [원본 | base 마네킹 | lift 마네킹] 으로 합성해 mp4+gif 출력.

행동 보존: 두 마네킹 모두 관절을 유지하므로 걷기/타격 등 동작이 그대로 남는다.
lift 쪽은 뼈길이 일정·시간 일관이라 떨림/순간이동/프레임마다 다른 포즈가 줄어든다.

사용:
  python tools/avatar_lift_demo.py _media/sphar/box1.mp4 --max-frames 48 --aerial 0
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.avatar import MannequinRenderer
from edge.pose import SkeletonEngine, make_backend
from edge.pose.lift3d import refine_track
from edge.pose.skeleton import extend_keypoints
from edge.privacy import redact_people
from edge.render import encode_gif

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--pose-preset", default="rtmpose-m")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tiles", type=int, default=1)
    ap.add_argument("--aerial", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--glb", default=os.path.join(HERE, "assets3d", "Xbot.glb"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    backend = make_backend("topview", detector_model=args.model,
                           pose_preset=args.pose_preset, imgsz=args.imgsz,
                           conf=args.conf, tiles=args.tiles, half=False)
    engine = SkeletonEngine(aerial=bool(args.aerial), default_fps=fps)
    avatar = MannequinRenderer(args.glb)

    # pass 1: extract
    frames, per_frame = [], []
    hist = {}
    n = 0
    while True:
        ok, f = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        people = engine.process(backend(f), n / fps)
        frames.append(f)
        ppl = [(p.id, p.ext.copy()) for p in people]
        per_frame.append(ppl)
        for pid, kp in ppl:
            hist.setdefault(pid, []).append((n, kp))
        n += 1
        if n % 10 == 0:
            print(f"  extract {n} frames")
    cap.release()

    # pass 2: refine per track
    refined = {pid: {} for pid in hist}
    for pid, h in hist.items():
        if len(h) < 8:
            for fi, kp in h:
                refined[pid][fi] = kp
            continue
        seq = np.stack([kp for _, kp in h])
        ref, info = refine_track(seq, W, H)
        print(f"  track {pid}: len={len(h)} bone_cv3d={info.get('bone_cv3d')}")
        for i, (fi, _) in enumerate(h):
            refined[pid][fi] = extend_keypoints(ref[i, :17])

    base_name = os.path.splitext(os.path.basename(args.video))[0]
    outdir = os.path.dirname(args.video) or "."
    out_mp4 = os.path.join(outdir, f"liftdemo_{base_name}.mp4")
    out_gif = os.path.join(outdir, f"liftdemo_{base_name}.gif")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 3, H))
    gif_frames = []

    def panel(frame, persons, label):
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        red = redact_people(rgb, persons, draw_skeleton=False)
        comp = avatar.compose(red, persons, supersample=args.supersample)
        bgr = cv2.cvtColor(np.asarray(comp), cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(bgr, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        return bgr

    for fi, frame in enumerate(frames):
        base_persons = per_frame[fi]
        lift_persons = [(pid, refined[pid][fi]) for pid, _ in base_persons if fi in refined.get(pid, {})]
        orig = frame.copy()
        cv2.putText(orig, "original", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(orig, "original", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        b = panel(frame, base_persons, "base mannequin")
        l = panel(frame, lift_persons, "3D-lift mannequin")
        grid = np.hstack([orig, b, l])
        writer.write(grid)
        im = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        im.thumbnail((900, 320))
        gif_frames.append(im)
        if fi % 10 == 0:
            print(f"  render {fi}/{len(frames)}")
    writer.release()
    if gif_frames:
        with open(out_gif, "wb") as fh:
            fh.write(encode_gif(gif_frames, fps=max(1, fps / 2)))
    print(f"[liftdemo] mp4 {out_mp4}")
    print(f"[liftdemo] gif {out_gif}")


if __name__ == "__main__":
    main()
