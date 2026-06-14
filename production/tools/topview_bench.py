#!/usr/bin/env python3
"""
Container-runnable top-view pose + mannequin benchmark.

This is the verification harness the night-shift loop never had: it runs the
WHOLE anonymization boundary (detector -> RTMPose ONNX -> SkeletonEngine ->
mannequin overlay) on real multi-angle CCTV clips, on a bare CPU container, with
NO OpenMMLab and NO manual zip uploads. Weights download themselves on first run.

For each clip it writes a 3-panel video (original | skeleton | mannequin) plus a
contact sheet, and prints the numbers that decide whether top-view behavior is
actually preserved:

  persons/frame   how many people we hold each frame
  kp_conf         mean confidence of visible joints (pose trustworthiness)
  jitter          per-mille frame-to-frame joint movement of TRACKED people
                  (the "mannequin slides / teleports" complaint, quantified)
  render_rate     fraction of people the mannequin actually skinned onto
  flicker/100     mannequin dropouts per 100 person-frames (the "blink" complaint)

Usage:
  python tools/topview_bench.py                       # all clips in _media/sphar_bench
  python tools/topview_bench.py _media/sphar_bench/falling_casia_topdownview_p01_faint_a1.mp4
  python tools/topview_bench.py --max-frames 200 --mode performance
"""

from __future__ import annotations

import argparse
import glob
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
from edge.pose import SkeletonEngine, make_backend
from edge.privacy import redact_people
from edge.render import EDGES, EDGES_EXT


def _angle_of(name: str) -> str:
    low = name.lower()
    for tag in ("topdownview", "angleview", "horizontalview"):
        if tag in low:
            return tag.replace("view", "")
    return "other"


def _draw_skeleton(frame, persons, title):
    out = frame.copy()
    H, W = out.shape[:2]
    for pid, kp in persons:
        n = kp.shape[0]
        edges = EDGES + (EDGES_EXT if n >= 25 else [])
        pts = [(int(x * W), int(y * H), c) for x, y, c in kp]
        for a, b in edges:
            if a < n and b < n and kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
                cv2.line(out, pts[a][:2], pts[b][:2], (60, 210, 245), 2, cv2.LINE_AA)
        for x, y, c in pts:
            if c > 0.2:
                cv2.circle(out, (x, y), 3, (245, 245, 245), -1, cv2.LINE_AA)
        xs = [p[0] for p in pts if p[2] > 0.2]
        ys = [p[1] for p in pts if p[2] > 0.2]
        if xs:
            cv2.putText(out, f"ID{pid}", (min(xs), max(12, min(ys) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 210, 245), 1, cv2.LINE_AA)
    _label(out, title)
    return out


def _mannequin_panel(frame_bgr, avatar, persons, title):
    rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    red = redact_people(rgb, persons, draw_skeleton=False)
    comp = avatar.compose(red, persons, supersample=1)
    out = cv2.cvtColor(np.asarray(comp), cv2.COLOR_RGB2BGR)
    _label(out, title)
    return out


def _label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)


def _contact_sheet(frames, out_path, cols=2):
    if not frames:
        return
    imgs = []
    for f in frames:
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        im.thumbnail((900, 360))
        imgs.append(im.copy())
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    rows = int(np.ceil(len(imgs) / cols))
    sheet = Image.new("RGB", (w * cols, h * rows), (16, 16, 18))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


def run_clip(path, backend, avatar, args):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [skip] cannot open {path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = float(np.hypot(W, H))
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    engine = SkeletonEngine(aerial=True, default_fps=fps, spawn_conf=args.spawn_conf)
    base = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(os.path.dirname(path) or ".", "bench_out")
    os.makedirs(outdir, exist_ok=True)
    out_mp4 = os.path.join(outdir, f"bench_{base}.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 3, H))

    persons_pf, kp_conf, jitter_samples = [], [], []
    render_ok, render_total, flicker = 0, 0, 0
    prev_kp = {}        # track_id -> ext[:, :3]  (previous frame, image px)
    prev_rendered = set()
    sheet_frames = []
    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (args.start + n) / fps
        dets = backend(frame)
        people = engine.process(dets, t)
        persons = [p.as_tuple() for p in people]
        persons_pf.append(len(persons))

        rendered_now = set()
        present_now = {pid for pid, _ in persons}
        for pid, kp in persons:
            vis = kp[:, 2] >= 0.15
            if vis.any():
                kp_conf.append(float(kp[vis, 2].mean()))
            # temporal jitter: confident-joint movement vs previous frame
            cur = kp.copy()
            if pid in prev_kp:
                pk = prev_kp[pid]
                both = (cur[:, 2] >= 0.2) & (pk[:, 2] >= 0.2)
                if both.any():
                    d = np.hypot((cur[both, 0] - pk[both, 0]) * W,
                                 (cur[both, 1] - pk[both, 1]) * H)
                    jitter_samples.append(float(np.median(d)) / diag * 1000.0)
            prev_kp[pid] = cur
            # mannequin render success (skin returns None on degenerate pose)
            render_total += 1
            skinned = avatar._skin(np.asarray(kp, np.float32), W, H, pid) is not None
            if skinned:
                render_ok += 1
                rendered_now.add(pid)
        # flicker = a track still present that rendered last frame but blinks out now
        flicker += len((prev_rendered & present_now) - rendered_now)
        prev_rendered = rendered_now

        if args.video:
            orig = frame.copy(); _label(orig, "original")
            skel = _draw_skeleton(frame, persons, "skeleton (RTMPose+engine)")
            mann = _mannequin_panel(frame, avatar, persons, "mannequin overlay")
            grid = np.hstack([orig, skel, mann])
            writer.write(grid)
            if len(sheet_frames) < 6 and n % max(1, args.max_frames // 6) == 0:
                sheet_frames.append(grid.copy())

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    writer.release()
    if args.video and sheet_frames:
        _contact_sheet(sheet_frames, os.path.join(outdir, f"bench_{base}_contact.png"))
    elif not args.video and os.path.exists(out_mp4):
        os.remove(out_mp4)

    dur = time.time() - t0
    return {
        "clip": base,
        "angle": _angle_of(base),
        "frames": n,
        "fps_proc": n / max(dur, 1e-6),
        "persons_pf": float(np.mean(persons_pf)) if persons_pf else 0.0,
        "kp_conf": float(np.mean(kp_conf)) if kp_conf else 0.0,
        "jitter": float(np.median(jitter_samples)) if jitter_samples else 0.0,
        "render_rate": render_ok / max(1, render_total),
        "flicker_100": flicker / max(1, render_total) * 100.0,
        "mp4": out_mp4 if args.video else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="clips or dirs (default: _media/sphar_bench)")
    ap.add_argument("--backend", default="rtmpose-onnx")
    ap.add_argument("--mode", default="balanced",
                    choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--spawn-conf", type=float, default=0.30,
                    help="min detection score to start a track (top-view needs lower)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="metrics only, skip mp4/contact-sheet rendering")
    ap.add_argument("--glb", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets3d", "Xbot.glb"))
    ap.set_defaults(video=True)
    args = ap.parse_args()

    targets = args.videos or [os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_media", "sphar_bench")]
    clips = []
    for t in targets:
        if os.path.isdir(t):
            clips.extend(sorted(glob.glob(os.path.join(t, "*.mp4"))))
        else:
            clips.append(t)
    if not clips:
        raise SystemExit("no clips found. Run tools/fetch_bench_videos.py first.")

    print(f"[bench] backend={args.backend} mode={args.mode} clips={len(clips)}")
    backend = make_backend(args.backend, mode=args.mode)
    avatar = MannequinRenderer(args.glb)

    rows = []
    for c in clips:
        print(f"[bench] {os.path.basename(c)} ...")
        r = run_clip(c, backend, avatar, args)
        if r:
            rows.append(r)

    print("\n" + "=" * 104)
    print(f" TOP-VIEW BENCH — backend={args.backend} mode={args.mode}")
    print("-" * 104)
    hdr = (f"{'clip':40s} {'angle':10s} {'pers/f':>7s} {'kp_conf':>8s} "
           f"{'jitter':>7s} {'render':>7s} {'flick/100':>9s} {'fps':>5s}")
    print(hdr)
    print("-" * 104)
    for r in rows:
        print(f"{r['clip'][:40]:40s} {r['angle']:10s} {r['persons_pf']:7.2f} "
              f"{r['kp_conf']:8.2f} {r['jitter']:7.2f} {r['render_rate']:7.2f} "
              f"{r['flicker_100']:9.2f} {r['fps_proc']:5.1f}")
    print("-" * 104)
    by_angle = {}
    for r in rows:
        by_angle.setdefault(r["angle"], []).append(r)
    print(" per-angle means (the completion criterion: stable across ALL angles):")
    for ang, rs in sorted(by_angle.items()):
        m = lambda k: float(np.mean([x[k] for x in rs]))
        print(f"   {ang:12s} pers/f={m('persons_pf'):.2f}  kp_conf={m('kp_conf'):.2f}  "
              f"jitter={m('jitter'):.2f}  render={m('render_rate'):.2f}  "
              f"flick/100={m('flicker_100'):.2f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
