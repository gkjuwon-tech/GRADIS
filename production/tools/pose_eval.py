#!/usr/bin/env python3
"""
Viewpoint-robust pose stability/continuity metrics — WITHOUT ground truth.

The top-view failures the team keeps hitting are all *temporal* and map to
concrete, measurable signals over a track's lifetime:

  user complaint                         -> metric
  ------------------------------------   -----------------------------------
  "마네킹이 순간이동/슬라이딩"           -> jitter_norm, accel_norm (낮을수록 좋음)
  "프레임마다 다른 작은 포즈"            -> bone_cv (뼈 길이 변동계수)
  "사람/마네킹이 깜빡거림"               -> flicker (관절 가시성 토글), id_churn
  "위에서 본 사람을 포즈화 못함"         -> anat_score, detect_rate
  "환각 관절이 실루엣 밖으로 튐"         -> outside_rate (mask 있을 때)

Everything is normalized by per-track torso size so it is comparable across
clips and viewpoints. No camera-angle hardcoding: we just measure how
self-consistent a body is through time, which is exactly what a correct pose
must be regardless of the angle it was filmed from.

Runs the REAL production path (make_backend + SkeletonEngine), so the numbers
describe what the mannequin layer actually receives.

Usage:
  python tools/pose_eval.py _media/sphar/top_walk.mp4 \
      --model yolo11m.pt --pose-preset rtmpose-m --imgsz 640 --max-frames 120
  python tools/pose_eval.py CLIP --backend-json '{...}' --tag rtmpose
  python tools/pose_eval.py CLIP --compare results_a.json results_b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.pose import SkeletonEngine, make_backend
from edge.pose.skeleton import BONES, anatomical_validity, torso_size

CONF_VIS = 0.2


def _track_metrics(history):
    """history: list of (t, kp[25,3]) for ONE track id. Returns metric dict."""
    if len(history) < 3:
        return None
    kps = np.stack([h[1] for h in history])            # [T,25,3]
    T = kps.shape[0]
    scales = np.array([torso_size(k) or np.nan for k in kps])
    scale = np.nanmedian(scales)
    if not np.isfinite(scale) or scale < 1e-4:
        return None

    xy = kps[:, :17, :2]
    vis = kps[:, :17, 2] > CONF_VIS

    # jitter: frame-to-frame displacement of jointly-visible joints / torso
    d = np.linalg.norm(np.diff(xy, axis=0), axis=2)     # [T-1,17]
    vv = vis[1:] & vis[:-1]
    jitter = float(np.median(d[vv]) / scale) if vv.any() else np.nan

    # accel (jerk proxy): 2nd difference magnitude / torso
    a = np.linalg.norm(np.diff(xy, n=2, axis=0), axis=2)  # [T-2,17]
    av = vis[2:] & vis[1:-1] & vis[:-2]
    accel = float(np.median(a[av]) / scale) if av.any() else np.nan

    # bone length variation: CV of each bone's projected length over time
    cvs = []
    for (i, j) in BONES:
        m = vis[:, i] & vis[:, j]
        if m.sum() < max(3, T * 0.3):
            continue
        L = np.linalg.norm(xy[m, i] - xy[m, j], axis=1)
        mu = L.mean()
        if mu > 1e-5:
            cvs.append(L.std() / mu)
    bone_cv = float(np.mean(cvs)) if cvs else np.nan

    # flicker: how often a joint's visibility toggles per frame
    toggles = np.abs(np.diff(vis.astype(np.int8), axis=0)).sum()
    flicker = float(toggles) / (T * 17)

    anat = float(np.mean([anatomical_validity(k[:17]) for k in kps]))
    return dict(T=T, jitter=jitter, accel=accel, bone_cv=bone_cv,
                flicker=flicker, anat=anat)


def _refine_histories(histories, w, h):
    """In-place 3D lift + reproject of each track's COCO-17 sequence."""
    from edge.pose.lift3d import refine_track
    from edge.pose.skeleton import extend_keypoints
    bone_cv3ds = []
    for tid, hist in histories.items():
        if len(hist) < 8:
            continue
        ts = [t for t, _ in hist]
        seq = np.stack([k for _, k in hist])          # [T,25,3]
        ref, info = refine_track(seq, w, h)
        if info.get("bone_cv3d") is not None:
            bone_cv3ds.append((info["bone_cv3d"], len(hist)))
        new = []
        for i, t in enumerate(ts):
            ext = extend_keypoints(ref[i, :17])        # rebuild 17..24
            new.append((t, ext))
        histories[tid] = new
    if not bone_cv3ds:
        return None
    v = np.array([x[0] for x in bone_cv3ds]); wt = np.array([x[1] for x in bone_cv3ds])
    return float((v * wt).sum() / wt.sum())


def evaluate(video, backend_kwargs, max_frames=120, start=0, aerial=True,
             annotate=None, lift=False):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    backend = make_backend("topview", half=False, **backend_kwargs)
    engine = SkeletonEngine(aerial=aerial, default_fps=fps)

    writer = None
    if annotate:
        writer = cv2.VideoWriter(annotate, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))

    histories = defaultdict(list)
    per_frame_persons = []
    seen_ids = set()
    id_first_frame = {}
    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (start + n) / fps
        raw = backend(frame)
        people = engine.process(raw, t)
        per_frame_persons.append(len(people))
        for p in people:
            histories[p.id].append((t, p.ext.copy()))
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                id_first_frame[p.id] = n
        if writer is not None:
            writer.write(_draw(frame, people))
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()

    bone_cv3d = _refine_histories(histories, W, H) if lift else None

    tm = [m for m in (_track_metrics(h) for h in histories.values()) if m]
    # weight per-track metrics by track length (long stable tracks matter more)
    def wmean(key):
        vals = [(m[key], m["T"]) for m in tm if np.isfinite(m[key])]
        if not vals:
            return None
        v = np.array([x[0] for x in vals]); w = np.array([x[1] for x in vals])
        return float((v * w).sum() / w.sum())

    # id_churn: new track ids created per 100 frames (lower = more continuous)
    id_churn = 100.0 * len(seen_ids) / max(1, n)
    track_lens = [m["T"] for m in tm]
    return {
        "video": os.path.basename(video),
        "frames": n,
        "fps_proc": round(n / max(time.time() - t0, 1e-6), 2),
        "detect_rate": round(float(np.mean(per_frame_persons)), 3) if per_frame_persons else 0,
        "n_tracks": len(seen_ids),
        "n_scored_tracks": len(tm),
        "mean_track_len": round(float(np.mean(track_lens)), 1) if track_lens else 0,
        "id_churn_per100": round(id_churn, 2),
        "jitter": _r(wmean("jitter")),
        "accel": _r(wmean("accel")),
        "bone_cv": _r(wmean("bone_cv")),
        "flicker": _r(wmean("flicker")),
        "anat": _r(wmean("anat")),
        "bone_cv3d": _r(bone_cv3d),
    }


def _r(x, k=4):
    return None if x is None else round(x, k)


def _draw(frame, people):
    out = frame.copy()
    H, W = out.shape[:2]
    for p in people:
        kp = p.ext
        for a, b in BONES:
            if kp[a, 2] > CONF_VIS and kp[b, 2] > CONF_VIS:
                pa = (int(kp[a, 0]), int(kp[a, 1]))
                pb = (int(kp[b, 0]), int(kp[b, 1]))
                cv2.line(out, pa, pb, (60, 210, 245), 1, cv2.LINE_AA)
        for j in range(17):
            if kp[j, 2] > CONF_VIS:
                cv2.circle(out, (int(kp[j, 0]), int(kp[j, 1])), 2,
                           (245, 245, 245), -1, cv2.LINE_AA)
    return out


LOWER_BETTER = {"jitter", "accel", "bone_cv", "flicker", "id_churn_per100"}
HIGHER_BETTER = {"detect_rate", "anat", "mean_track_len"}


def _compare(paths):
    runs = [json.load(open(p)) for p in paths]
    keys = ["detect_rate", "n_tracks", "mean_track_len", "id_churn_per100",
            "jitter", "accel", "bone_cv", "flicker", "anat", "bone_cv3d"]
    names = [r.get("tag", os.path.basename(p)) for r, p in zip(runs, paths)]
    print(f"{'metric':18s} " + " ".join(f"{n:>14s}" for n in names) + "   winner")
    print("-" * (20 + 15 * len(names) + 10))
    for k in keys:
        vals = [r.get(k) for r in runs]
        cells = " ".join(f"{('' if v is None else f'{v:.4g}'):>14s}" for v in vals)
        fin = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
        win = ""
        if fin and k in LOWER_BETTER:
            win = names[min(fin, key=lambda x: x[1])[0]]
        elif fin and k in HIGHER_BETTER:
            win = names[max(fin, key=lambda x: x[1])[0]]
        print(f"{k:18s} {cells}   {win}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", "--detector-model", dest="detector_model",
                    default="yolo11m.pt")
    ap.add_argument("--pose-preset", default="rtmpose-m")
    ap.add_argument("--pose-config", default=None)
    ap.add_argument("--pose-checkpoint", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tiles", type=int, default=1)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--annotate", default=None, help="write skeleton overlay mp4")
    ap.add_argument("--lift", action="store_true",
                    help="apply VideoPose3D 3D lift + reproject refinement")
    ap.add_argument("--rot-tta", action="store_true",
                    help="top-view rotation test-time augmentation (0/45/.../315)")
    ap.add_argument("--tta-angles", default=None,
                    help="comma-separated TTA angles (deg), overrides --rot-tta default")
    ap.add_argument("--tta-margin", type=float, default=0.35,
                    help="bbox expansion margin for the rotated crop")
    ap.add_argument("--compare", nargs="+", help="print a comparison table of result JSONs and exit")
    args = ap.parse_args()

    if args.compare:
        _compare(args.compare)
        return

    rot_tta = ()
    if args.tta_angles:
        rot_tta = tuple(int(a) for a in args.tta_angles.split(",") if a.strip())
    elif args.rot_tta:
        from edge.pose.backends import DEFAULT_ROT_TTA
        rot_tta = DEFAULT_ROT_TTA
    bk = dict(detector_model=args.detector_model, pose_preset=args.pose_preset,
              pose_config=args.pose_config, pose_checkpoint=args.pose_checkpoint,
              imgsz=args.imgsz, conf=args.conf, tiles=args.tiles,
              tracker=args.tracker, rot_tta=rot_tta, tta_margin=args.tta_margin)
    res = evaluate(args.video, bk, max_frames=args.max_frames, start=args.start,
                   annotate=args.annotate, lift=args.lift)
    res["tag"] = args.tag or args.pose_preset
    res["config"] = {k: bk[k] for k in ("detector_model", "pose_preset", "imgsz",
                                        "conf", "tiles")}
    res["config"]["rot_tta"] = list(rot_tta)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[pose_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
