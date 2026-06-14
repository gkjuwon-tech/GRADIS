#!/usr/bin/env python3
"""
아바타 합성 파이프라인 테스트 — 'VLM이 보게 될 화면'을 영상으로 뽑는다.

  원본 → YOLO detector/tracker + MMPose + SkeletonEngine
       → privacy.redact_people(사람 강블러, 기존 그대로)
       → MannequinRenderer(졸라맨 → 회색 마네킹 빙의)  ← 새 레이어
       → 합성 mp4 + 최종 GIF

사용:
  python tools/avatar_on_video.py _media/wide_45881.mp4 --max-frames 250
  python tools/avatar_on_video.py _media/wide_45881.mp4 --single 100   # 한 프레임 PNG만
"""
import sys, os, time, argparse
import numpy as np
import cv2
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.pose import SkeletonEngine, make_backend
from edge.privacy import redact_people
from edge.avatar import MannequinRenderer
from edge.render import encode_gif

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", "--detector-model", dest="detector_model",
                    default="yolo11x.pt",
                    help="YOLO person detector weights, not *-pose.pt")
    ap.add_argument("--pose-preset", default="rtmpose-m",
                    choices=["rtmpose-m", "vitpose-s"])
    ap.add_argument("--pose-config", default=None)
    ap.add_argument("--pose-checkpoint", default=None)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--rot-tta", action="store_true",
                    help="top-view rotation test-time augmentation (0/45/.../315)")
    ap.add_argument("--tta-margin", type=float, default=0.35)
    ap.add_argument("--tiles", type=int, default=1,
                    help=">1 enables NxN tile inference for small/distant people")
    ap.add_argument("--glb", default=os.path.join(HERE, "assets3d", "Xbot.glb"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--single", type=int, default=-1,
                    help="해당 프레임 1장만 PNG로 (디버그)")
    ap.add_argument("--skeleton", action="store_true",
                    help="마네킹 위에 스켈레톤 라인도 함께")
    ap.add_argument("--supersample", type=int, default=2,
                    help="오프라인 고품질 렌더 배율(2 권장, 시임/계단 제거). 1=라이브 동급")
    ap.add_argument("--no-cache", action="store_true",
                    help="kp 캐시 로드/저장 안 함(강제 재추출)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"못 엶: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    avatar = MannequinRenderer(args.glb)

    base = os.path.splitext(os.path.basename(args.video))[0]
    outdir = os.path.dirname(args.video) or "."
    # kp 캐시: YOLO 포즈 추출은 한 번만. 이후 렌더러 튜닝은 캐시로 재렌더(연산 최소화).
    model_tag = os.path.splitext(os.path.basename(args.detector_model))[0].replace(".", "_")
    cache_tag = f"{model_tag}_{args.pose_preset}_i{args.imgsz}_t{args.tiles}_c{args.conf:g}"
    if args.rot_tta:
        cache_tag += "_tta"
    cache_path = os.path.join(outdir, f"avatar_{base}_{cache_tag}_kp.npz")
    use_cache = (not args.no_cache) and args.single < 0 and os.path.exists(cache_path)
    cached = None
    if use_cache:
        cached = list(np.load(cache_path, allow_pickle=True)["persons"])
        backend = engine = None
        print(f"[avatar] kp 캐시 로드: {cache_path} ({len(cached)}f) — YOLO 스킵")
    else:
        from edge.pose.backends import DEFAULT_ROT_TTA
        backend = make_backend("topview",
                               detector_model=args.detector_model,
                               pose_preset=args.pose_preset,
                               pose_config=args.pose_config,
                               pose_checkpoint=args.pose_checkpoint,
                               imgsz=args.imgsz,
                               conf=args.conf,
                               tiles=args.tiles,
                               tracker=args.tracker,
                               rot_tta=(DEFAULT_ROT_TTA if args.rot_tta else ()),
                               tta_margin=args.tta_margin,
                               half=False)
        engine = SkeletonEngine(aerial=True, default_fps=fps)
    out_mp4 = os.path.join(outdir, f"avatar_{base}.mp4")
    out_gif = os.path.join(outdir, f"avatar_{base}.gif")
    writer = None
    if args.single < 0:
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"[avatar] {args.video} {W}x{H}@{fps:.0f} {total}f  "
          f"detector={os.path.basename(args.detector_model)}@{args.imgsz} "
          f"pose={args.pose_preset} "
          f"conf={args.conf:g} tiles={args.tiles} "
          f"rig={os.path.basename(args.glb)}")
    n = 0
    gif_frames = []
    rec_persons = []          # 캐시 저장용(이번에 추출한 프레임별 persons)
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.single >= 0 and n < args.single - 6:
            n += 1
            continue   # 트래커 워밍업(confirmed까지 3히트)을 위해 6프레임 전부터 처리

        t = n / fps
        if use_cache:
            persons = list(cached[n]) if n < len(cached) else []
        else:
            raw = backend(frame)
            people = engine.process(raw, t)
            persons = [p.as_tuple() for p in people]      # (id, kp25)
            rec_persons.append(persons)

        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        red = redact_people(rgb, persons, draw_skeleton=args.skeleton)
        out = avatar.compose(red, persons, supersample=args.supersample)  # ← 마네킹 빙의

        if args.single >= 0:
            if n >= args.single:
                p = os.path.join(outdir, f"avatar_{base}_{n}.png")
                out.save(p)
                print(f"[avatar] single frame -> {p}  ({len(persons)} persons)")
                return
            n += 1
            continue

        writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))
        if n % 3 == 0:                                   # GIF ~8fps
            thumb = out.copy()
            thumb.thumbnail((560, 315))
            gif_frames.append(thumb)

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 40 == 0:
            print(f"   ...{n}/{total} ({n/(time.time()-t0):.1f} fps)")

    cap.release()
    if writer:
        writer.release()
    if rec_persons and args.single < 0 and not args.no_cache:
        np.savez_compressed(cache_path, persons=np.array(rec_persons, dtype=object))
        print(f"[avatar] kp cache saved: {cache_path} ({len(rec_persons)}f)")
    if gif_frames:
        gif = encode_gif(gif_frames, fps=8)
        with open(out_gif, "wb") as f:
            f.write(gif)
    print("=" * 60)
    print(f" AVATAR COMPOSITE — {base}")
    print(f" frames     : {n}  ({n/(time.time()-t0):.1f} fps end-to-end)")
    print(f" mp4        : {out_mp4}")
    print(f" final GIF  : {out_gif}  ({len(gif_frames)} frames)")
    print("=" * 60)


if __name__ == "__main__":
    main()
