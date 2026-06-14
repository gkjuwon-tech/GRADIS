#!/usr/bin/env python3
"""
실제 영상에서 스켈레톤 추출 — 진짜 사람 모션으로 엔진 검증.

내가 만든 합성 데이터 말고, 무료 라이선스 '연출된' 격투 스톡영상(배우/안무)에
YOLO person detector/tracker + MMPose + SkeletonEngine(추적/폐색복원/평활)을 돌린다.

출력:
  - 오버레이 영상(_media/out_*.mp4): 원본 위에 스켈레톤. 복원된 관절은 빨강.
  - 샘플 GIF + 프레임 PNG
  - 콘솔 통계: 인원/프레임, 평균 신뢰도, 복원 관절 비율, 추적 ID 수, 처리 FPS

(판단은 VLM의 몫 — 이 도구는 '추출 품질'만 측정한다. v2 아키텍처.)

사용:
  python tools/extract_on_video.py _media/fight_45874.mp4 --model yolo11x.pt --pose-preset rtmpose-m --imgsz 1280 --conf 0.25 --tiles 1
  python tools/extract_on_video.py _media/two_4605.mp4 --max-frames 200
"""
import sys, os, time, argparse
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.pose import SkeletonEngine, make_backend
from edge.pose.skeleton import N_KPT
from edge.render import EDGES, encode_gif

YELLOW = (60, 200, 245)   # BGR
WHITE = (235, 235, 235)
RED = (60, 60, 255)       # 복원된(추정) 관절


def draw(frame_bgr, people):
    H, W = frame_bgr.shape[:2]
    for p in people:
        pts = [(int(x * W), int(y * H), c) for (x, y, c) in p.kp]
        for a, b in EDGES:
            if p.kp[a, 2] > 0.2 and p.kp[b, 2] > 0.2:
                cv2.line(frame_bgr, pts[a][:2], pts[b][:2], YELLOW, 2, cv2.LINE_AA)
        for j, (x, y, c) in enumerate(pts):
            if c > 0.2:
                col = RED if p.inferred[j] else WHITE
                cv2.circle(frame_bgr, (x, y), 3, col, -1, cv2.LINE_AA)
        # ID + 품질
        xs = [q[0] for q in pts if q[2] > 0.2]
        ys = [q[1] for q in pts if q[2] > 0.2]
        if xs:
            cv2.putText(frame_bgr, f"ID{p.id} q{p.quality:.2f}",
                        (min(xs), max(0, min(ys) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    return frame_bgr


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
    ap.add_argument("--aerial", action="store_true", help="탑다운(얼굴 가중↓). 스톡영상은 off")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--tiles", type=int, default=1,
                    help=">1이면 NxN 멀티스케일 타일(원거리 소형 인물, 배치 추론)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"못 엶: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    backend = make_backend("topview",
                           detector_model=args.detector_model,
                           pose_preset=args.pose_preset,
                           pose_config=args.pose_config,
                           pose_checkpoint=args.pose_checkpoint,
                           imgsz=args.imgsz,
                           conf=args.conf,
                           tiles=args.tiles,
                           tracker=args.tracker,
                           half=False)
    engine = SkeletonEngine(aerial=True if args.aerial else False, default_fps=fps)

    base = os.path.splitext(os.path.basename(args.video))[0]
    outdir = os.path.dirname(args.video)
    out_path = os.path.join(outdir, f"out_{base}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"[extract] {args.video}  {W}x{H} {fps:.0f}fps {total}f  "
          f"detector={args.detector_model}@{args.imgsz} pose={args.pose_preset}")
    n = 0
    persons_per = []
    conf_acc = []
    inferred_acc = []
    quality_acc = []
    ids_seen = set()
    gif_frames = []
    t_proc0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        raw = backend(frame)                  # 원시 추출(익명화: 좌표만)
        people = engine.process(raw, t)       # 추적/복원/평활

        persons_per.append(len(people))
        for p in people:
            ids_seen.add(p.id)
            vis = p.kp[:, 2] > 0.2
            if vis.any():
                conf_acc.append(float(p.kp[vis, 2].mean()))
            inferred_acc.append(float(p.inferred.mean()))
            quality_acc.append(p.quality)

        vis_frame = draw(frame, people)
        writer.write(vis_frame)
        if 0.3 * total <= n <= 0.3 * total + 70:   # 중간 70프레임 GIF 샘플
            rgb = cv2.cvtColor(cv2.resize(vis_frame, (W // 2, H // 2)), cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(rgb))
        if n in (int(total * 0.3), int(total * 0.5), int(total * 0.7)):
            cv2.imwrite(os.path.join(outdir, f"sample_{base}_{n}.png"), vis_frame)

        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if n % 40 == 0:
            print(f"   ...{n}/{total}  ({n/(time.time()-t_proc0):.1f} proc-fps)")

    cap.release(); writer.release()
    if gif_frames:
        gif = encode_gif(gif_frames, fps=10)
        with open(os.path.join(outdir, f"sample_{base}.gif"), "wb") as f:
            f.write(gif)

    dur = time.time() - t_proc0
    def mm(a): return float(np.mean(a)) if a else float("nan")
    print("=" * 64)
    print(f" EXTRACTION REPORT — {base}  (REAL staged-fight footage)")
    print("-" * 64)
    print(f" frames processed   : {n}   ({n/dur:.1f} proc-fps on CPU)")
    print(f" persons/frame (avg) : {mm(persons_per):.2f}")
    print(f" mean kpt confidence : {mm(conf_acc):.3f}")
    print(f" mean pose quality   : {mm(quality_acc):.3f}")
    print(f" occlusion-recovered : {mm(inferred_acc)*100:.1f}% of joints (filled by engine)")
    print(f" unique track IDs    : {len(ids_seen)}  (ideal ~= real #people)")
    print(f" overlay video       : {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
