#!/usr/bin/env python3
"""
GRADIS v2 파이프라인 실전 테스트 — 비디오 파일로 '진짜 경로' 전체를 돌린다.

v2 아키텍처: Py 엔진은 스켈레톤만 추출(익명화), 판단은 전부 VLM.
이 스크립트는 그 경로를 그대로 재현한다:

  1) 비디오 프레임 → PoseEstimator(YOLO detector/tracker + MMPose + SkeletonEngine) → 스켈레톤
  2) privacy.redact_people() → 사람 강블러+스켈레톤 오버레이 (VLM이 보는 화면)
  3) (--vlm 지정 시) Cognition이 redact 프레임을 판단 → 상황/항법 로그
  4) 스켈레톤 오버레이 결과 비디오(.mp4) + GIF 저장

사용:
  python tools/test_video_pipeline.py --video production/_media/fight_40969.mp4
  python tools/test_video_pipeline.py --video a.mp4 --vlm ollama --vlm-model moondream
"""
import sys
import os
import time
import argparse
import cv2
import numpy as np
from PIL import Image

# GRADIS 패키지 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.pose import PoseEstimator
from edge.privacy import redact_people
from edge.cognition import Cognition
from edge.render import draw_skeleton, encode_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="production/_media/fight_40969.mp4",
                        help="테스트할 입력 비디오 파일 경로")
    parser.add_argument("--output", default=None,
                        help="출력할 결과 비디오 파일 경로 (기본값: [입력파일명]_skeleton.mp4)")
    parser.add_argument("--model", "--detector-model", dest="detector_model",
                        default="yolo11x.pt",
                        help="YOLO person detector weights, not *-pose.pt")
    parser.add_argument("--pose-preset", default="rtmpose-m",
                        choices=["rtmpose-m", "vitpose-s"])
    parser.add_argument("--pose-config", default=None)
    parser.add_argument("--pose-checkpoint", default=None)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="detector inference size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="person detector confidence threshold")
    parser.add_argument("--tiles", type=int, default=1,
                        help=">1 enables NxN tile inference for small/distant people")
    parser.add_argument("--fps", type=float, default=None,
                        help="처리할 FPS (기본값: 비디오 원본 FPS)")
    # --- VLM (판단 주체) ---
    parser.add_argument("--vlm", default="none", choices=["none", "ollama", "openai"])
    parser.add_argument("--vlm-model", default="moondream")
    parser.add_argument("--vlm-host", default="http://localhost:11434")
    parser.add_argument("--vlm-interval", type=float, default=1.0,
                        help="VLM 판단 주기(영상 시간 기준, 초)")
    parser.add_argument("--save-redacted", action="store_true",
                        help="VLM이 본 redact 프레임도 비디오로 저장")
    args = parser.parse_args()

    input_path = args.video
    if not os.path.exists(input_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"[오류] 비디오 파일을 찾을 수 없습니다: {input_path}")
            sys.exit(1)

    print(f"[START] Video file: {input_path}")
    print(f"[MODEL] detector: {args.detector_model}  pose: {args.pose_preset}  /  judge: {args.vlm}:{args.vlm_model}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[오류] 비디오 파일을 열 수 없습니다: {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = args.fps if args.fps is not None else orig_fps
    dt = 1.0 / fps

    print(f"[INFO] Video Info: {width}x{height} @ {orig_fps:.2f} FPS (total {total_frames} frames)")

    base, _ = os.path.splitext(input_path)
    output_path = args.output or f"{base}_skeleton.mp4"
    gif_path = os.path.splitext(output_path)[0] + ".gif"
    redacted_path = f"{base}_redacted.mp4"

    try:
        pose_estimator = PoseEstimator(backend="topview",
                                       detector_model=args.detector_model,
                                       pose_preset=args.pose_preset,
                                       pose_config=args.pose_config,
                                       pose_checkpoint=args.pose_checkpoint,
                                       imgsz=args.imgsz,
                                       conf=args.conf,
                                       tiles=args.tiles,
                                       tracker=args.tracker)
    except Exception as e:
        print(f"[오류] PoseEstimator 초기화 실패. 의존성이 설치되어 있는지 확인하세요.\n{e}")
        sys.exit(1)

    cog = Cognition(backend=args.vlm, model=args.vlm_model, host=args.vlm_host)
    if args.vlm != "none" and not cog.available:
        print("[경고] VLM 백엔드 응답 없음 — 추출만 수행합니다.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    red_writer = (cv2.VideoWriter(redacted_path, fourcc, fps, (width, height))
                  if args.save_redacted else None)

    print(f"[OUTPUT] Video output path: {output_path}")
    print(f"[OUTPUT] GIF output path: {gif_path}")
    print("[INFO] Analyzing frames...")

    annotated_images = []
    vlm_logs = []
    frame_idx = 0
    last_vlm_t = -1e9
    t_start = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_current = frame_idx * dt

        # 1) 스켈레톤 추출 (GRADIS-25: 0..16 COCO + 17..24 파생 관절, 추적 ID 포함)
        persons = pose_estimator(frame_bgr, t=t_current)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 2) redact — VLM이 보는 화면 (환경 그대로, 사람은 강블러+스켈레톤)
        redacted = None
        if (cog.available or args.save_redacted) and persons:
            redacted = redact_people(pil_img, persons)
            if red_writer is not None:
                red_writer.write(cv2.cvtColor(np.array(redacted), cv2.COLOR_RGB2BGR))

        # 3) VLM 판단 (영상 시간 기준 주기적)
        if cog.available and redacted is not None and \
           t_current - last_vlm_t >= args.vlm_interval:
            last_vlm_t = t_current
            res = cog.infer(redacted)
            if res:
                sit, nav = res["situation"], res["navigation"]
                msg = (f"t={t_current:6.2f}s  [{res['_latency_ms']:5d}ms] "
                       f"sit={sit['kind']:9s} risk={sit['risk']:.2f}  "
                       f"nav={nav['action']:8s} :: {sit['reason'][:70]}")
                vlm_logs.append(msg)
                print(f"[VLM] {msg}")

        # 4) 스켈레톤 오버레이 비디오
        annotated_pil = draw_skeleton(pil_img, persons)
        writer.write(cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR))

        if frame_idx % 3 == 0:
            thumb = annotated_pil.copy()
            thumb.thumbnail((480, 270))
            annotated_images.append(thumb)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            print(f"   [진행] {frame_idx}/{total_frames} 프레임 완료 ({frame_idx/elapsed:.1f} fps)")

    cap.release()
    writer.release()
    if red_writer is not None:
        red_writer.release()
        print(f"[INFO] Redacted video saved: {redacted_path}")

    if annotated_images:
        print("[INFO] Saving GIF file...")
        gif_bytes = encode_gif(annotated_images, fps=fps / 3)
        if gif_bytes:
            with open(gif_path, "wb") as f:
                f.write(gif_bytes)
            print(f"[INFO] GIF saved successfully: {gif_path}")

    t_total = time.time() - t_start
    print("=" * 60)
    print("[SUCCESS] v2 pipeline test finished (extract → redact → VLM judge)")
    print(f"[STATS] Total time elapsed: {t_total:.1f}s (avg {frame_idx/max(t_total,1e-9):.1f} FPS)")
    print(f"[STATS] VLM judgments: {len(vlm_logs)}")
    for log in vlm_logs[:10]:
        print(f"  - {log}")
    if len(vlm_logs) > 10:
        print(f"  ...외 {len(vlm_logs)-10}건 더 있음.")
    print("=" * 60)


if __name__ == "__main__":
    main()
