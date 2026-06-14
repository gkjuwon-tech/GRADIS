#!/usr/bin/env python3
"""
GRADIS Edge Agent — 드론 한 대의 두뇌(인지 파트).

아키텍처 (v4 — 신뢰경계):
  로컬(Jetson Orin)의 역할은 익명 스켈레톤 추출이다.
  YOLO는 사람 후보 탐지/트래킹만 담당하고, 포즈는 MMPose RTMPose/ViTPose가 담당한다.
  생영상(raw)은 기기 밖으로 절대 안 나가고 추출 후 즉시 버린다.
  스켈레톤 → 마네킹(아바타) 렌더가 '기기를 떠나는 유일한 표상'(얼굴·몸·옷 0)이고,
  그 익명 마네킹 프레임만 클라우드 Gemini Flash Lite로 보내 항법+상황을 판단한다.
  마네킹이 곧 방화벽 — 그래서 판단이 클라우드여도 프라이버시가 안 깨진다.

루프:
  프레임 획득 → 탐지/트래킹 → MMPose 포즈 추출(익명화) → 링버퍼 push(원본, RAM only)
     → 마네킹 프레임을 판단 워커에 비차단 제출(최신 프레임만, 밀리면 드롭)
  판단 워커(Gemini):
     ├ navigation : PROCEED/HOLD/ASCEND/... → Core nav_commands로 전달
     └ situation  : 위험 판정 → 링버퍼에서 원본 클립 구출 → 코어로 전송
  + 주기적 하트비트(배터리/위치)

소스:
  --source synthetic         카메라 없이 합성 스켈레톤으로 전체 파이프라인 실행(검증/데모)
  --source webcam            로컬 웹캠 (cv2) — '드론 카메라' 스탠드인
  --source rtsp --url ...    실드론 RTSP 스트림   (예: rtsp://192.168.1.10:8554/cam)
  --source file --url a.mp4  녹화 파일

실행 (키는 .env의 GEMINI_API_KEY에서 자동 로드):
  python -m edge.agent --source synthetic --avatar
  python -m edge.agent --source rtsp --url rtsp://... --core http://core:8088 --avatar
"""
import argparse, time, random, sys, threading
import queue as _q

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .buffer import RingBuffer
from .render import draw_skeleton, encode_gif
from .uplink import Uplink
from .privacy import redact_people
from .cognition import Cognition
from .dronetools import DroneToolbox

BANNER = r"""
   __ _ _ __ __ _  __| (_)___
  / _` | '__/ _` |/ _` | / __|   GRADIS edge agent
 | (_| | | | (_| | (_| | \__ \   skeletons on python. judgment on VLM.
  \__, |_|  \__,_|\__,_|_|___/
   |___/
"""

VLM_KIND = {"fall": "Fall / Collapse", "fight": "Fight", "medical": "Medical Emergency",
            "fall_risk": "Fall Risk (Ledge)", "crowd": "Crowd Crush Risk",
            "other": "Suspicious Activity"}


def iter_synthetic(fps):
    from . import synthetic
    for t, frame, persons in synthetic.frames(fps=fps):
        yield t, frame, persons, True  # realtime-paced by caller


class _LatestFrameGrabber:
    """캡처 전용 스레드 — 항상 '가장 최신' 프레임 1장만 들고 있는다.

    cv2.VideoCapture는 내부 큐에 프레임을 쌓는다. 추론이 카메라 fps보다 느리면
    오래된 프레임을 순서대로 처리하게 되어 레이턴시가 무한히 누적된다(드론이
    3초 전 세상을 보고 판단하는 참사). 그래서 읽는 족족 덮어쓰고, 소비자는
    항상 최신 프레임만 가져간다. 밀린 프레임은 그냥 버린다 — 실시간이 정의다.
    """

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._stop.set()
                break
            with self._lock:
                self._frame = frame
                self._seq += 1

    def latest(self, last_seq):
        """last_seq 이후의 새 프레임이 있으면 (seq, frame), 없으면 (last_seq, None)."""
        with self._lock:
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._frame

    @property
    def alive(self):
        return not self._stop.is_set()

    def stop(self):
        self._stop.set()


def iter_camera(source, url, fps, detector_model, pose_preset, pose_config,
                pose_checkpoint, imgsz, conf, tiles, tracker):
    """webcam/rtsp/file 공통. cv2로 읽고 top-view pose pipeline으로 스켈레톤 추출.

    webcam/rtsp는 그래버 스레드로 최신 프레임만 처리(레이턴시 누적 차단).
    file은 모든 프레임 순서대로(오프라인 분석이니 빠짐없이).
    """
    import cv2
    from PIL import Image
    from .pose import PoseEstimator

    cap = cv2.VideoCapture(0 if source == "webcam" else url)
    if not cap.isOpened():
        print(f"[agent] 소스 못 엶: {source} {url}")
        sys.exit(2)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 드라이버 큐도 최소로
    except Exception:
        pass

    pose = PoseEstimator(backend="topview",
                         detector_model=detector_model,
                         pose_preset=pose_preset,
                         pose_config=pose_config,
                         pose_checkpoint=pose_checkpoint,
                         imgsz=imgsz,
                         conf=conf,
                         tiles=tiles,
                         tracker=tracker)
    t0 = time.time()

    if source == "file":
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t = time.time() - t0
            persons = pose(frame_bgr, t)                   # ← 여기서 익명화
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield t, Image.fromarray(rgb), persons, False
        cap.release()
        return

    grab = _LatestFrameGrabber(cap)
    seq = 0
    try:
        while grab.alive:
            seq, frame_bgr = grab.latest(seq)
            if frame_bgr is None:
                time.sleep(0.002)
                continue
            t = time.time() - t0
            persons = pose(frame_bgr, t)                   # ← 여기서 익명화
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield t, Image.fromarray(rgb), persons, False
    finally:
        grab.stop()
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "webcam", "rtsp", "file"])
    ap.add_argument("--url", default=None, help="rtsp/file 소스 경로")
    ap.add_argument("--core", default="http://127.0.0.1:8088")
    ap.add_argument("--drone-id", default="GRADIS-01")
    ap.add_argument("--zone", default="Riverside Park Area 2")
    ap.add_argument("--lat", type=float, default=37.5283)
    ap.add_argument("--lon", type=float, default=126.9650)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--buffer-secs", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true", help="synthetic 무한 반복")
    # --- 스켈레톤 추출 엔진 ---
    ap.add_argument("--detector-model", "--model", dest="detector_model",
                    default="yolo11x.pt",
                    help="YOLO detector weights. Do not use *-pose.pt weights here.")
    ap.add_argument("--pose-preset", default="rtmpose-m",
                    choices=["rtmpose-m", "vitpose-s"],
                    help="MMPose preset. Start with rtmpose-m; use vitpose-s for quality checks.")
    ap.add_argument("--pose-config", default=None,
                    help="MMPose config path. Overrides --pose-preset when paired with --pose-checkpoint.")
    ap.add_argument("--pose-checkpoint", default=None,
                    help="MMPose checkpoint path or URL. Overrides --pose-preset when paired with --pose-config.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="person detector confidence threshold")
    ap.add_argument("--tiles", type=int, default=1,
                    help=">1이면 NxN 멀티스케일 타일(원거리 소형 인물, 배치 추론)")
    ap.add_argument("--tracker", default="botsort.yaml",
                    help="Ultralytics tracker config, e.g. botsort.yaml or bytetrack.yaml")
    # --- 대뇌(VLM) — 유일한 판단 주체 ---
    ap.add_argument("--vlm", default="gemini",
                    choices=["none", "gemini"],
                    help="판단 비전 모델 (gemini=클라우드 Flash Lite, 익명 마네킹만 업로드; "
                         "none=판단 끄고 반사신경만)")
    ap.add_argument("--vlm-model", default=None,
                    help="미지정 시 .env의 GEMINI_MODEL 또는 gemini-3.1-flash-lite")
    ap.add_argument("--vlm-host", default="http://localhost:11434")
    ap.add_argument("--vlm-api-key", default=None,
                    help="gemini는 미지정 시 GEMINI_API_KEY/GOOGLE_API_KEY 환경변수 사용")
    ap.add_argument("--vlm-interval", type=float, default=1.0,
                    help="VLM 추론 최소 주기(초). 모델이 빠르면 사실상 연속 판단.")
    ap.add_argument("--vlm-risk", type=float, default=0.6,
                    help="이 위험도 이상이면 VLM이 사건 에스컬레이트")
    ap.add_argument("--avatar", action="store_true",
                    help="redact 위에 회색 마네킹(리깅 빙의) 합성 — VLM 이해도 ↑")
    ap.add_argument("--avatar-glb", default=None,
                    help="리깅 GLB 경로 (기본: assets3d/RiggedFigure.glb)")
    ap.add_argument("--escalate-cooldown", type=float, default=10.0,
                    help="같은 종류 사건 재보고 최소 간격(초) — 알림 폭주 방지")
    args = ap.parse_args()

    # gemini 백엔드 기본값 자동 해석 (.env에서 키/모델 로드 — 코드·CLI에 안 박음)
    if args.vlm == "gemini":
        import os
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        if args.vlm_host == "http://localhost:11434":
            args.vlm_host = "https://generativelanguage.googleapis.com"
        if not args.vlm_model:
            args.vlm_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        if not args.vlm_api_key:
            args.vlm_api_key = (os.environ.get("GEMINI_API_KEY")
                                or os.environ.get("GOOGLE_API_KEY"))

    print(BANNER)
    print(f"[agent] {args.drone_id}  source={args.source}  core={args.core}")
    print(f"[agent] zone='{args.zone}'  buffer={args.buffer_secs}s  fps={args.fps}")

    buf = RingBuffer(seconds=args.buffer_secs, fps=args.fps)
    avatar = None
    if args.avatar:
        import os
        from .avatar import MannequinRenderer
        glb = args.avatar_glb or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets3d", "Xbot.glb")
        avatar = MannequinRenderer(glb)
        print(f"[agent] avatar layer: {glb}")
    up = Uplink(args.core, args.drone_id)
    toolbox = DroneToolbox(up)            # M4TD 내장 기능 = VLM의 손발
    cog = Cognition(backend=args.vlm, model=args.vlm_model,
                    host=args.vlm_host, api_key=args.vlm_api_key,
                    toolbox=toolbox)
    print(f"[agent] cognition: backend={args.vlm} model={args.vlm_model} "
          f"available={cog.available}")
    if not cog.available:
        print("[agent] !! 판단 모델 없음 — 스켈레톤 추출/하트비트만 동작, 상황 판단 불가.")
        print("[agent] !!   (.env에 GEMINI_API_KEY 채우면 됨)")
    else:
        print("[agent] VLM warmup (모델 RAM 상주)...")
        cog.warmup()

    battery = 100.0
    last_hb = 0.0
    last_vlm = 0.0
    nav_state = "PATROL"
    last_nav_sent = {"t": 0.0, "action": None}
    escalate_cool = {}            # kind -> 다음 허용 시각(time.time())
    dt = 1.0 / args.fps
    n_incidents = 0

    def run_once():
        nonlocal battery, last_hb, last_vlm, nav_state, n_incidents
        if args.source == "synthetic":
            stream = iter_synthetic(args.fps)
        else:
            stream = iter_camera(args.source, args.url, args.fps,
                                 args.detector_model, args.pose_preset,
                                 args.pose_config, args.pose_checkpoint,
                                 args.imgsz, args.conf, args.tiles,
                                 args.tracker)

        def escalate(kind, risk, conf, ids, t, persons_snap, reason=""):
            """링버퍼에서 원본 구출 → GIF 인코딩+전송은 백그라운드(루프 비차단).

            사건은 드물고(쿨다운) 무겁다(인코딩+업링크 수백 ms). 메인 루프가
            그걸 기다리면 그 시간 동안 눈을 감는 셈이라, 구출만 동기로 하고
            (버퍼가 덮어쓰기 전에!) 나머지는 스레드로 넘긴다.
            """
            nonlocal n_incidents
            now = time.time()
            if now < escalate_cool.get(kind, 0.0):
                return
            escalate_cool[kind] = now + args.escalate_cooldown
            n_incidents += 1
            clip_frames = buf.rescue(now=t, pre=3.0, post=0.0)   # 동기: 원본 확보가 우선

            def _ship():
                annotated = [draw_skeleton(f, persons_snap) for f in clip_frames[-30:]] \
                    if args.source == "synthetic" else clip_frames[-30:]
                gif = encode_gif(annotated, fps=10)

                class _V:  # uplink가 기대하는 형태
                    pass
                v = _V()
                v.kind, v.risk, v.confidence = kind, risk, conf
                v.t_utc = now
                skel = [{"id": int(i)} for i in ids]
                up.send_incident(v, (args.lat + random.uniform(-3e-4, 3e-4),
                                     args.lon + random.uniform(-3e-4, 3e-4)),
                                 args.zone, clip_bytes=gif, skeletons=skel)
                print(f"  >> ESCALATE [VLM] {kind:20s} risk={risk:.2f} ids={ids} "
                      f"clip={len(annotated)}f {reason}")

            threading.Thread(target=_ship, daemon=True).start()

        # --- VLM 워커 스레드 ---------------------------------------------------
        # VLM은 CPU에서 수백 ms ~ 수 초 걸린다. 절대 캡처/추출 루프를 막으면 안 된다.
        # 별도 스레드에서 '최신 프레임 1장'만 처리하고, 나머진 버린다(드롭).
        job_q = _q.Queue(maxsize=1)
        stop_evt = threading.Event()

        def vlm_worker():
            nonlocal nav_state
            while not stop_evt.is_set():
                try:
                    frame_j, persons_j, t_j = job_q.get(timeout=0.4)
                except _q.Empty:
                    continue
                view = redact_people(frame_j, persons_j)     # ← 프라이버시 redact
                if avatar is not None:
                    view = avatar.compose(view, persons_j)   # ← 마네킹 빙의(형상 번역)
                res = cog.infer(view)
                if not res:
                    continue
                nav, sit = res["navigation"], res["situation"]
                nav_state = nav["action"] + (" obstacle" if nav["obstacle_ahead"] else "")
                if nav["action"] != "PROCEED" or nav["obstacle_ahead"]:
                    now_nav = time.time()
                    action_key = (nav["action"], nav["obstacle_ahead"])
                    if (
                        action_key != last_nav_sent["action"]
                        or now_nav - last_nav_sent["t"] > 3.0
                    ):
                        last_nav_sent["t"] = now_nav
                        last_nav_sent["action"] = action_key
                        up.nav_command(
                            nav["action"],
                            reason=nav["reason"],
                            source="vlm-navigation",
                            ttl_s=6,
                        )
                # 툴 호출 실행 — VLM이 지휘관으로서 기체 기능을 직접 다룬다
                for tc in res.get("tools", []):
                    ok, msg = toolbox.call(tc["name"], tc["args"],
                                           reason=sit["reason"][:120])
                    print(f"  [TOOL] {msg}" if ok else f"  [TOOL] REJECT {msg}")

                print(f"  [VLM {res['_latency_ms']}ms] nav={nav['action']} "
                      f"obs={nav['obstacle_ahead']} | sit={sit['kind']} "
                      f"risk={sit['risk']:.2f} tools={[t['name'] for t in res.get('tools', [])]} "
                      f":: {sit['reason'][:80]}")
                if sit["risk"] >= args.vlm_risk and sit["kind"] != "none":
                    ids = [p for p, _ in persons_j]
                    escalate(VLM_KIND.get(sit["kind"], "Suspicious Activity"),
                             sit["risk"], 0.7, ids, t_j, persons_j,
                             reason=f":: {sit['reason'][:60]}")

        worker = None
        if cog.available:
            worker = threading.Thread(target=vlm_worker, daemon=True)
            worker.start()

        wall0 = time.time()
        try:
            for t, frame, persons, paced in stream:
                tick_start = time.time()

                # 1) 원본 프레임을 RAM 링버퍼에 (디스크엔 안 적음)
                buf.push(t, frame)

                # 2) 대뇌(VLM)에 '최신 프레임' 비차단 제출. 추출 루프는 안 막힌다.
                now = time.time()
                if cog.available and persons and (now - last_vlm) >= args.vlm_interval:
                    last_vlm = now
                    try:
                        job_q.put_nowait((frame, persons, t))
                    except _q.Full:
                        pass   # 아직 이전 추론 중 → 이번 프레임은 드롭

                # 3) 하트비트 (2초마다) — 비행상태 = VLM 항법 판단 반영
                battery = max(0.0, battery - 0.02)
                if now - last_hb > 2.0:
                    state = nav_state if battery > 25 else "RTL-LOWBATT"
                    up.heartbeat(battery, (args.lat, args.lon), state)
                    last_hb = now

                # 실시간 페이싱(합성). 카메라는 자체 fps.
                if paced:
                    sleep = dt - (time.time() - tick_start)
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            stop_evt.set()
            if worker:
                worker.join(timeout=2)
        return time.time() - wall0

    try:
        while True:
            dur = run_once()
            print(f"[agent] 스트림 종료 ({dur:.1f}s). incidents={n_incidents}")
            if not (args.source == "synthetic" and args.loop):
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[agent] 종료 신호.")
    finally:
        buf.wipe()   # 원본 전부 소멸. 복구 불가.
        print("[agent] RingBuffer wiped. 원본 영상 소멸 완료. bye.")


if __name__ == "__main__":
    main()
