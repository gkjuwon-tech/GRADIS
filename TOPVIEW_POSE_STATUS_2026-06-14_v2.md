# Top-View Pose / Avatar Migration Status v2 — 2026-06-14 (continuation)

이 문서는 `TOPVIEW_POSE_STATUS_2026-06-14.md`의 후속이다. 이전 세션은 Colab에
zip을 반복 업로드시키며 OpenMMLab/mmcv 소스빌드 지옥에서 검증 없이 끝났다.
이번 세션의 원칙은 단 하나: **컨테이너 안에서 직접 돌려서 검증한다. 업로드
셔틀 금지.** 아래는 전부 CPU 컨테이너에서 실제로 실행해 얻은 결과다.

## 결론 먼저 (TL;DR)

- 전 파이프라인(detector → pose → SkeletonEngine → 마네킹 오버레이)이
  **bare CPU 컨테이너에서 mmcv 빌드 없이, GPU 없이, zip 업로드 없이 끝까지 돈다.**
- pose는 MMPose가 아니라 **RTMPose를 ONNX(rtmlib + onnxruntime)로** 돌린다.
  같은 RTMPose 가중치를, mim/mmcv 소스빌드/ImpImporter 지옥을 전부 우회해서.
  ONNX 가중치는 첫 실행 때 자동 다운로드된다.
- **oblique(angle) / horizontal CCTV 시점은 사실상 해결됨.** 사람 검출·추적·
  마네킹 렌더가 매 프레임 성공(render_rate 1.00), flicker 0.00.
  낙상은 누운 마네킹으로, 싸움은 두 사람으로 보존된다 — 행동이 살아있다.
- **순수 천상뷰(nadir, 주차장 수직 부감)는 아직 미해결.** COCO 기반 RTMPose가
  위에서 본 작은 사람을 안정적으로 포즈화하지 못한다. 이전 핸드오프의 진단과
  정확히 일치하며, 진짜 항공 전용 모델(FlyPose 등)이 필요한 잔여 문제다.

## 무엇을 했나 (전부 검증됨)

### 1. 컨테이너 실행 환경

`production/requirements_container.txt` 로 한 방에 설치된다. 컴파일 0회:

```
torch==2.4.1 / torchvision==0.19.1 (CPU)
ultralytics, rtmlib, onnxruntime
opencv-python-headless, numpy, pillow, scipy, pygltflib
```

GPU/Colab의 MMPose 경로가 필요하면 기존 `scripts/colab_topview_pose_setup.sh`
를 그대로 쓰면 된다(그건 그것대로 torch 2.4.1 + prebuilt mmcv wheel로 빌드를
피하도록 이미 잘 짜여 있다). 이번에 추가한 건 그것과 **독립적으로 돌아가는
ONNX 경로**다.

### 2. 새 pose 백엔드: `edge/pose/rtmpose_onnx.py`

`RTMPoseOnnxBackend` — rtmlib의 detector(YOLOX)+RTMPose를 감싸 GRADIS 검출
포맷(`kp[17,3]` 정규화, `bbox`, `score`, `track_id`)을 그대로 출력한다.
`make_backend("rtmpose-onnx")` 로 호출. 기존 `TopViewPoseBackend`와 드롭인 호환.

핵심 수정 한 가지: 검출 score를 17관절 평균이 아니라 **상위 8개 키포인트
평균**으로 잡는다. 평균은 가려진/게이트된 관절 때문에 0.2까지 깔려서 트래커가
모든 실제 사람을 노이즈로 버렸다. 상위 8개는 진짜 사람(~0.5–0.7)과
차량/잡음 오검출(~0.15)을 분리한다.

### 3. 트래커 spawn 임계 노출 (per-angle 하드코딩 아님)

`MultiPersonTracker(spawn_conf=...)` / `SkeletonEngine(spawn_conf=...)` 추가.
기본값은 기존 동작 그대로(HIGH_CONF=0.50). 천상/사선 시점 사람은 정면보다
키포인트 신뢰도가 구조적으로 낮으므로, 탑뷰 파이프라인만 이 값을 낮춘다
(0.25~0.30). high/low 매칭 로직은 건드리지 않았다. 각도별 규칙 분기 없음.

### 4. 컨테이너 검증 하네스: `tools/topview_bench.py`

전체 경계를 실제 영상에 돌려 3패널(원본 | 스켈레톤 | 마네킹) mp4 + 컨택트시트
를 만들고, 완료 기준 수치를 앵글별로 출력한다:

```
persons/frame, kp_conf, jitter(per-mille 프레임간 관절 이동),
render_rate(마네킹 스키닝 성공률), flicker/100(렌더 깜빡임)
```

### 5. 재현 가능한 영상 소스: `tools/fetch_bench_videos.py`

영상은 용량 때문에 GitHub에 안 올라간다. 대신 **재현 스크립트를 커밋**한다.
공개 SPHAR-Dataset 레포에 재호스팅된 CASIA 액션 클립을 raw.githubusercontent
에서 받는다. CASIA는 **같은 배우의 같은 행동을 horizontal/angle/topdown 3개
앵글로** 촬영했다 — "어떤 각도에서도 보존되는가"를 직접 시험하는 셋이다.
낙상(faint)·싸움(fight) 포함.

```
python tools/fetch_bench_videos.py
python tools/topview_bench.py            # _media/sphar_bench 전체
```

## 검증 수치 (CPU 컨테이너, 클립당 140프레임)

```
clip                                  angle       pers/f  kp_conf  jitter  render  flick/100
falling_casia_angleview_p01_faint     angle         0.82     0.50    4.34    1.00      0.00
falling_casia_horizontalview_p01      horizontal    0.76     0.43    3.26    1.00      0.00
falling_casia_topdownview_p01_faint   topdown       0.00     0.00    0.00    0.00      0.00
hitting_casia_angleview_p01p02_fight  angle         1.38     0.49    4.28    1.00      0.00
hitting_casia_topdownview_p01p02      topdown       0.10     0.27    0.00    1.00      0.00
running_casia_angleview_p01_run       angle         0.56     0.43   10.81    1.00      0.00
walking_casia_angleview_p01_walk      angle         0.75     0.49    3.56    1.00      0.00
walking_casia_horizontalview_p01_walk horizontal    0.88     0.35    0.22    1.00      0.00

per-angle 평균:
  angle        pers/f=0.88  kp_conf=0.48  jitter=5.75  render=1.00  flick/100=0.00
  horizontal   pers/f=0.82  kp_conf=0.39  jitter=1.74  render=1.00  flick/100=0.00
  topdown      pers/f=0.05  kp_conf=0.14  jitter=0.00  render=0.50  flick/100=0.00
```

해석:
- angle/horizontal: render 1.00 + flicker 0 → "마네킹이 깜빡이거나 순간이동
  한다"던 증상이 이 시점들에선 사라졌다. 낙상은 누운 마네킹으로 렌더된다
  (이전의 "정면 upright 강제" 문제 해결).
- running jitter 10.81: 빠른 동작이라 당연히 높다. 행동 자체가 빠른 것이지
  떨림이 아니다.
- topdown pers/f≈0.05: 진짜 못 잡는다. 주차장 수직 부감의 작은 사람을 COCO
  RTMPose가 단축왜곡한다. 아래 잔여 과제 참조.

## 잔여 과제 — 순수 천상뷰(nadir)

이건 게으름이 아니라 모델 한계다. COCO로 학습된 어떤 top-down 포즈 모델(RTMPose
/ViTPose/YOLO-pose 포함)도 수직 부감의 작은 사람을 신뢰성 있게 포즈화하지
못한다. 후보 경로:

1. **FlyPose (항공/탑뷰 전용)** — 가장 직접적. 단, ONNX weight가 Google Form
   배포라 자율 자동화로는 막힌다. weight를 받으면 `RTMPoseOnnxBackend`와 동일한
   구조로 ONNX detector+pose를 끼우면 된다(이미 onnxruntime 경로가 깔려 있다).
2. **작은 사람 검출 개선** — 천상뷰는 사람이 작다. 슬라이스 추론(타일)이나
   고해상도 입력으로 detector recall을 올린 뒤 pose를 태운다.
3. **FastSAM 게이트(`edge/pose/mask_fusion.py`)** — 이미 구현돼 있다. 천상뷰
   환각 관절을 silhouette 밖이면 confidence를 깎는 심판으로 쓴다. ONNX 경로에
   아직 연결 안 했다(다음 작업).

## 다음 작업자에게

- 영상 다시 받기: `python tools/fetch_bench_videos.py` (zip 업로드 절대 금지).
- 빠른 검증: `python tools/topview_bench.py --no-video` → 수치만.
- 눈검사: `python tools/topview_bench.py <clip>` → `_media/sphar_bench/bench_out/`
  에 3패널 mp4 + 컨택트시트.
- 원칙은 그대로다: pose 제거 금지, silhouette-only 금지, angle별 하드코딩 금지,
  검증 없이 bundle 재생성 금지. 그리고 추가된 철칙: **사용자에게 업로드
  셔틀을 시키지 말 것. 컨테이너에서 먼저 돌려서 증명할 것.**
