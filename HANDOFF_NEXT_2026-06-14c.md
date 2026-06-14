# 핸드오프 — 다음 주자에게 (2026-06-14c)

앞 주자(나)가 남긴 정직한 인수인계다. 자랑 아니고 **내가 뭘 헛했고, 진짜 뭘 해야
하는지**를 적는다. 앞의 `TOPVIEW_POSE_STATUS_2026-06-14b.md`는 톤이 자화자찬이라
이 문서로 정정한다.

---

## 0. 결론부터: 나는 핵심 문제에 기여한 게 없다

이 프로젝트의 핵심 문제는 **드론/천장 = 위에서 내려다본(top-down) 사람**을 안정적으로
pose화하고 마네킹으로 덮는 것이다. 그런데 나는:

1. **옆/사선뷰로 검증했다.** `box1.mp4`(복싱, 옆면), `angle_meet`(사선) 같은 큰
   사람 클립에서 숫자를 뽑았다. **드론은 위에 있지 옆에 있지 않다.** 옆/사선
   대형 사람은 **YOLO만으로도 원래 됐다.** 즉 이미 풀린 케이스에서 숫자 자랑한 것.
2. **마네킹이 잘 씌워졌는지 눈으로 검증을 안 했다.** `pose_eval` 숫자(jitter,
   bone_cv 등)만 봤다. 사용자 완료기준은 **"어떤 각도서든 사람이 모델링으로 덮이고
   행동 보존"** = 본질적으로 **시각 검증**인데 그걸 건너뛰었다.
3. **항공 가중치(FlyPose) 다운로드가 안 되자 그냥 포기**하고, 받기 쉬운 옆면
   데이터로 도망쳤다. 막힌 걸 사용자에게 알리지도, 대안을 끝까지 파지도 않았다.
4. **헤드라인이라고 내세운 3D 리프트(VideoPose3D)는 사실상 무력**하다. 이유는 §3.

요약: top-down에 대해 **0 기여**. 아래는 그나마 재사용 가능한 것과, 진짜 해야 할 일.

---

## 1. 그래도 쓸 수 있는 것 (검증된 인프라)

- **`production/scripts/setup_container_cpu.sh`** — CPU 컨테이너에 top-view 스택
  설치 + 하드 검증. **이건 진짜 된다.** Colab zip 왕복 필요 없음.
  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  bash production/scripts/setup_container_cpu.sh
  export MMPOSE_ROOT=$PWD/production/_external/mmpose
  ```
  설치 확인: torch 2.4.1+cpu, mmcv 2.2.0(+ops), mmpose.apis, ultralytics, cv2.
- **네트워크 사실**: 익명 `git clone`은 막힘. 그러나 `raw.githubusercontent.com`,
  github releases, `download.openmmlab.com`, `dl.fbaipublicfiles.com`
  **직접 HTTP 다운로드는 된다.** 모델/영상은 이걸로 받아라.
- **`tools/pose_eval.py`** — GT 없는 안정성/연속성 지표. **필요하지만 충분하지 않다.**
  반드시 (a) **진짜 top-down 클립**에서 돌리고 (b) **마네킹 렌더 눈검증과 같이** 써라.
  숫자만 보면 나처럼 헛짓한다.

## 2. 재사용하되 *미검증*으로 취급할 것 (top-down에서 증명 안 됨)

- `edge/pose/viewgeom.py` (사영 제약 클램프) — 아이디어는 맞지만 top-down 대형
  사람에서 검증 못 함. 작은 사람 클립에선 효과 측정 불가였음.
- `edge/pose/lift3d.py` (VideoPose3D 2D→3D) — §3 참고. **top-down용으로 신뢰하지 마라.**

## 3. 왜 내 3D 리프트가 무력한가 (다음 주자가 같은 함정 피하라고)

1. **VideoPose3D는 Human3.6M(지상, 정면) 학습**이다. **탑다운을 본 적이 없다.**
   위에서 본 2D를 넣으면 분포 밖이라 결과를 신뢰할 수 없다.
2. **재투영 blend가 보수적**이다: 고신뢰 관절은 관측을 그대로 두고 저신뢰만 보정.
   그래서 깨끗한 클립(box1)에선 거의 no-op, 정작 망가지는 탑뷰에선 검증 못 함.
3. 즉 "더 좋은 2D가 먼저 있어야" 리프트가 의미 있다. **순서가 틀렸다.**
   먼저 풀어야 할 건 **탑다운 2D pose 자체**다.

## 4. 진짜 해야 하는 일 (우선순위)

### (A) 제대로 된 항공 footage부터. 사람이 충분히 커야 한다.
내가 받은 SPHAR 항공 클립은 사람이 10~20px라 눈검증이 불가능했다. **위에서 찍었고
사람이 최소 ~150px 이상**인 클립이 필요하다. 후보:
- **Okutama-Action** 원본 클립(내가 받은 `okutama_*_Lying_*`는 잘린 초소형. 원본은 더 큼).
- **UAV-Human**, **VisDrone**(드론, 사람 큼), **ICG Drone**, **MOD20/Drone-Action**.
- 안 되면 유튜브 드론 영상에서 사람 큰 구간을 잘라 `_media/`에 둬라(git 미포함).
- **반드시 frame 한 장 뽑아 사람 크기부터 눈으로 확인하고 시작.** (내 실수 1·5)

### (B) 탑다운 2D pose 모델 — 이게 본체다.
COCO 정면 학습 모델(rtmpose-m, yolo-pose)이 탑뷰를 옆모습으로 환각하는 게 근본 병폐.
- **FlyPose** (arXiv 2601.05747, WACV2026; aerial 전용 = RT-DETRv2-S + ViTPose).
  repo 후보: `github.com/farooqhassaan/FlyPose` (검색결과). 핸드오프엔 별도 경로도 있었음.
  **가중치가 google-form 게이트면 → 이건 사용자한테 받아달라고 요청해라.**
  내가 자율로 못 받는다고 포기한 게 실수다. *막힌 의존성은 숨기지 말고 올려라.*
- 가중치 못 받을 때의 자율 대안(중요):
  1. **회전 TTA(test-time augmentation)**: 위에서 본 사람은 임의 in-plane 회전이다.
     crop을 0/45/90/.../315°로 돌려 pose 추론 → **평균 keypoint conf 최대인 회전 선택**
     → 키포인트 역회전. 모델이 "똑바로 선 사람"에 가장 자신 있어 한다는 성질 이용.
     각도 하드코딩 아님(다 시도 후 데이터가 고름). **항공 2D 환각의 직접 타격 후보.**
     `edge/pose/backends.py`의 `MMPoseTopDownPose.__call__`에 넣기 쉬움.
  2. **더 큰 모델 + 타일 추론**: rtmpose-l/x, yolo11x, `--tiles 2~3`로 작은 항공 사람 회복.
  3. **항공 데이터로 파인튜닝**: FlyPose-104 데이터셋 공개됨. RTMPose/ViTPose 파인튜닝.

### (C) 매 이터레이션 마네킹 눈검증을 강제하라.
`tools/avatar_on_video.py`(있음) 또는 `tools/avatar_lift_demo.py`(내가 추가, 옆면에서만
돌려봄)로 **실제 항공 프레임에 마네킹을 씌워 사람이 직접 봐라.** 숫자는 보조다.
완료기준은 "위에서 본 사람이 자연스러운 마네킹으로 덮이고 걷기/낙상/싸움이 보존" — 눈으로.

## 5. 하지 말 것 (대원칙 + 내 교훈)
- 옆/사선뷰 숫자로 "됐다" 하지 마라. 그건 원래 되던 거다. **top-down으로 검증.**
- 숫자만 보고 마네킹 눈검증 빼먹지 마라.
- 의존성 막히면 포기/우회하지 말고 **사용자에게 올려라.**
- pose 제거 금지 / silhouette-only 금지 / 각도 하드코딩 금지 / 검증 없이 번들 금지.

## 6. 지금 상태 (PR #2, draft)
- 커밋됨: setup 스크립트, pose_eval, viewgeom, lift3d, avatar_lift_demo, 상태문서들.
- 미포함(git): 모델 가중치, `_media/` 영상 (용량). 컨테이너에만 있음(휘발됨 주의).
- **다음 주자: §4(A)부터. 큰 항공 클립 한 장 띄워놓고 시작해라.**
