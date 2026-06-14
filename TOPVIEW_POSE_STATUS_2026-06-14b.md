# Top-View Pose — Container Engine Work, 2026-06-14 (이어서)

이전 핸드오프(`TOPVIEW_POSE_STATUS_2026-06-14.md`)를 이어받아, **Colab zip 왕복을
버리고 컨테이너에서 직접** 환경을 세우고 핵심 엔진을 고도화했다.

## 0. 한 줄 요약

- 환경: CPU 컨테이너에 top-view pose 스택 **완전 설치 + 하드 검증 통과**.
- 영상: GitHub raw에서 **날것 CCTV/항공 영상 확보**(CASIA/CAVIAR/MEVA/Okutama/boxing).
- 측정: GT 없는 **시점-강건 정량 하네스**(`tools/pose_eval.py`) 신설 — 이전 작업의
  최대 공백(검증 수단 부재)을 메움.
- 엔진: 2개 업그레이드 통합·측정.
  1. **사영 제약 클램프**(`edge/pose/viewgeom.py`, 온라인)
  2. **2D→3D 리프팅 정제**(`edge/pose/lift3d.py`, VideoPose3D, 오프라인)
- 핵심 발견: 3D 리프트는 **2D가 쓸만할 때 강체 3D를 만든다**(boxing에서 3D 뼈
  변동계수 0.013). 그러나 **탑다운 초소형 항공 사람의 근본 문제는 탐지+2D pose
  자체**이고, 그건 항공-학습 모델(FlyPose 등)이 있어야 풀린다.

## 1. 환경 (재현 가능)

```bash
python3 -m venv .venv && . .venv/bin/activate
bash production/scripts/setup_container_cpu.sh   # 멱등, 마지막에 검증 게이트
export MMPOSE_ROOT=$PWD/production/_external/mmpose
```

`setup_container_cpu.sh` = Colab 스크립트의 CPU/py3.11 포팅. 핵심: torch 2.4.1+cpu로
고정 → mmcv 2.2.0을 **prebuilt wheel**로 설치(소스빌드/ mim 도박 제거), numpy<2 ABI
잠금. 검증 통과 항목: torch/torchvision/mmengine/mmcv(+ops)/mmpose.apis/ultralytics/
cv2/onnxruntime, RTMPose config 디스크 존재.

설치 확인된 버전: torch 2.4.1+cpu, mmcv 2.2.0, mmpose 1.3.2, ultralytics 8.4.67,
cv2 4.10.0, numpy 1.26.4. **GPU 없음(CPU 온리), ~3 fps.**

가중치(전부 직접 다운로드, git 미포함):
- YOLO11 detector: ultralytics가 자동 다운로드.
- RTMPose-m: openmmlab에서 자동 다운로드.
- VideoPose3D: `_external/weights/videopose3d_h36m_coco.bin`
  (`dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin`).

> 네트워크: 익명 `git clone`은 막혀 있으나 `raw.githubusercontent.com`,
> github releases, `download.openmmlab.com`, `dl.fbaipublicfiles.com` **직접 다운로드는 됨**.

## 2. 영상 (GitHub raw, `_media/sphar/`, git 미포함)

출처: **SPHAR-Dataset**(surveillance-perspective action recognition, MP4로 정리됨).
`raw.githubusercontent.com/AlexanderMelde/SPHAR-Dataset/master/videos/<cat>/<file>`.

확보분과 특성(yolo11m로 측정한 최대 사람 키 비율):
- `box1.mp4` 320x240 — boxing 2인, **maxH 0.74**(큰 사람, 빠른 격투=행동검증용)
- `angle_meet.mp4` 320x240 — 사선 CCTV 2인, maxH 0.33
- `angle_walk/horiz_walk/top_walk/top_fall/angle_fall` — CASIA/항공 주차장, **사람 작음(0.1~0.2)**
- `drone_lying.mp4` 100x186 — Okutama 실제 드론
- `caviar_chair/caviar_fall.mp4` 384x288 — CAVIAR 쇼핑몰 CCTV

**한계**: GitHub raw에 "큰 사람 + 진짜 천장 탑다운" 클립이 희귀. 사선/정면 대형
사람(box1, caviar) 또는 탑다운 소형 사람(CASIA/Okutama)으로 양분됨.

## 3. 측정 하네스 — `tools/pose_eval.py`

GT 없이, 트랙 수명 동안의 **자기일관성**을 측정(시점 무관). 사용자 불만 → 지표:
- 순간이동/슬라이딩 → `jitter`, `accel`
- 프레임마다 다른 포즈 → `bone_cv`(투영 뼈길이 변동계수)
- 깜빡임 → `flicker`, `id_churn`
- 탑뷰 포즈화 실패 → `anat`, `detect_rate`
- **`bone_cv3d`**: 3D 리프트 후 3D 뼈길이 변동계수 = **진짜 시점 불변 일관성 지표**

```bash
python tools/pose_eval.py _media/sphar/box1.mp4 --model yolo11m.pt \
  --pose-preset rtmpose-m --imgsz 640 --lift --out r.json
python tools/pose_eval.py x --compare a.json b.json   # 비교표
```

### 측정된 베이스라인 (rtmpose-m, imgsz640, 120f)
| clip | detect/fr | track_len | bone_cv(2D) | bone_cv3d | anat |
|------|----:|----:|----:|----:|----:|
| top_walk (소형 항공) | 0.23 | 28 | **0.48** | – | 0.77 |
| angle_meet (소형 사선) | 1.45 | 87 | 0.22 | 0.25(노이즈) | 0.96 |
| box1 (대형) | – | 79 | – | **0.013** | – |

→ bone_cv3d가 box1 0.013 vs angle_meet 0.25. **3D 일관성은 입력 2D 크기/품질에
직결.** 큰 사람이면 리프트가 강체 3D를 안정적으로 만든다(합성 클린=0.005에 근접).

## 4. 엔진 업그레이드

### 4.1 사영 제약 클램프 — `edge/pose/viewgeom.py` (온라인, tracker 통합)
원리: **투영된 뼈는 실제 3D 뼈보다 길어질 수 없다.** 트랙별로 (뼈길이/bbox대각)의
최근 윈도 **80분위**를 실제길이 근사로 보고, 그걸 크게 넘는(물리 불가) 사지를 distal
관절을 끌어당겨 클램프. **max가 아니라 분위수**인 게 핵심(환각이 최댓값이라 max는 영영
안 걸림). 각도 하드코딩 없음. 미래 프레임 불필요(실시간용).

### 4.2 2D→3D 리프팅 정제 — `edge/pose/lift3d.py` (오프라인)
아키텍처:
1. 트랙 2D COCO-17 시퀀스 수집 →
2. **사람 중심 정규화**(root 중심화 + 트랙 고정 스케일; H36M 학습분포 정렬) →
3. **VideoPose3D TemporalModel**(RF=243) → H36M-17 3D, 뼈길이 일정 →
4. 프레임별 **약투영 카메라 affine을 관측 2D에 최소제곱 피팅** →
5. 3D 전체 재투영 → 보정 2D (시간일관 + 해부유효 + 폐색복원 + 그 프레임 실제 시점 반영).

각도 하드코딩 0 — 카메라가 데이터로 추정됨. `pose_eval --lift`로 측정 통합.

## 5. 정직한 결론과 다음 방향

- **잘 되는 것**: 사람이 충분히 크면 사선/정면 어떤 각도든 안정적 스켈레톤·마네킹.
  리프트가 강체 3D를 줌 → 3D-구동 마네킹의 토대 마련.
- **여전히 안 되는 것**: 탑다운 **초소형** 항공 사람. 병목은 (a) 탐지(작은 사람),
  (b) 2D pose 환각. 리프트는 **2D가 좋아진 뒤에야** 효과 → 근치 아님.
- **VideoPose3D는 H36M(지상 정면) 학습**이라 탑다운을 본 적 없음. 핸드오프가
  FlyPose(항공 전용)를 지목한 이유와 정확히 일치.

### 다음 루프 후보 (우선순위)
1. **탑다운 in-plane 회전 보정 추론**: 위에서 본 사람은 임의 방향 = 임의 회전.
   crop을 주방향으로 세워 pose 추론 후 키포인트 역회전. 각도가 아니라 *그 사람의
   방향*을 데이터로 추정 → 일반적. 항공 2D 환각의 직접 타격 후보.
2. 더 큰 2D 모델(rtmpose-l/x) + 타일 추론으로 소형 항공 탐지/포즈 회복.
3. **3D-네이티브 마네킹 리타깃**: 리프트 3D로 리그를 직접 구동(2D FK보다 안정).
4. FlyPose 가중치 확보 시 `backends.py`에 항공 2D 백엔드로 편입(드롭인).

### 대원칙(유지)
pose 제거 금지 / silhouette-only 금지 / 각도 하드코딩 금지 / 검증 없이 번들 금지 /
무엇이든 `pose_eval`로 베이스라인 대비 수치 증명 후 반영.
