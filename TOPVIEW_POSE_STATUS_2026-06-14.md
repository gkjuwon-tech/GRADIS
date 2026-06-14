# Top-View Pose / Avatar Migration Status - 2026-06-14

## 현재 상황

탑뷰/천장/드론 시점에서 기존 `YOLO*-pose` 기반 마네킹 오버레이가 치명적으로 불안정하다.

관찰된 문제:

- 사람이 위에서 보일 때 무릎을 굽힌 옆모습처럼 환각됨.
- 마네킹이 양옆으로 순간이동하거나 슬라이딩함.
- 같은 사람인데 프레임마다 다른 작은 포즈로 찍힘.
- 사람/마네킹이 깜빡거리며 나타났다 사라짐.
- 특히 드론/천장뷰에서는 “위에서 본 사람” 자체를 안정적으로 포즈화하지 못함.

핵심 진단:

- 행동 분석에는 관절 pose가 필수다. silhouette-only, upright-only, 정면 마네킹 덮어쓰기 방식은 폐기.
- 문제는 “pose를 버릴 것인가”가 아니라 “탑뷰에서 신뢰 가능한 pose 모델을 어떻게 확보하고, 환각 관절을 어떻게 검증할 것인가”다.
- 각도별 하드코딩은 금지. 20도/21도/탑뷰/사선뷰를 규칙으로 나눠 처리하는 방향은 유지보수 불가능하다.
- AI 모델을 적극 활용해야 한다.

## 되돌린/폐기한 방향

다음 방향은 부적절하다고 판단되어 폐기했다.

- `upright` / `pose-mode` 기반 탑뷰 정면 마네킹 강제
- silhouette-only avatar
- 관절을 버리고 위치/방향/속도만으로 절차적 보행을 만드는 방식

이유:

- 위에서 본 장면에 정면 사람 모델을 덮는 것은 자연스럽지 않다.
- 관절 정보가 사라지면 낙상, 싸움, 걷기, 몸짓 등 행동 분석이 불가능하다.
- 사용자 요구는 “어느 각도에서든 실제 움직임을 유지하는 pose 기반 모델 오버레이”다.

## 현재 남아 있는 실험 코드

현재 실험 방향은 다음이다.

### 1. Pose는 주 신호

파일:

- `C:\Users\wonma\Documents\GRADIS\production\edge\pose\backends.py`
- `C:\Users\wonma\Documents\GRADIS\production\edge\pose\engine.py`
- `C:\Users\wonma\Documents\GRADIS\production\tools\avatar_on_video.py`

현재 production 방향:

- YOLO는 사람 후보 탐지/트래킹 용도
- MMPose/RTMPose/ViTPose 같은 별도 pose 모델이 관절 추출
- `SkeletonEngine`이 추적, temporal smoothing, confidence 처리

단, 로컬 Python에는 아직 MMPose 계열이 설치되어 있지 않았다.

확인 결과:

- `mmpose`: 미설치
- `mmdet`: 미설치
- `mmcv`: 미설치
- `mmengine`: 미설치
- `torch`: `2.12.0+cpu`
- `ultralytics`: `8.4.62`

### 2. FastSAM/SAM은 보조 AI 심판

파일:

- `C:\Users\wonma\Documents\GRADIS\production\edge\pose\mask_fusion.py`
- `C:\Users\wonma\Documents\GRADIS\production\tools\avatar_pose_mask_fusion.py`

목적:

- pose 모델이 만든 관절을 버리지 않는다.
- 대신 FastSAM/SAM이 실제 보이는 사람 영역을 segmentation으로 잡고,
- 관절/팔다리 segment가 mask 밖으로 튀면 confidence를 낮춘다.
- 즉, segmentation은 pose 대체가 아니라 “환각 관절 검문소”다.

로컬 검증:

1. YOLO-seg 기반 person segmentation은 탑뷰에서 거의 실패했다.
   - `matched masks/frame = 0.00`
   - 결론: 일반 YOLO person-seg는 탑뷰 사람을 잘 못 잡음.

2. FastSAM bbox prompt는 1프레임 테스트에서 동작했다.
   - pose bbox를 FastSAM에 넣었고 mask 1개가 나왔다.
   - `FastSAM-s.pt` 다운로드됨.

3. `caviar_meet_crowd.mpg`의 실제 탑뷰 구간에서 FastSAM fusion 테스트:

실행 조건:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
python -X utf8 tools\avatar_pose_mask_fusion.py `
  _media\real_world_hard\caviar_meet_crowd.mpg `
  --start 178 `
  --pose-model ..\yolov8s-pose.pt `
  --mask-source fastsam `
  --imgsz 1280 `
  --pose-conf 0.08 `
  --sam-imgsz 640 `
  --sam-conf 0.10 `
  --max-frames 50
```

결과:

- `raw persons/frame`: 2.20
- `fused persons/frame`: 2.20
- `matched masks/frame`: 1.64
- `poses/frame`: 2.02
- `outside keypoints/frame`: 0.68
- `bad limbs/frame`: 1.22

출력:

- `C:\Users\wonma\Documents\GRADIS\production\_media\real_world_hard\fusion_caviar_meet_crowd_pyolov8s-pose_fastsam_FastSAM-s.mp4`
- `C:\Users\wonma\Documents\GRADIS\production\_media\real_world_hard\fusion_caviar_meet_crowd_pyolov8s-pose_fastsam_FastSAM-s.gif`
- `C:\Users\wonma\Documents\GRADIS\production\_media\real_world_hard\fusion_caviar_meet_crowd_pyolov8s-pose_fastsam_FastSAM-s_contact.png`

해석:

- FastSAM은 실제로 사람 주변 mask를 잡고, 일부 환각 limb confidence를 깎았다.
- 그러나 `yolov8s-pose` 자체가 탑뷰 관절을 옆모습처럼 잘못 만드는 한계가 여전히 크다.
- 따라서 FastSAM gate만으로는 최종 해결이 아니다.

## FlyPose 상태

FlyPose는 현재 문제와 가장 직접적으로 맞는 후보 모델이다.

이유:

- aerial/top-view human pose estimation을 직접 겨냥함.
- 구조가 detector + pose ONNX 파이프라인.
- 논문/README 기준으로 UAV-Human 등 항공 시점 문제를 다룸.

로컬 상태:

repo는 클론됨:

- `C:\Users\wonma\Documents\GRADIS\production\_external\FlyPose`

중요 파일:

- `C:\Users\wonma\Documents\GRADIS\production\_external\FlyPose\model\README.md`
- `C:\Users\wonma\Documents\GRADIS\production\_external\FlyPose\model\src\flypose\inference.py`
- `C:\Users\wonma\Documents\GRADIS\production\_external\FlyPose\model\configs\default_config.py`
- `C:\Users\wonma\Documents\GRADIS\production\_external\FlyPose\model\configs\flypose_h_config.py`

문제:

- ONNX weight 파일은 repo에 포함되어 있지 않다.
- README에 따르면 Google Form을 통해 weight를 받아야 한다.
- 필요한 예상 파일:

```text
model/checkpoints/detector/flyposeDetector.onnx
model/checkpoints/pose/flypose_s/end2end.onnx
model/checkpoints/pose/flypose_h/end2end.onnx
```

다음 작업은 FlyPose weight 확보부터 시작해야 한다.

## 앞으로 이어갈 목표

최종 목표:

> 어떤 각도에서 찍힌 영상이든 사람의 실제 pose/행동을 최대한 유지하면서 자연스러운 익명 마네킹/모델로 오버레이한다.

구체 목표:

1. FlyPose ONNX weights 확보
2. FlyPose sample image inference 로컬 실행
3. FlyPose를 `GRADIS` detection format으로 변환
   - 입력: frame
   - 출력: `kp[17,3]`, normalized bbox, score, optional track_id
4. 같은 탑뷰 영상에서 비교
   - 기존 `yolov8s-pose`
   - `yolov8s-pose + FastSAM gate`
   - `FlyPose`
   - `FlyPose + FastSAM gate`
5. 수치 비교
   - persons/frame
   - matched masks/frame
   - bad limbs/frame
   - temporal jitter
   - mannequin flicker count
   - track continuity
6. 눈검사용 contact sheet / mp4 생성
7. 결과가 확실히 개선될 때만 production pipeline과 Colab bundle에 반영

## 다음 실행 후보

FlyPose weight를 받은 뒤:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production\_external\FlyPose
pip install -r model\requirements.txt
python model\src\flypose\run_image_inference.py `
  --config model\configs\default_config.py `
  --input model\input_examples\sample2.jpg
```

그 다음 GRADIS 영상 1프레임에 직접 붙여 확인:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
python -X utf8 tools\flypose_probe.py `
  _media\real_world_hard\caviar_meet_crowd.mpg `
  --start 178 `
  --max-frames 20
```

`tools\flypose_probe.py`는 아직 작성되지 않았다. 다음 작업자가 작성해야 한다.

## 중요한 원칙

- pose는 절대 제거하지 않는다.
- silhouette-only는 금지.
- angle별 하드코딩 금지.
- AI 모델을 적극 활용한다.
- 검증 없이 Colab bundle을 다시 만들지 않는다.
- 로컬에서 최소 1프레임, 가능하면 20~50프레임 눈검사와 수치검사를 통과한 뒤 bundle 반영.
- 사용자에게 zip 업로드 반복을 시키기 전에 반드시 로컬/스크립트 레벨 검증을 먼저 끝낸다.

