# 🏙️ GRADIS 디지털 트윈 v2 — Gazebo Harmonic + ArduPilot SITL

> "두뇌는 진짜, 도시는 가짜, 비행은 대역배우." — M4TD 자율스택(비공개)은
> ArduPilot SITL이 대역하고, AI 판단·툴·게이트는 프로덕션 파일 그대로 돈다.

**빌드 완료 / 실행 대기** — RTX 3060 + Ubuntu에서 실행 예정.
(Webots 버전(`sim/`)은 MX150 노트북용 경량판. 이쪽이 본게임.)

## 1. 왜 이 구성이 '진짜' 디지털 트윈인가

| 레이어 | 시뮬 | 실기체 (M4TD + Dock 3) | 동일? |
|---|---|---|---|
| AI 파이프라인 | edge.agent (YOLO detector→MMPose→아바타→VLM 툴) | 〃 | ✅ 같은 파일 |
| 관제 | core/server.py | 〃 | ✅ 같은 파일 |
| 명령 게이트 | nav_commands (감사/지오펜스) | 〃 | ✅ 같은 파일 |
| 툴 의미론 | dronetools.py (autopilot/goto/thermal...) | 〃 → DJI Cloud API | ✅ 같은 액션 |
| 자율비행 | ArduCopter SITL + mavlink_companion (대역) | DJI 자율 스택 + Dock 3 | 대역 — 의미론 동일 |
| 기체 물리 | m4td SDF (1.85kg, 실측 기하) | 실물 | 스펙 일치 |
| 도크 | dji_dock3 모델 (패드/리드/조명) | DJI Dock 3 | 시각 트윈 |

**무작위 주행이 아니다**: companion이 `autopilot on` 순찰 미션을 돌리고,
VLM이 카메라에서 이상을 보면 `autopilot off → goto → thermal/zoom` 툴 콜로
개입한다 — 프로덕션과 동일한 의사결정 루프.

## 2. 구성물

```
sim/gazebo/
├── worlds/gradis_city_night.sdf  ★기본★ 야간 엣지케이스 도시 (아래 §2.1)
├── worlds/gradis_city.sdf      주간 기본 도시 (디버그/튜닝용 — WORLD=gradis_city)
├── models/m4td/                M4TD 디지털 트윈 (1.85kg 실스펙 기하 + 짐벌
│                                EO/IR 카메라 + ArduPilotPlugin/LiftDrag 결선)
├── models/dji_dock3/           DJI Dock 3 트윈 (개방 리드/내부 패드/LED/기상마스트)
├── scripts/camera_bridge.py    gz 카메라 토픽 → MJPEG :8881 (agent 무변경 구독)
├── scripts/fetch_assets.sh     액터 애니메이션 DAE 다운로드 (1회)
├── scripts/run_sim.sh          원클릭 기동 (Core→GZ→SITL→bridge→agent→companion)
└── sitl/m4td.parm              1.85kg 기준 SITL 파라미터
```

### 2.1 야간 엣지케이스 월드 (gradis_city_night — 기본)

환경: 달빛 + 가로등 9개의 '빛 웅덩이'만 있는 어두운 도시. 창문 불빛 빌딩 9동
(순찰존 안쪽에 골목을 만드는 2동 포함), 주차 차량 10대, 나무 4그루, 군중.

| ID | 위치 | 시나리오 | 기대 동작 |
|----|------|----------|-----------|
| E1 | (28, 28) | **골목 격투** — 빌딩 사이 3.5m, 가로등 없음 | 차폐+저조도. goto/orbit으로 각 확보, night/thermal 툴 |
| E2 | (5, 3) | **군중 밀집 8인** — 4×4m 클러스터 | CROWD(압사 위험) 판정 |
| E3 | (18, −2) | **나무 아래 쓰러진 사람** — 캐노피 차폐 | 수직 시야 한계 — 사각 진입 시 감지 |
| E4 | (12, −14.6) | **차 뒤의 사람** — 하반신 가림 | 부분 검출에서도 추적 유지 |
| E5 | 광장 외곽 순환 | **조거** (고속 러닝) | '추격' 오탐 금지 |
| E6 | (0, 14) | **포옹 커플** (정지 근접 2인) | '몸싸움' 오탐 금지 |
| E7 | (−5, 10) | **벤치 착석** | '쓰러짐' 오탐 금지 |
| E8 | 4개 경로 | **교차 보행자** | 전부 무알람 |
| — | (12, 9) | 개활지 쓰러짐 (가로등 옆, 대조용) | 기본 DOWN 감지 |

오탐 함정(E5/E6/E7)과 진짜 사건(E1/E2/E3/—)이 섞여 있다 — **정밀도와 재현율을
한 월드에서 동시에** 측정한다. 주간 월드(gradis_city)는 같은 지형의 단순판.

## 3. 설치 (Ubuntu 22.04/24.04, 1회)

```bash
# 1) Gazebo Harmonic
sudo apt install gz-harmonic python3-gz-transport13 python3-gz-msgs10

# 2) ArduPilot (SITL)
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot ~/ardupilot
cd ~/ardupilot && Tools/environment_install/install-prereqs-ubuntu.sh -y

# 3) ardupilot_gazebo 플러그인 (ArduPilotPlugin)
sudo apt install libgz-sim8-dev rapidjson-dev
git clone https://github.com/ArduPilot/ardupilot_gazebo ~/ardupilot_gazebo
cd ~/ardupilot_gazebo && mkdir build && cd build && cmake .. && make -j4

# 4) GRADIS 의존성 + VLM
pip3 install ultralytics pillow opencv-python pymavlink pygltflib openmim
bash scripts/colab_topview_pose_setup.sh
ollama pull qwen2.5vl:3b

# 5) 액터 애니메이션
cd <GRADIS>/production/sim/gazebo/scripts && ./fetch_assets.sh
```

## 4. 실행

```bash
cd production/sim/gazebo/scripts
./run_sim.sh                      # 전부 자동
# 환경변수: ARDUPILOT=~/ardupilot  ARDUPILOT_GAZEBO=~/ardupilot_gazebo/build
#          VLM_MODEL=qwen2.5vl:3b
```

- 대시보드 http://127.0.0.1:8088/ · 카메라 http://127.0.0.1:8881/cam.mjpg

## 5. 검증 체크리스트 (폐루프의 '진짜' 증명)

- [ ] Gazebo에서 M4TD가 Dock 3 패드에서 이륙 (SITL ARM/TAKEOFF 로그)
- [ ] companion `PATROL` 상태로 사각형 순찰 (autopilot on)
- [ ] 광장 통과 시 agent 콘솔: `[VLM ...] sit=fight risk=...`
- [ ] **VLM 툴 콜**: `[TOOL] autopilot -> AUTOPILOT_OFF` → `goto -> GOTO`
      → Gazebo에서 기체가 실제로 광장으로 기수 전환 ← 이게 백미
- [ ] 쓰러진 사람 발견 → `thermal -> CAM_THERMAL_ON` (PayloadBridge 로그)
- [ ] 대시보드에 ESCALATE + 클립, 행인은 무반응(오탐 0)
- [ ] 상황 종료 → `autopilot -> AUTOPILOT_ON` 순찰 복귀

## 6. 첫 실행 튜닝 노트 (정직한 예고)

1. **호버 추력**: LiftDrag 계수는 1차 추정. 이륙이 안 되거나 튀면
   `model.sdf`의 `<area>`(±30%)와 `m4td.parm`의 `MOT_THST_HOVER`를 조정.
   SITL 로그의 학습된 hover throttle을 다시 parm에 반영.
2. **카메라 토픽명**: `gz topic -l | grep camera`로 실제 토픽 확인 후
   `camera_bridge.py --topic ...` (월드/모델명 바뀌면 경로 바뀜).
3. **플러그인 로드 실패**: `GZ_SIM_SYSTEM_PLUGIN_PATH`가 ardupilot_gazebo
   build 디렉토리를 가리키는지 확인.
4. **액터 격투 퀄리티**: 기본은 run.dae(고에너지) + 궤적 안무. Mixamo 격투
   FBX→DAE로 교체하면 진짜 복싱 (assets/animations/README.md).

## 7. Webots판과의 관계

| | Webots (`sim/`) | **Gazebo (`sim/gazebo/`)** |
|---|---|---|
| 용도 | 노트북(MX150) 데모/개발 | RTX 본검증 |
| 물리 | 포즈 미러(키네마틱) | **SITL↔물리 락스텝 결합** |
| 기체 | Defender25 시각 모델 | **M4TD 실스펙 모델 + DJI Dock 3** |
| 사람 | 관절 모터 안무 | 스킨 애니메이션 액터 |

*"같은 펌웨어, 같은 에이전트, 같은 관제 — 다른 건 세상이 픽셀이라는 것뿐."* 🦅
