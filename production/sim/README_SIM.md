# 🌆 GRADIS 디지털 트윈 — Webots + ArduCopter SITL

> "야매 아님. 펌웨어는 진짜, 판단도 진짜, 도시만 가짜."

> 🆕 **본게임은 `gazebo/` 폴더** — M4TD 실스펙 모델 + DJI Dock 3 + SITL 물리
> 락스텝 결합 + 야간 엣지케이스 도시. 프로덕션 파이프라인 무변경. RTX/Ubuntu 실행.
> 이 Webots판은 저사양 노트북용 경량 데모로 유지. → `gazebo/README_GAZEBO.md`

도시 위를 **진짜 비행 펌웨어(ArduCopter SITL)**로 나는 드론이, **진짜 YOLO detector + MMPose**로
스켈레톤을 뽑고, **진짜 VLM(ollama)**이 redact된 화면을 보고 판단해서, **진짜
관제 서버**로 에스컬레이트하는 폐루프 시뮬레이션. 코드 어디에도 시뮬 전용
분기가 없다 — 실전 스택 그대로 입력만 시뮬레이터에서 온다.

---

## 1. 왜 Webots인가 (플랫폼 비교)

| 플랫폼 | 비주얼 | 사람 액터 | SITL 연동 | 요구 사양 | 판정 |
|--------|--------|----------|-----------|----------|------|
| UE5 + Colosseum(구 AirSim) | 포토리얼 | Mixamo | PX4/AP | RTX 8GB+ | 사양 미달 ❌ |
| NVIDIA Isaac Sim + Pegasus | 포토리얼 | 있음 | PX4 | RTX 필수 | 사양 미달 ❌ |
| Gazebo Harmonic (WSL2) | 낮음 | 빈약 | 공식 | 낮음 | 그래픽/사람 한계 △ |
| **Webots R2025a** | 중간 | 관절 구동 가능 | 가능 | **MX150에서 돌아감** | **채택 ✅** |

이 노트북(i5-8250U / 8GB / MX150 2GB) 기준의 최적해다. 나중에 RTX 데스크탑이
생기면 §6의 UE5/Colosseum 업그레이드 경로를 타면 된다 — **GRADIS 스택은 카메라
URL만 바꾸면 그대로 붙는다.** 그게 이 아키텍처의 포인트.

## 2. 아키텍처 — 누가 뭘 책임지나

```
┌─ ArduCopter SITL (sim/sitl/ArduCopter.elf) ── 진짜 펌웨어 ──────────┐
│  비행역학·EKF·모드·페일세이프. 실기 FC와 동일 코드베이스.              │
│  tcp:5760 ←→ mavlink_companion.py (우리 펌웨어 레이어: 순찰/지오펜스/도크)│
│  tcp:5762 ←→ Webots 브리지(아래)                                     │
└──────────────────────────────────────────────────────────────────┘
┌─ Webots (sim/worlds/gradis_city.wbt) ── 세상 렌더링 ────────────────┐
│  gradis_drone.py(Supervisor): SITL 텔레메트리 → 드론 노드 포즈 미러링   │
│                               드론 카메라 → MJPEG http://:8881        │
│  fight_human.py: 휴머노이드 관절 구동 (몸싸움→낙상→미동없음 연기)        │
└──────────────────────────────────────────────────────────────────┘
┌─ GRADIS 실전 스택 (시뮬 분기 0줄) ──────────────────────────────────┐
│  edge.agent --source rtsp --url http://127.0.0.1:8881/cam.mjpg      │
│    → YOLO detector + MMPose(GRADIS-25) → redact → ollama VLM 판단   │
│    → ESCALATE → core/server.py → 대시보드                            │
│    → nav_command → companion → SITL → 드론 기동 → Webots 화면 반영    │
└──────────────────────────────────────────────────────────────────┘
```

움직임 경로는 임의 스크립트가 아니다: **VLM이 HOLD/ASCEND/YAW를 결정하면
Core를 거쳐 companion이 MAVLink GUIDED로 실제 펌웨어에 명령**하고, 그 결과가
화면에 보인다. 실기와 단 하나 다른 점: 모터가 픽셀이라는 것.

## 3. 설치 (이미 끝났으면 §4로)

1. **Webots R2025a** — `C:\Program Files\Webots` (설치 완료됨)
2. **ArduCopter SITL** — `sim/sitl/` (다운로드 완료: ArduCopter.elf + cygwin DLL
   + copter.parm). 출처: firmware.ardupilot.org (Mission Planner SITL 빌드)
3. **ollama 모델** — `ollama pull qwen2.5vl:3b` (설치 확인됨)
4. **Python 의존성** — `pip install pymavlink pillow opencv-python ultralytics openmim` 후 `scripts/colab_topview_pose_setup.sh` 참고

## 4. 실행

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
.\sim\run_sim.ps1                       # 전부 자동 기동
.\sim\run_sim.ps1 -VlmModel moondream   # 더 가벼운 VLM
```

- 관제 대시보드: http://127.0.0.1:8088/
- 드론 카메라(브라우저로 직접 확인 가능): http://127.0.0.1:8881/cam.mjpg
- 시나리오: (15,12) 지점에서 두 명이 몸싸움 → **t=40s** 한 명이 쓰러져 미동 없음
  → 120초 주기 반복. (-12,-9)의 행인은 대조군 — 얘 보고 알람 울리면 오탐이다.

타임라인 기대치: SITL EKF 수렴(~40s) → companion 이륙(GUIDED, 25m) → 사각형
순찰 → 카메라가 광장 통과 시 VLM이 fight/medical 판정 → 대시보드에 사건 + 클립.

## 5. 검증 포인트 (디지털 트윈의 '진짜' 체크리스트)

- [ ] SITL 콘솔에 GUIDED 전환/ARM/TAKEOFF 로그 (companion이 보낸 진짜 명령)
- [ ] Webots에서 드론이 순찰 사각형을 비행 (SITL 포즈 미러)
- [ ] agent 콘솔에 `[VLM ...ms] sit=fight risk=...` (진짜 ollama 호출)
- [ ] 대시보드에 ESCALATE + 클립 GIF 도착
- [ ] 배터리 시뮬 저하 → companion이 RETURN-TO-DOCK (도크 = 월드 원점 노란 패드)
- [ ] 지오펜스: 운영자가 300m 밖 GOTO 명령 → companion이 클램프 (로그 확인)

## 6. 업그레이드 경로 (사양/시간이 생기면)

| 항목 | 현재 (v1) | 업그레이드 |
|------|-----------|-----------|
| 드론 비주얼 | 파라메트릭 시네훕 (치수 실측) | Sketchfab/Fab에서 FPV 시네훕 glTF 다운 → `Defender25.proto`의 Shape를 `CadShape`로 교체 |
| 사람 모션 | 관절 모터 안무 (fight_human.py) | Webots `Skin` 노드 + CMU mocap BVH(무료, 복싱/낙상 클립) 또는 Mixamo FBX→BVH 변환 |
| 도시 | 박스 빌딩 + 광장 + 나무 | OSM 실지형: `osm2webots`로 용산 일대 추출, 또는 Webots `projects/vehicles/worlds/city.wbt` 에셋 재활용 |
| 비행역학 | SITL 표준 쿼드 | SITL JSON 커스텀 프레임으로 Defender 25 질량(0.24kg)/추력 모델링 |
| 물리 결합 | 포즈 미러(키네마틱) | ArduPilot 공식 `webots-python` 모델로 Webots 물리 ↔ SITL 완전 결합 |
| 렌더러 | Webots | RTX 확보 시: UE5 + Colosseum + City Sample — agent는 카메라 URL만 교체 |

## 7. 트러블슈팅

- **드론이 안 뜸**: SITL 콘솔에서 `ekf` 수렴 전 ARM 거부일 수 있음.
  companion 창에서 재시작하거나 40초 더 기다릴 것. (`gradis.parm`이
  ARMING_CHECK=0이지만 EKF origin은 필요)
- **카메라 스트림 빈 화면**: Webots 창에서 시뮬레이션이 일시정지 상태인지 확인
  (▶ 버튼). `--mode=realtime`으로 시작해야 함.
- **VLM이 사람을 못 알아봄**: v1 휴머노이드는 캡슐 림이라 YOLO 검출률이
  실영상보다 낮다. 카메라 고도를 낮추거나(`takeoff_alt_m` 15), §6의 Skin+BVH
  업그레이드로 해결.
- **포트 충돌**: 5760/5762(SITL), 8088(Core), 8881(MJPEG), 11434(ollama).
- **프레임률 낮음**: MX150이다. Webots 그래픽 설정에서 그림자 OFF,
  `--mode=fast`는 쓰지 말 것(렌더 스킵돼서 카메라가 안 나옴).

---

*"실펌웨어로 날고, 실모델로 보고, 실모델로 판단한다. 도시만 빌려왔다."* 🦅
