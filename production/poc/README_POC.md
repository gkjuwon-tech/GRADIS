# GRADIS PoC - DJI Mini 2 + Litchi

> 한 줄 요약: Mini 2는 Litchi가 날리고, GRADIS는 Litchi 라이브 영상을 받아 사람/상황을 판단하고 관제 이벤트와 미션 산출물을 만든다.

## 1. 아키텍처

| 레이어 | Mini 2 + Litchi PoC | Gazebo/SITL | 양산 M4TD + Dock 3 |
|---|---|---|---|
| 관제/감사 | `core/server.py` | 동일 | 동일 |
| 영상 AI | `edge/agent.py` | 동일 | 동일 |
| VLM 툴 의미론 | `edge/dronetools.py` | 동일 | 동일 |
| 비행 실행 | Litchi Android 앱 | MAVLink companion | DJI Cloud API companion |
| 영상 입력 | Litchi RTMP -> mediamtx -> RTSP | sim camera RTSP | DJI livestream RTMP/RTSP |
| PoC 브리지 | `poc/litchi_companion.py` | `firmware/companion/mavlink_companion.py` | `firmware/m4td/m4td_companion.py` |

핵심은 "자동 조종 흉내"가 아니라 책임 분리다. Litchi는 Mini 2를 실제로 날리는 앱이고, GRADIS는 영상 판단, 이벤트 저장, 알림, 운영자에게 줄 다음 행동 산출물을 담당한다. 이 구조는 Mini 2에서 직접 SDK 명령을 쏘지 않아도 PoC의 진짜 질문인 "현장 영상에서 위험을 빨리 보고 관제에 띄우는가"를 검증한다.

## 2. 중요한 현실 제약

- Litchi for DJI Drones는 Mini 2를 지원한다.
- Mini 2 계열의 Litchi waypoint는 기체에 완전 업로드되는 엔터프라이즈 미션처럼 취급하면 안 된다. 조종기/앱 링크가 계속 살아 있어야 한다.
- GRADIS는 Litchi를 로컬 API로 직접 조종하지 않는다. `litchi_companion.py`는 Core 명령을 받아 Litchi CSV와 운영자 카드로 바꾼다.
- Mini 2에는 열화상, 스포트라이트, 스피커가 없다. 해당 툴 호출은 관제 감사 로그와 양산 페이로드 의미론 검증용으로만 남긴다.
- 비행은 반드시 현지 법규, 장소 허가, 조종자 시야, RTH 고도, 배터리 여유를 만족해야 한다.

## 3. 준비물

- DJI Mini 2
- RC-N1 조종기
- Android 폰 또는 태블릿
- Litchi for DJI Drones 앱
- Windows 노트북
- 선택: USB 케이블 + `scrcpy`로 Android 화면 미러링
- 선택: RunPod GPU pod

## 4. 로컬 실행

노트북에서:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
.\poc\run_poc.ps1
```

스크립트가 마지막에 Litchi RTMP URL을 출력한다.

```text
rtmp://<laptop-lan-ip>:1935/gradis-mini2
```

Litchi Android 앱에서 라이브 스트림 공유 버튼을 열고 위 URL을 Custom RTMP 주소로 넣는다. 영상이 들어오면 GRADIS 대시보드에서 fleet, incident, nav command가 움직인다.

대시보드:

```text
http://127.0.0.1:8088/
```

생성된 Litchi CSV/운영자 카드:

```text
production\poc\outbox
```

## 5. RunPod 실행

RunPod에 코드 복사:

```powershell
cd C:\Users\wonma\Documents\GRADIS
scp -P <ssh-port> -r production root@<pod-ip>:/workspace/gradis
ssh -p <ssh-port> root@<pod-ip> "cd /workspace/gradis/poc/runpod && bash setup_runpod.sh && bash start_server.sh"
```

노트북 게이트웨이 실행:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
.\poc\relay_laptop.ps1 -Pod root@<pod-ip> -Port <ssh-port>
```

Litchi에는 게이트웨이 스크립트가 출력하는 URL을 넣는다.

```text
rtmp://<laptop-lan-ip>:1935/gradis-mini2
```

이때 영상은 `Android Litchi -> 노트북 1935 -> SSH tunnel -> RunPod mediamtx -> edge.agent` 순서로 흐른다. Core 대시보드는 노트북의 `http://127.0.0.1:8088/`로 본다.

## 6. 데모 시나리오

| # | 장면 | 합격 기준 |
|---|---|---|
| P1 | Mini 2가 Litchi waypoint 임무로 안전 고도 순찰 | GRADIS fleet에 `GRADIS-MINI2-LITCHI-POC` 표시 |
| P2 | 정상 보행자/자전거/행인을 통과 관찰 | 오탐 없이 로그만 유지 |
| P3 | 배우가 매트 위에서 쓰러져 5초 정지 | Fall/Medical incident가 3초 내 대시보드에 표시 |
| P4 | 두 배우가 밀침/몸싸움 연기 | Fight/Assault incident 표시, clip 저장 |
| P5 | 관제자가 GOTO 명령 발행 | `poc/outbox`에 GOTO CSV + operator card 생성 |
| P6 | 관제자가 ORBIT 명령 발행 | 6점 orbit CSV 생성 |
| P7 | 배터리 낮음 또는 링크 품질 저하를 조종자가 선언 | Litchi/DJI RTH로 회수, GRADIS는 사건/명령 로그 유지 |

## 7. Litchi CSV 운용법

`poc/litchi_companion.py`는 시작할 때 기본 순찰 CSV를 만든다. 이후 Core에서 `GOTO`, `ROUTE`, `ORBIT`, `RETURN_TO_DOCK` 명령이 들어오면 새 CSV와 운영자 카드를 만든다.

CSV는 waypoint 행만 담는다. Litchi 전역 설정은 가져온 뒤 사람이 확인해야 한다.

- Drone model: Mini 2
- Finish action: RTH 또는 None 중 데모 목적에 맞게 선택
- Cruising speed: 3 m/s 전후
- RTH altitude: 주변 장애물보다 충분히 높게
- Altitude mode: 이륙 지점 기준인지 AGL인지 현장에 맞게 확인
- Signal loss: RTH

## 8. 운영자 체크리스트

- 장소 허가와 비행 가능 공역 확인
- 조종자 시야 확보
- 사람 위 직접 비행 금지
- 프로펠러/배터리/RTH 고도 확인
- Litchi에 표시되는 홈포인트 확인
- RTMP 영상이 GRADIS 대시보드까지 들어오는지 이륙 전 확인
- `poc/outbox`의 CSV는 바로 실행하지 말고 반드시 지도에서 검토

## 9. 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| 대시보드에 영상 이벤트가 없다 | Litchi RTMP URL 오타 또는 방화벽 | Windows 방화벽에서 1935 허용, URL 재입력 |
| RunPod 모드에서 영상이 안 들어온다 | SSH 터널이 닫힘 | `relay_laptop.ps1` 재실행 |
| `edge.agent`가 RTSP를 못 연다 | mediamtx가 아직 스트림을 못 받음 | Litchi에서 스트리밍 시작 후 5초 대기 |
| CSV가 너무 가까운 곳으로 잘린다 | geofence clamp 작동 | `config/litchi_mini2_poc.json`의 `geofence_radius_m` 확인 |
| Mini 2가 임무 중 멈춘다 | RC 링크 불안정 | 즉시 RTH/수동 회수, 다음 시연은 짧은 경로로 축소 |

## 10. 외부 참고

- Litchi 공식 도움말: Mini 2 호환 목록과 앱 사용법.
- Litchi 포럼: Mini 2 waypoint는 연속 RC 링크가 필요한 방식이라는 운영상 주의.
- Litchi CSV 포맷 문서: CSV는 waypoint 파라미터를 담고, 전역 미션 설정은 수동 확인이 필요하다.
