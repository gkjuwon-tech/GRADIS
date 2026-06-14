# GRADIS Mini 2 + Litchi 왕초보 실행 가이드

목표는 간단하다. Mini 2는 Litchi가 날리고, Litchi가 보내는 라이브 영상을 GRADIS가 받아서 위험 상황을 관제 화면에 띄운다.

## 0. 오늘 할 일

1. 노트북에서 GRADIS 실행
2. Litchi 앱에 RTMP 주소 입력
3. Mini 2 이륙 전 영상 수신 확인
4. 안전한 짧은 Litchi waypoint 비행
5. 배우가 넘어지는 장면으로 알림 확인

## 1. 준비물

- DJI Mini 2
- RC-N1 조종기
- Android 폰
- Litchi for DJI Drones 앱
- Windows 노트북
- 같은 Wi-Fi 또는 폰 핫스팟 네트워크
- 넓은 야외 공간
- 착륙 매트와 안전요원 1명

Mini 2는 실내 장난감 드론처럼 쓰면 안 된다. 사람 위를 직접 지나가지 말고, 조종자는 항상 기체를 눈으로 봐야 한다.

## 2. 노트북 실행

PowerShell:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
.\poc\run_poc.ps1
```

마지막에 이런 주소가 나온다.

```text
rtmp://192.168.x.x:1935/gradis-mini2
```

이 주소를 그대로 Litchi에 넣을 것이다.

## 3. Litchi 영상 연결

1. Android 폰을 RC-N1에 연결한다.
2. Mini 2 전원, 조종기 전원, Litchi 앱을 켠다.
3. Litchi 카메라 화면에서 비디오 공유/라이브 스트림 버튼을 연다.
4. Custom RTMP 주소에 노트북이 출력한 주소를 붙여넣는다.
5. 스트리밍을 시작한다.
6. 노트북 브라우저에서 `http://127.0.0.1:8088/`를 연다.

영상이 정상으로 들어오면 `edge.agent` 창 로그가 움직이고, 관제 대시보드에 incident/fleet 데이터가 표시된다.

## 4. 기본 미션 CSV

실행하면 `production\poc\outbox`에 기본 순찰 CSV가 생긴다.

이 CSV를 Litchi Mission Hub 또는 Litchi 앱으로 가져온 뒤 반드시 직접 확인한다.

- waypoint가 실제 장소 위에 있는지
- 고도가 안전한지
- RTH 고도가 장애물보다 높은지
- 끝나고 RTH할지, hover할지
- 배터리 여유가 충분한지

확인 전에는 절대 실행하지 않는다.

## 5. 실제 시연 순서

1. 이륙 전 대시보드 열기: `http://127.0.0.1:8088/`
2. Litchi RTMP 스트림 시작
3. 대시보드/로그에서 영상 수신 확인
4. Litchi에서 짧은 waypoint 미션 시작
5. 배우가 카메라 시야 안에서 천천히 걷기
6. 배우가 매트 위에 쓰러지고 5초 정지
7. 대시보드에 Fall/Medical incident가 뜨는지 확인
8. 조종자는 즉시 회수 가능한 상태 유지

## 6. RunPod로 돌릴 때

RunPod 서버를 켠 뒤 노트북에서:

```powershell
cd C:\Users\wonma\Documents\GRADIS\production
.\poc\relay_laptop.ps1 -Pod root@<pod-ip> -Port <ssh-port>
```

이 스크립트도 Litchi에 넣을 RTMP 주소를 출력한다.

```text
rtmp://<laptop-lan-ip>:1935/gradis-mini2
```

RunPod 모드에서도 Litchi에는 노트북 IP를 넣는다. 노트북이 영상을 받아 SSH 터널로 RunPod에 넘긴다.

## 7. 자주 터지는 문제

| 증상 | 해결 |
|---|---|
| Litchi 스트림 시작은 됐는데 GRADIS가 조용함 | Windows 방화벽에서 1935 포트 허용 |
| RTMP 주소가 안 먹힘 | 폰과 노트북이 같은 네트워크인지 확인 |
| RunPod에서 영상이 안 뜸 | `relay_laptop.ps1` 창이 살아 있는지 확인 |
| 대시보드가 안 열림 | `core/server.py` 창 확인, `http://127.0.0.1:8088/` 사용 |
| 미션 CSV가 이상한 곳에 찍힘 | `config/litchi_mini2_poc.json`의 home 좌표를 실제 이륙지로 수정 |
| 사고 감지가 늦음 | `--vlm-model qwen2.5vl:3b`로 가볍게 실행하거나 RunPod GPU 사용 |

## 8. 꼭 기억할 것

GRADIS가 Mini 2를 몰래 자동 조종하는 구조가 아니다. Litchi가 비행하고, 조종자가 최종 책임을 지고, GRADIS는 영상 판단과 관제 자동화를 검증한다. 이게 Mini 2 + Litchi PoC의 안전하고 솔직한 버전이다.
