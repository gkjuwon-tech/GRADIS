# 🧩 GRADIS — 코드 아키텍처 문서

> "코드는 뼈대고, 우리도 뼈만 본다. 운명이다."

**문서 버전:** v0.2 (2026-06-10)
**전제:** 엣지(드론)에서 최대한 다 처리하고, 위험할 때만 클라우드로 쏜다.
**철학:** *Anonymize at the edge, decide at the edge, forget at the edge.*

> 🔄 **v0.2 아키텍처 결정 (확정):** 룰 기반 행동분석(`edge/action`, ActionDetector)을
> **제거**했다. Py 엔진의 역할은 **스켈레톤 추출(GRADIS-25) = 익명화** 단 하나.
> 위험/항법 **판단은 온디바이스 VLM이 전담**한다 — privacy-redact된 장면(환경 원본 +
> 사람은 강블러+스켈레톤)을 통째로 이해해서 판단한다. 룰이 못 푸는
> "푸쉬업 vs 낙상", "포옹 vs 몸싸움"을 장면 맥락으로 푼다.
> 스켈레톤은 GRADIS-25 (COCO-17 검출 + neck/pelvis/spine/head/hands/feet 파생 8관절).
>
> **아바타 레이어 (`edge/avatar.py`, `--avatar`):** VLM에게 점·선(졸라맨)만 주면
> 장면 이해도가 떨어진다. 그래서 추출된 관절로 리깅된 '얼굴 없는 회색 마네킹'
> (Mixamo X Bot, 오프라인 박제)을 빙의시켜 블러 위에 합성한다. 리타깃은 v4
> **풀 3D**: 본 길이 보존 깊이 복원(dz=√(L²−proj²)) + 시간 연속성 부호 잠금 +
> 본별 3D 최단호 회전. VLM은 사람 '형상'을 보고, 프라이버시는 그대로
> (마네킹은 좌표 생성물이지 원본 픽셀이 아님).
>
> **툴 컨트롤 (`edge/dronetools.py`):** AI는 조종사가 아니라 지휘관이다.
> VLM이 autopilot on/off, goto, orbit, thermal, zoom, spotlight, speaker 등을
> '툴'로 호출하면 Core 게이트(감사 가능)를 거쳐 컴패니언이 MAVLink/페이로드로
> 변환한다. 평시엔 autopilot(on)으로 기체가 알아서 순찰한다.

---

## 1. 시스템 컴포넌트 맵

```
gradis/
├── edge/                      # 드론에 올라가는 모든 것 (실시간, 목숨걸림)
│   ├── capture/               # 카메라 프레임 획득
│   ├── pose/                  # 스켈레톤 추출 (익명화의 핵심)
│   ├── action/                # 행동 분석 (위험 판정)
│   ├── buffer/                # 30초 RAM 링버퍼 (절대 디스크 X)
│   ├── decision/              # ESCALATE vs DISCARD
│   ├── uplink/                # 위험 시 암호화 전송
│   ├── flight/                # 자율비행/항법 (별도 비행컨트롤러 연동)
│   └── healthwatch/           # 배터리/온도/링크 상태 (안 죽게)
│
├── core/                      # 지상/클라우드 (GRADIS Core)
│   ├── gateway/               # 드론들의 ESCALATE 수신
│   ├── triage/                # 위험 우선순위 정렬
│   ├── dispatch/              # 경찰 상황실 연동
│   ├── audit/                 # 불변 감사 로그 (누가 뭘 봤나)
│   └── dashboard/             # 관제 UI
│
├── ml/                        # 모델 학습 (오프라인, 스켈레톤만 먹임)
│   ├── datasets/              # ⚠️ 좌표 텐서만. 얼굴 절대 금지.
│   ├── train_pose/
│   ├── train_action/
│   └── eval/
│
└── shared/                    # 프로토콜, 스키마, 암호 유틸
```

---

## 2. 엣지 파이프라인 (드론 안에서 벌어지는 일)

### 2.1 데이터 흐름 — 프레임 한 장의 생애

```
프레임 도착 (t=0ms)
   │
   ├─→ [pose] 스켈레톤 추출 ──→ keypoints (17~33 joints)
   │                              │
   │                              ▼
   │                       [action] 슬라이딩 윈도우(예: 2초)에
   │                       스켈레톤 시퀀스 쌓아서 위험점수 계산
   │
   └─→ [buffer] 원본 프레임을 RAM 링버퍼에 push
                (30초 넘으면 가장 오래된 프레임 덮어씀)

위험점수 > 임계치?
   ├─ YES → buffer.rescue(clip) → uplink.send(clip + meta) → buffer.wipe()
   └─ NO  → (아무것도 안 함. 링버퍼가 알아서 덮어씀 = 자연 소멸)
```

### 2.2 핵심 의사코드

```python
# edge/main.py — 드론의 심장
async def gradis_loop(cam, pose, action, buffer, decider, uplink):
    async for frame in cam.stream():           # 4K30, 멈추면 안 됨
        skeletons = pose.extract(frame)        # ← 여기서 얼굴 영원히 버려짐
        buffer.push(frame)                     # 원본은 RAM에서만 30초 산다

        scores = action.update(skeletons)      # temporal 위험 점수들
        verdict = decider.judge(scores)        # ESCALATE | DISCARD

        if verdict.escalate:
            clip = buffer.rescue(window=verdict.window)   # 위험 순간 구출
            await uplink.send(Incident(
                kind=verdict.kind,             # FALL-01, BRAWL-01 ...
                risk=verdict.risk,
                gps=cam.geotag(),
                clip=encrypt(clip),            # 경찰 외엔 못 봄
            ))
            buffer.wipe()                      # 보냈으면 로컬은 즉시 소멸
        # else: 아무것도 안 함. 링버퍼가 조용히 미래의 프레임으로 덮어씀.
```

> 🔑 법적 킬포인트: `pose.extract()`가 `buffer.push()`와 분리돼 있고, **학습 파이프라인은 오직 `skeletons`만** 먹는다. 원본 `frame`은 학습 데이터셋에 **절대** 안 들어간다. 이거 코드 리뷰에서 목숨 걸고 지킨다.

### 2.3 링버퍼 — 우리의 양심

```python
# edge/buffer/ringbuffer.py
class RingBuffer:
    """30초짜리 기억상실증 버퍼. 디스크 안 건드림. RAM only."""
    def push(self, frame):       # 꽉 차면 제일 오래된 프레임 자동 폐기
        ...
    def rescue(self, window):    # 위험 순간 전후를 복사해서 반환
        ...
    def wipe(self):              # 전체를 0으로 덮어씀. 복구 불가.
        ...
```

---

## 3. 모델 스택

| 단계 | 역할 | 채택 (v0.2) | 비고 |
|------|------|------------|------|
| Pose Estimation | 사람 → 스켈레톤 | YOLO-pose (v8s~11x) + 멀티스케일 타일 배치추론 | OKS-NMS로 중복 제거 |
| Skeleton Refine | 폐색복원/평활/확장 | SkeletonEngine (One-Euro + 뼈길이 EMA + GRADIS-25 파생) | 해부학 게이트 |
| Multi-person Track | 누가 누구인지(ID, 얼굴X) | ByteTrack식 2단계 연관 + 헝가리안 매칭 | 좌표 기반 ID만 |
| **Judgment** | **redact 장면 → 상황+항법** | **온디바이스 VLM** (moondream / qwen2.5vl 등) | **유일한 판단 주체** |

> 💡 룰/ST-GCN 분류기는 제거됐다. VLM이 redact 장면 전체(환경 맥락 포함)를 보고
> 판단하므로, 별도 행동분류 모델 학습·배포 파이프라인 자체가 사라졌다.
> 스켈레톤(좌표 텐서)은 여전히 학습/감사용으로 보존 — 얼굴 없음, 숫자만.

---

## 4. 통신 프로토콜 (엣지 → 코어)

```protobuf
// shared/proto/incident.proto
message Incident {
  string  drone_id   = 1;
  int64   ts_utc     = 2;
  Kind    kind       = 3;   // FALL_01, BRAWL_01, LEDGE_01 ...
  float   risk       = 4;   // 0.0 ~ 1.0
  GeoTag  gps        = 5;
  bytes   clip_enc   = 6;   // AES-GCM 암호화된 원본 클립
  Skel[]  skeletons  = 7;   // 분석 근거 (좌표만)
}
```

- 평상시 업링크: **하트비트 + 위치 + 상태만** (영상 없음, 대역폭 아낌)
- ESCALATE 시에만: 위 `Incident` 전체 전송
- 전송: 5G/LTE, mTLS, E2E 암호화

---

## 5. GRADIS Core (지상)

```
[Gateway] ─수신→ [Triage 큐] ─우선순위→ [Dispatch] ─연동→ 경찰 112/상황실
                      │
                      └→ [Audit Log] (append-only, 변조 불가)
                      └→ [Dashboard] (지도 위 실시간 핀, 영상 재생)
```

- **Triage:** 동시다발 사건 시 위험도순 정렬 (압사 > 둔기 > 몸싸움 > 낙상 ...)
- **Audit:** "이 영상 누가, 언제, 왜 열람" 전부 기록. 내부 통제용.
- **Dispatch:** 기존 112 시스템 / 관제 표준과 연동 (어댑터 패턴)

---

## 6. 안전장치 (Fail-safe)

| 상황 | 동작 |
|------|------|
| 통신 두절 | 자율 귀환(RTH) + 로컬 버퍼는 그대로 30초 후 소멸 |
| 배터리 부족 | 가장 가까운 도킹스테이션으로 자동 복귀 |
| AI 확신 낮음 | "관제사 확인 요망"으로 보내서 사람이 판단 (human-in-the-loop) |
| 모델 오작동 의심 | 섀도우 모드(알람 안 보내고 로그만) 전환 가능 |
| 누가 영상 빼돌리려 함 | 감사로그 + 권한분리 + 암호화로 차단 |

---

## 7. 테스트 전략

- **유닛:** 링버퍼 wipe가 진짜 복구 불가인지 (메모리 검증)
- **통합:** "푸쉬업 vs 낙상" 같은 함정 케이스 회귀 테스트 셋
- **레드팀:** 일부러 속여보기 (춤, 운동, 포옹, 장난) → 오탐률 측정
- **프라이버시 감사:** 학습 데이터셋에 원본 프레임이 1바이트라도 새면 빌드 실패시키는 CI 훅 🔒

---

*코드 끝. 핵심만 기억해: **엣지에서 익명화하고, 엣지에서 결정하고, 엣지에서 잊는다.***
