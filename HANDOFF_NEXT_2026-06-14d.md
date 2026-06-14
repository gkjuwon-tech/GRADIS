# 핸드오프 — 다음 주자에게 (2026-06-14d, facing/앞뒤 라운드)

이전 핸드오프 `HANDOFF_NEXT_2026-06-14c.md`(운영법·눈검증 체크리스트·STOP 기준)와
`TOPVIEW_POSE_STATUS_2026-06-14*.md`(문제정의·FlyPose 맥락)를 먼저 읽어라.
이 문서는 그 위에 "이번 라운드에 무엇을 했고, 어디까지가 한계인가"만 정직하게 덧붙인다.

---

## 0. 한 줄 요약

> **비스듬 위(oblique elevated) CCTV에서 서 있는/걷는 사람을 마네킹으로 덮을 때
> 가장 먼저 깨지는 건 "앞/뒤(facing)"였다.** 등 보이고 걸어가는 사람을 렌더러가
> "카메라를 향한다"고 가정해 깊이가 뒤집혀 → 웅크린/꺾인 기괴한 포즈가 됐다.
> 이번에 **앞/뒤 이진 판정(facing-sign)**으로 그걸 고쳐서, **가림 없는 보행자는
> "등지고 걸어감"이 읽히게** 됐다. 단, 이건 **180° 플립까지만**이다 — 연속 방향(yaw)은
> 아직 못 한다. (다음 주자의 본진)

---

## 1. 이번 라운드에 한 것

### 1.1 검증 무대를 옳게 옮김
- 이전 주자(와 내가 처음에) 검증한 **"위에서 본 누운 사람"(mixkit 33459/15817)**은
  카메라 축에 수직 평면 = **정면 보기와 동일** → COCO가 원래 풀던 거. **함정.**
- 진짜 무대 = **고정 CCTV가 천장/벽 높이에서 비스듬히 내려다보는 실제 영상**(서 있는/걷는,
  foreshortening 있는). 깃허브(Intel sample-videos)에서 실다운로드:
  - `store-aisle-detection.mp4` (720×404, 보행자 5명, 최대 286px) — **주 무대**
  - `people-detection.mp4` (768×432, 232px) — 보조
  - 사람 키 게이트(≥150px)는 `production/tools/footage_probe.py`로 확인.

### 1.2 회전-TTA는 비스듬 CCTV에서 무효임을 확인 (가설 폐기)
- §4-B "회전-TTA(평면 회전 환각 때리기)"를 store-aisle에서 측정:
  - baseline: jitter 0.0095 / bone_cv 0.182 / anat 0.948
  - rot-TTA:  jitter 0.0113 / bone_cv 0.202 / anat 0.953  → **오히려 악화**
- 이유: TTA는 **in-plane(화면 평면) 회전**을 때린다. 비스듬 CCTV의 문제는
  **view-angle foreshortening + 앞/뒤**라 축이 다르다. → TTA는 opt-in(`--rot-tta`)으로
  코드만 남기고 **기본 OFF**. 이 무대의 답이 아니다.

### 1.3 근본 원인 = 앞/뒤(facing) 판정 실패 → 고침
- **측정으로 신호 검증**(store-aisle 보행자 전원이 등지고 걸어감):
  - **얼굴 키포인트 신뢰도는 신호로 못 씀.** RTMPose가 **뒤통수에도 코/눈을 conf
    0.4~0.95로 환각**한다. (face-conf 기반 앞/뒤 판정은 여기서 깨진다.)
  - **쓸 수 있는 신호 = 어깨/엉덩이의 화면 x 좌우 순서.** 정면이면 해부학적 왼쪽이
    화면 오른쪽(x_L > x_R), 등지면 뒤집힌다. store-aisle 전원이 `x_L - x_R < 0`(등짐)
    으로 **일관**되게 나왔다 → 검출기 L/R 라벨이 자기일관적이라 facing을 읽을 수 있다.
- **수정(`production/edge/avatar.py`)**:
  - `_facing_sign(kp, pid)`: 어깨(가중1.0)+엉덩이(0.6) 화면 x 차를 conf 가중·EMA(0.7)로
    누적 → **+1(정면) / −1(후면)** 한 부호. 프레임 간 hysteresis로 깜빡임 억제.
  - `_lift(...)`: 깊이 첫 관측 기본 부호를 `base.z + facing*dz`로 — **사지를 몸-앞쪽으로**
    띄운다(등지면 −z=카메라 반대쪽). (기존엔 무조건 +z=카메라 쪽으로 띄워서 등진 사람이
    웅크려 보였다.)
  - `_root_rotation(...)`: forward 깊이 부호를 facing에 **고정**(`facing*fwd_t[2]<0`이면
    side_t 반전=루트 180° about up) → 등진 사람 좌우/앞뒤 꼬임 차단.
  - facing 부호가 뒤집히면 해당 pid의 z-state를 리셋해 깊이를 새 부호로 재시드.
  - A/B용 환경변수: `GRADIS_NO_FACING_FIX=1`이면 fix 끄고 옛 동작.

---

## 2. 검증 결과 (눈검증 PRIMARY)

대상: `_media/cctv/store_aisle_12fps.mp4` (720×404@12, 146f, 전 구간).
콘택트시트: `_media/cctv/cmp_facing_store.png` (원본 | facing-fix, f40/80/120),
확대: `_media/cctv/zoom_right.png`(보행자), `_media/cctv/zoom_left.png`(가림쌍).

| 항목 | baseline | facing-fix |
|---|---|---|
| 가림 없는 보행자(우측 2~3인) | 웅크림/꺾임, "걷기" 안 읽힘 (§3.3 FAIL) | **등지고 서서 걸어감 읽힘 (PASS)** |
| 앞/뒤 일관성 | 프레임마다 뒤집힘 | 안정 (등짐 유지) |
| 가림+겹침 쌍(좌측, 접시 테이블 뒤) | 융합/기괴 | 여전히 융합/기괴 (미해결) |
| anat / bone_cv / jitter | 0.948 / 0.182 / 0.0095 | facing은 렌더단 변경이라 2D 지표 동일, **눈검증으로만 개선 확인** |

→ **부분 통과.** 핵심 결손(앞/뒤)은 고쳤고, **가림·겹침**과 **연속 yaw**는 남았다.

---

## 3. 한계 (★ 다음 주자가 반드시 알 것)

1. **앞/뒤 이진(180°)까지만.** facing은 +1/−1 부호다. **오른쪽 35°(3/4 측면)로 걷는
   사람**은 toward/away 둘 중 하나로 스냅돼 → 마네킹 몸통 yaw가 실제와 어긋난다.
   사장님 지적 그대로: *"오른쪽 35°를 보는 사람을 마네킹으로 덮어씌울 수 없다."*
   **연속 방향(yaw) 추정**이 진짜 다음 단계다.
2. **인물 간 가림/겹침 미해결.** 좌측 쌍처럼 두 사람이 화면에서 겹치거나 접시 테이블에
   하반신이 가리면 트랙 키포인트가 섞여 마네킹이 융합·기괴해진다. facing fix는
   이걸 못 고친다(occlusion/association 문제).
2.5. **사람 간 깊이 정렬/가림 없음 (render z-ordering).** 뒤에 있는 사람이 앞 사람에게
   가려져야 하는데, 지금은 **각 사람 마네킹을 독립 패스로 렌더해 합성**하므로 그리는
   순서가 깊이와 무관하다 → **뒤 사람이 앞 사람 위에 덮인다(가림 반대).** 사장님 캡처
   그대로. 근본원인: 트랙 간 공유 depth buffer가 없고 페인터(far→near) 정렬이 없음.
   해결방향: (a) 프레임마다 사람을 깊이로 정렬(발 끝/pelvis의 화면 y, 또는 bbox 바닥,
   또는 사람 스케일=가까울수록 큼)해 **far→near 순서로 합성**, (b) 더 정확히는 사람들을
   **하나의 공유 z-buffer**에 같이 래스터화해 앞 사람 픽셀이 뒤 사람을 자연히 가리게.
   현재 렌더 루프는 사람별로 `_skin`→합성이라 (a)는 합성 순서만 바꾸면 되고, (b)는
   depth 통합이 필요하다.
3. **여전히 2D→깊이 휴리스틱 lift.** `dz=sqrt(L²−proj²)` + 부호선택은 단안 깊이
   휴리스틱이라 foreshortening이 크면 사지 깊이가 과·오추정된다. 3D-네이티브가 아님.
4. **저해상 + COCO-17 한계.** 286px 사람에서 손/발/머리 디테일은 거칠다. 얼굴 방향은
   애초에 키포인트로 안 들어온다(코/눈 환각).

---

## 4. 다음 아키텍처 권고 (우선순위)

> 한 문장 목표: **"앞/뒤 부호"를 "연속 몸통 yaw(0~360°)"로 승급**하고, 가능하면
> 휴리스틱 깊이 lift를 **단안 3D mesh recovery**로 대체한다.

1. **(권장 본진) 단안 3D human mesh recovery 통합** — 4D-Humans(HMR2.0)/CLIFF/
   BEDLAM류. 출력이 **SMPL(전역 orientation 포함)**이라:
   - 연속 yaw·pitch가 공짜로 나온다 → 35° 측면도 정확히 덮인다.
   - 깊이 부호 휴리스틱(`_lift`/facing-sign) **자체가 불필요**해진다.
   - 마네킹은 SMPL 포즈를 리타깃만 하면 됨(현 FK 리그 재사용 가능).
   - CPU 추론 비용이 관건 → 사람 crop만 배치 추론, 트랙별 캐시.
2. **(경량 대안) 방향 회귀 헤드/리프터로 yaw만 연속화** — 어깨·엉덩이 라인 + 2D
   포즈로 몸통 yaw를 연속 추정(작은 회귀기 or 기하 기반). facing-sign을 부드러운
   각도로 교체. mesh recovery보다 싸지만 정확도/일반화는 낮다.
3. **FlyPose 항공-네이티브 2D (게이트)** — 비스듬/항공에서 2D 자체 품질을 올린다.
   Google Form 게이트라 가중치 받아야 함(§6).
4. **다중 인물 깊이 합성 + 가림/association 보강** —
   (a) **render z-ordering**: 사람을 깊이로 정렬해 far→near 합성, 또는 공유 z-buffer로
   같이 래스터화 → 앞 사람이 뒤 사람을 가리게(현재 반대로 덮임, §3.2.5).
   (b) 트랙 간 키포인트 소유권(겹칠 때) + 사람-물체 가림 마스킹.
   mesh recovery로 가면 (a)는 사람별 실제 depth가 생겨 자연 해결되고 (b)도 일부 완화되나
   여전히 별도 처리 필요.

---

## 5. 실행 방법 (재현)

```bash
# 환경 (CPU/py3.11, Colab 금지)
bash production/scripts/setup_container_cpu.sh
source .venv/bin/activate
export MMPOSE_ROOT=$PWD/production/_external/mmpose
cd production

# 클립 게이트(사람 ≥150px 확인)
python tools/footage_probe.py _media/cctv/store-aisle-detection.mp4

# 마네킹 렌더 (facing fix 기본 ON) — 출력 _media/cctv/avatar_<base>.mp4
python tools/avatar_on_video.py _media/cctv/store_aisle_12fps.mp4 \
  --model yolo11m.pt --pose-preset rtmpose-m --imgsz 960 --conf 0.3 --max-frames 146

# A/B: facing fix 끄기
GRADIS_NO_FACING_FIX=1 python tools/avatar_on_video.py ... (위와 동일)

# 수치
python tools/pose_eval.py _media/cctv/store_aisle_12fps.mp4 \
  --model yolo11m.pt --pose-preset rtmpose-m --imgsz 960 --conf 0.3 --max-frames 146 \
  --out _media/cctv/pe_base_store.json
# rot-TTA(이 무대엔 무효, 참고용): 위에 --rot-tta 추가
```

검증은 **항상 눈검증 우선**(원본|마네킹 전 구간, `HANDOFF_NEXT_2026-06-14c.md §3` 체크리스트).
수치만 좋고 눈으로 깨지면 통과 아님.

---

## 6. 남은 블로커 (도망 말고 사장님께 올릴 것)

1. **FlyPose 가중치 + FlyPose-104 (Google Form 게이트)** — 항공-네이티브 2D의 본체.
   - 가중치: https://docs.google.com/forms/d/e/1FAIpQLSeod94pdwgIum41gbu3f7Q43nv7E2BLDq43WMYY4Fd20gYmVQ/viewform
   - FlyPose-104: https://docs.google.com/forms/d/e/1FAIpQLSdu98Ukj6---OFhHWNGc5_PLH8L0RcikVS1voJ7vZNdORFnwg/viewform
2. **낙상/싸움 들어간 큰-사람 top-down/비스듬 클립** — DONE 기준(≥3 클립, 1개는 행동)용.
   무료 스톡 top-down은 죄다 정적("누워있기")이라 행동 보존 검증이 막힘.

---

## 7. §8 로그 (이번 라운드)

- 가설: 비스듬 CCTV 마네킹 붕괴의 1차 원인은 앞/뒤(facing) 오판이다.
  - 결과: **맞음.** 어깨/엉덩이 화면 x 부호 기반 facing-sign으로 깊이 pop·루트 forward를
    고정 → 가림 없는 보행자 "등지고 걸어감" 눈검증 PASS. (얼굴 conf는 환각으로 무효)
  - 다음: 연속 yaw(단안 3D mesh recovery 권장)로 승급. **사람 간 깊이 정렬/가림(뒤
    사람이 앞 사람에 가려지게, §3.2.5)** + 가림/association 별도.
- 가설(폐기): 회전-TTA가 비스듬 CCTV를 개선한다 → **틀림**(축이 다름, 수치 악화). opt-in만 유지.
