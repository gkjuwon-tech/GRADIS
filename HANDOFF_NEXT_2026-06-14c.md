# 핸드오프 — 다음 주자에게 (2026-06-14c, 상세판)

앞 주자(나)의 정직한 인수인계 + 다음 주자가 그대로 따라 돌릴 수 있는 **무한루프
운영 매뉴얼**이다. 자랑 아니라 **내 실수 → 진짜 해야 할 일 → 통과/중단 기준 →
현재 vs 목표 아키텍처** 순서로 적는다. 앞의 `TOPVIEW_POSE_STATUS_2026-06-14b.md`는
톤이 자화자찬이라 이 문서로 정정·대체한다.

---

## 0. 미션 한 줄

> **어떤 각도에서 찍힌 영상이든**(특히 **드론/천장 = top-down**) 사람의 실제
> pose/행동을 유지하면서, 자연스러운 익명 마네킹/3D 모델로 **완벽하게 덮어씌운다.**

핵심 단어 두 개: **top-down(위에서)** 과 **눈으로 본 완벽한 덮기**. 이 둘을 놓치면
헛짓이다(내가 그랬다).

---

## 1. 내가 한 헛짓 (같은 함정 피하라고 박제)

1. **옆/사선뷰로 검증했다.** `box1`(복싱, 옆면), `angle_meet`(사선)에서 숫자를 뽑음.
   **드론은 위에 있다.** 옆/사선 대형 사람은 **YOLO만으로 원래 됐다.** 이미 풀린
   케이스에서 숫자 자랑 = 기여 0.
2. **마네킹 눈검증을 안 했다.** `pose_eval` 숫자만 봤다. 완료기준은 **시각**인데.
3. **FlyPose 다운 막히자 항공 자체를 포기**하고 받기 쉬운 옆면 데이터로 도망. 막힌 걸
   사용자에게 올리지도 않음.
4. **헤드라인이라던 3D 리프트(VideoPose3D)는 top-down에 무력**(§5-B 이유).
5. **클립을 "받기 쉬운 것"으로 골랐다.** 사람이 10~20px라 눈검증 불가였는데도 진행.

→ **정확히 반대로 해라**: 항공 가중치 실제 확보 / 진짜 드론샷(큰 사람) 확보 /
엔진 고도화 / **top-down 마네킹 눈검증** / 될 때까지 무한루프.

---

## 2. 무한루프 운영 원칙 (이대로 돌려라)

각 이터레이션 = **가설 1개 → 구현 → top-down 눈검증 + 수치 → 판정 → 기록**.

```
LOOP:
  1. 다음 가설 1개 선택 (§4 백로그에서). "왜 top-down이 개선되는가"를 한 문장으로.
  2. 최소 변경으로 구현.
  3. 검증 (반드시 둘 다, top-down 클립에서):
       (a) 눈검증: 원본 | 마네킹 나란히 영상/콘택트시트 → §3 체크리스트로 PASS/FAIL.
       (b) 수치: tools/pose_eval.py 로 베이스라인 대비. 숫자는 보조, 눈이 우선.
  4. 판정:
       - 눈검증 PASS & 수치 개선  → 채택, 커밋(메시지에 어느 클립/지표인지 명시).
       - 눈검증 FAIL 또는 수치 악화 → 되돌리고 가설 폐기. _media/eval에 근거 남김.
  5. 기록: 이 문서 §8 로그에 1줄(가설/결과/다음).
  6. 중단조건(§7) 점검 → 아니면 LOOP.
```

철칙:
- **숫자만 보고 PASS 금지.** 반드시 영상/이미지를 Read 툴로 직접 띄워 눈으로 봐라.
- **옆/사선으로 PASS 금지.** 판정은 항상 **top-down(가능하면 90° 천장)** 클립으로.
- **검증 없이 커밋/번들 금지.**
- **막힌 의존성(게이트 다운로드 등)은 숨기지 말고 즉시 사용자에게 올려라.**
- pose 제거 금지 / silhouette-only 금지 / 각도 하드코딩 금지.

---

## 3. 눈검증 통과(PASS) 기준 — 마네킹이 "완벽하게 덮였나"

원본 영상과 마네킹 영상을 **나란히 놓고, 전 구간을 재생**하며(체리픽 프레임 금지)
**사람마다** 아래를 확인한다. 하나라도 깨지면 FAIL.

### 3.1 덮임(coverage) — 모양/정렬
- [ ] 마네킹이 원본 사람의 **몸통 위에 정렬**된다(머리는 머리, 발은 발). 좌우/상하
      밀림(슬라이딩) 없음.
- [ ] 마네킹 사지가 원본 실루엣을 **크게 벗어나 삐져나오지 않는다**(환각 팔다리 X).
- [ ] 원본 사람의 픽셀이 마네킹/블러로 **가려져 신원 식별 불가**(프라이버시).
- [ ] 사람이 여럿일 때 **마네킹이 엉뚱한 사람에게 옮겨붙지 않는다**(ID 안정).

### 3.2 시간 안정성
- [ ] 마네킹이 **깜빡(나타났다 사라짐)거리지 않는다.**
- [ ] **순간이동/슬라이딩** 없음. 프레임 사이 자연스러운 연속 이동.
- [ ] 같은 사람의 포즈가 **프레임마다 딴판으로 튀지 않는다**(뼈길이 일정해 보임).

### 3.3 행동 보존(behavior) — **마네킹만 보고 동작이 읽히나**
- [ ] 원본을 가린 채 **마네킹만** 봐도 걷기/달리기/앉기/**낙상**/밀기/손드는 동작이
      **그대로 식별**된다.
- [ ] 특히 **낙상(top_fall류)·싸움**처럼 안전에 중요한 동작이 마네킹에서 살아있다.

### 3.4 각도 일반화
- [ ] **top-down(90°)**, **사선(45°)**, 수평 클립 **모두**에서 위 항목 통과.
- [ ] 같은 사람이 **회전하며 이동**해도(위에서 보면 방향이 계속 바뀜) 마네킹이
      따라 돈다(머리가 거꾸로 박히는 등 깨짐 없음).

> 판정 기록: 클립명 + 통과/실패 항목 + 콘택트시트 경로를 §8 로그에 남겨라.

---

## 4. 진짜 해야 할 일 — 백로그(우선순위)

### (A) 제대로 된 항공 footage 확보 — 사람이 충분히 커야(≥~150px)
- 내가 받은 SPHAR 항공 클립은 사람이 10~20px라 눈검증 불가. **이게 1순위 블로커.**
- 후보: **Okutama-Action**(github `miquelmarti/Okutama-Action`, UAV 10~45m, 45°/90°,
  걷기·달리기·앉기·**눕기(낙상)** 등 12종 — 미션에 딱. 단 원본은 대용량/접근 절차 있을
  수 있음), **UAV-Human**, **VisDrone**, **ICG/Drone-Action**, **MOD20**.
- 다운로드 사실: 익명 `git clone`은 막힘. **HTTP 직접 다운로드는 됨**
  (`raw.githubusercontent.com`, github releases, `dl.fbaipublicfiles.com`,
  `download.openmmlab.com`). 대용량/드라이브/폼 게이트면 **사용자에게 요청**.
- **반드시 frame 한 장 띄워 사람 크기부터 눈으로 확인하고 시작.**

### (B) 탑다운 2D pose 모델 — **본체**
COCO 정면 학습 모델(rtmpose-m, yolo-pose)이 탑뷰를 "옆모습 굽힌 팔"로 환각하는 게
근본 병폐. 후보(우선):
1. **FlyPose** (arXiv **2601.05747**, WACV2026. 구성: RT-DETRv2-S 검출 + ViTPose 포즈,
   aerial 전용). repo: `github.com/farooqhassaan/FlyPose`.
   - **가중치 = Google Form 게이트** (확인함):
     `docs.google.com/forms/d/e/1FAIpQLSeod94pdwgIum41gbu3f7Q43nv7E2BLDq43WMYY4Fd20gYmVQ/viewform`
   - **FlyPose-104 데이터셋도 Form 게이트**:
     `docs.google.com/forms/d/e/1FAIpQLSdu98Ukj6---OFhHWNGc5_PLH8L0RcikVS1voJ7vZNdORFnwg/viewform`
   - **→ 이건 자율로 못 받는다. 사용자에게 폼 작성/가중치 업로드를 요청해라.**
     받으면 `model/checkpoints/detector/*.onnx`, `model/checkpoints/pose/*/end2end.onnx`
     구조. `edge/pose/backends.py`에 **항공 2D 백엔드로 드롭인**(아래 §6 인터페이스).
2. **가중치 받기 전 자율 대안(병행 가능)**:
   - **회전 TTA**: 위에서 본 사람은 임의 in-plane 회전. crop을 0/45/.../315°로 돌려
     pose 추론 → **평균 keypoint conf 최대인 회전 채택** → 키포인트 역회전.
     각도 하드코딩 아님(데이터가 고름). 항공 2D 환각 직접 타격. `MMPoseTopDownPose.__call__`에
     삽입. **(B) 모델 없이도 즉시 top-down 개선 가능성 — 먼저 시도할 만함.**
   - 더 큰 모델(rtmpose-l/x, vitpose) + `--tiles 2~3`로 작은 항공 사람 회복.
   - 항공 데이터로 RTMPose/ViTPose 파인튜닝(FlyPose-104 확보 시).

### (C) 마네킹 리타깃 — 2D FK → **3D-네이티브**(아키텍처 전환, §6)
- 현재 마네킹은 2D 키포인트 + 깊이복원 휴리스틱으로 FK. top-down에서 깊이 모호 → 깨짐.
- 항공 2D가 좋아지면(또는 2D→3D lift가 신뢰되면) **3D 관절로 리그를 직접 구동**하고
  추정 카메라로 투영. 시점 일관 마네킹.

### (D) 측정 하네스 보강
- `pose_eval.py`에 **top-down 전용 케이스**(콘택트시트 자동생성 + 원본 나란히)를 넣어
  눈검증을 한 명령으로. 현재는 `--annotate`(스켈레톤)만. 마네킹 콘택트시트가 필요.

---

## 5. 현재 아키텍처 (코드 맵 + 데이터 흐름)

```
영상 프레임 (BGR)
  │
  ▼  edge/pose/backends.py : TopViewPoseBackend
  ├─ YoloPersonDetector (yolo11x.pt)        : 사람 박스 + BoT-SORT track_id   [잘 됨]
  └─ MMPoseTopDownPose (rtmpose-m/vitpose-s): 박스별 17 COCO 키포인트         [top-down서 환각]
        └ _gate_topview_limbs: 저신뢰 사지 conf=0 (임시 게이트)
  │  list[dict{kp[17,3] norm, bbox, score, track_id}]
  ▼  edge/pose/engine.py : SkeletonEngine.process
  ├─ aerial weight(얼굴 conf↓)  edge/pose/skeleton.py: AERIAL_KPT_WEIGHT
  ├─ tracker.py MultiPersonTracker: track_id 우선 매칭, OneEuro 평활
  │    └ occlusion.py recover: 강체이동/뼈길이/대칭으로 폐색 관절 복원
  │    └ viewgeom.py update_and_clamp: (내가 추가) 사영 제약 클램프 [top-down 미검증]
  ├─ anatomical_validity(kp): 2D 뼈길이 비율 게이트  [★ top-down서 오작동, §5-A]
  └─ extend_keypoints: COCO-17 → GRADIS-25(파생 8관절)
  │  list[Person{id, ext[25,3], inferred, quality, norm}]
  ▼  edge/privacy.py redact_people: 원본 사람 블러
  ▼  edge/avatar.py MannequinRenderer.compose: GLB 리그 v5 FK 리타깃 [★ 2D FK, §5-C]
  │    (assets3d/Xbot.glb, Mixamo)
  ▼  합성 프레임 → mp4/gif (tools/avatar_on_video.py)
```
오프라인 실험: `tools/avatar_pose_mask_fusion.py`(FastSAM 게이트), `tools/pose_eval.py`(지표),
`tools/avatar_lift_demo.py`(내가 추가, 옆면서만 돌려봄), `edge/pose/lift3d.py`(VideoPose3D).

### 5-A 왜 top-down에서 깨지나 (근본 원인)
1. **2D 정면 학습 pose 모델의 시점 모호성**: 위에서 보면 어깨≈팔꿈치≈손목이 한 점에
   뭉쳐 → 모델이 "옆모습 굽힌 팔"로 환각. **데이터/크기 문제가 아니라 표현 문제**
   (평면 투영에서 깊이 소실). → 항공-학습 2D(§4-B) 또는 회전 TTA 필요.
2. **`anatomical_validity`가 2D 뼈길이 비율로 검증** → top-down은 단축(foreshortening)으로
   뼈가 짧게/torso가 짧게 투영 → 정상인데도 "비정상"으로 깐다(detect_rate↓). 또한
   **투영은 뼈를 짧게만 만들 수 있어** "너무 짧음"이 환각인지 단축인지 단일프레임 구분
   불가 = **단일프레임 2D 검증은 원리적으로 약함** → 시간축/3D 필요.
3. **마네킹 2D FK + 깊이복원 휴리스틱**: top-down 깊이 모호 → 리그가 비틀림.

### 5-B 왜 내 3D 리프트(lift3d.py)가 top-down에 무력
- VideoPose3D는 **Human3.6M(지상 정면) 학습** → 탑다운을 본 적 없음(분포 밖).
- 재투영 blend가 보수적이라 **고신뢰 관절은 관측 유지, 저신뢰만 보정** → 깨끗한
  클립선 no-op, 정작 망가지는 탑뷰선 미검증. **"좋은 2D가 먼저" 순서가 틀림.**

---

## 6. 목표 아키텍처 (바뀌어야 할 것)

```
프레임
  ▼ 검출/추적            : YOLO11 (+aerial 파인튜닝 가능) — 유지, tiles로 소형 회복
  ▼ ★ 항공 2D pose       : FlyPose(RT-DETRv2-S+ViTPose) ─ 신규 백엔드
       또는 회전-TTA로 감싼 RTMPose/ViTPose (가중치 확보 전 대안)
       └ 출력 동일 계약: kp[17,3] norm, bbox, score, track_id  ← 드롭인 핵심
  ▼ 시점-강건 검증        : 2D 뼈비율 게이트 → (a)단축 허용 + (b)시간축 사영제약/3D 일관
       └ anatomical_validity 재설계: 짧은 뼈 페널티 제거, "길어짐 불가" 제약으로
  ▼ (선택) 2D→3D lift     : 항공-적합 3D(예: 항공 데이터 lift) — H36M VideoPose3D는 부적합
  ▼ ★ 마네킹 리타깃       : 2D FK → 3D-네이티브(3D 관절로 리그 구동 + 추정 카메라 투영)
  ▼ redact + compose → mp4/gif + 마네킹 콘택트시트(눈검증 자동 산출)
```

전환 포인트(우선):
1. **`backends.py`에 항공 2D 백엔드 드롭인**(계약 동일하면 엔진/마네킹 무수정).
2. **`anatomical_validity` 시점-강건화**(단축 허용; top-down detect_rate 회복).
3. **회전-TTA**(가중치 전 즉효 가능).
4. **마네킹 3D-네이티브**(가장 큰 공사, 2D pose 신뢰 회복 후).

---

## 7. 중단(STOP) 기준

### 7.1 성공으로 종료 (DONE)
- **서로 다른 top-down/aerial 클립 ≥3개**(가능하면 90° 천장 포함, 사람 ≥150px)에서,
  **전 구간 재생** 눈검증 §3 체크리스트를 **사람마다 모두 PASS**.
- 그 중 ≥1개는 **낙상/싸움** 등 행동 보존이 중요한 클립.
- 보조로 `pose_eval` 수치가 베이스라인 대비 개선(jitter/accel/bone_cv/flicker↓,
  detect_rate/track_len/anat↑). **수치만 좋고 눈검증 실패면 DONE 아님.**
- DONE 시: 검증 영상/콘택트시트 경로 + 수치표를 문서화하고, production 파이프라인/
  (필요시) 번들에 반영.

### 7.2 사용자에게 올리고 대기 (BLOCKED → ASK)
- **게이트 다운로드**(FlyPose 가중치/FlyPose-104, 대용량 드론셋 등)가 막힘 → 폼 작성/
  업로드를 사용자에게 요청. **혼자 우회로 옆면 도망 금지(내 실수).**
- 아키텍처 분기(예: 마네킹 3D 전환 범위)가 크고 모호 → 사용자 확인.

### 7.3 가설 폐기/피벗
- 한 가설로 **3 이터레이션 연속 개선 없음** → 폐기하고 다음 백로그로.
- 눈검증 FAIL인데 수치만 좋아지는 패턴 반복 → 지표가 현실과 괴리. 지표 재설계.

---

## 8. 재사용 자산 / 미검증 스캐폴딩 / 진행 로그

### 검증된 인프라 (그대로 써라)
- `production/scripts/setup_container_cpu.sh` — CPU 컨테이너 설치+검증(멱등). **됨.**
  `python3 -m venv .venv && . .venv/bin/activate && bash production/scripts/setup_container_cpu.sh`
  `export MMPOSE_ROOT=$PWD/production/_external/mmpose`
- `tools/pose_eval.py` — 지표 + `--compare`. **단, 반드시 top-down 클립 + 눈검증과 병용.**
- 네트워크/다운로드 사실은 §4-A.

### 미검증(top-down서 증명 안 됨) — 신뢰 말고 재평가
- `edge/pose/viewgeom.py` (사영 클램프), `edge/pose/lift3d.py` (VideoPose3D),
  `tools/avatar_lift_demo.py` (옆면서만 확인). 아이디어 참고용. top-down 검증 필수.

### 진행 로그 (다음 주자가 이어서 1줄씩 추가)
- 2026-06-14b(나): env 설치✅ / SPHAR footage 확보(사람 작음) / pose_eval 신설 /
  viewgeom·lift3d 추가 → **전부 옆·사선서만 측정, top-down 기여 0. 눈검증 안 함.** 폐기 교훈.
- (다음) ▢ 큰-사람 항공 클립 1개 확보 + frame 눈확인 → ▢ 회전-TTA 시도 →
  ▢ FlyPose 가중치 사용자 요청 → ▢ top-down 마네킹 콘택트시트 눈검증 …

### 상태
- PR **#2 (draft)**. 커밋: setup·pose_eval·viewgeom·lift3d·avatar_lift_demo·문서.
- git 미포함(용량): 모델 가중치, `_media/` 영상. **컨테이너 휘발 주의 — 코드만 살아남음.**
- CI 워크플로 없음(깨질 것 없음).

**다음 주자: §4-A부터. 큰 항공 클립 한 장 띄워놓고, §3 눈검증을 매 루프 강제하며 시작해라.**
