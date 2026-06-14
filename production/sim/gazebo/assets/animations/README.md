# 액터 애니메이션 에셋

`../../scripts/fetch_assets.sh` 실행 → Gazebo Fuel 표준 액터 DAE(walk/run)가
이 폴더에 들어온다. 월드의 fighter/bystander 액터가 이 파일들을 참조한다.

## 고품질 격투/낙상 업그레이드 (권장)
1. [Mixamo](https://www.mixamo.com)에서 X Bot + "Boxing", "Brutal Assassination",
   "Falling Back Death" 등 애니메이션을 FBX(without skin 아님, with skin)로 다운로드
2. Blender에서 FBX 임포트 → Collada(.dae) 익스포트 (Z-up, 단위 m 확인)
3. `fight_a.dae`, `fight_b.dae`, `fall.dae`로 저장 후
   `worlds/gradis_city.sdf`의 fighter 액터 `<skin>/<animation>` 파일명 교체

주의: Gazebo 액터 스키닝은 DAE 스켈레톤 애니메이션을 직접 재생한다 —
트래젝토리(이동)와 스킨 애니메이션(몸짓)을 함께 쓰면 격투처럼 보인다.
