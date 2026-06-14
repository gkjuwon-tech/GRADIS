#!/usr/bin/env python3
"""
스켈레톤 엔진 품질 검증 — '좋아졌다'를 말이 아니라 숫자로 증명한다.

방법:
  1) 합성 시나리오의 깨끗한 스켈레톤 = '정답(GT)'
  2) 실제 pose 모델처럼 열화: 고주파 지터 + 관절 폐색(드롭) + 이상치(outlier)
  3) 열화된 검출을 SkeletonEngine에 통과
  4) 엔진 전(raw) vs 엔진 후(engine)의 오차를 GT와 비교

지표:
  MPJPE  : 관절당 평균 위치오차 (정규화 좌표; 낮을수록 좋음)
  Jitter : 프레임간 떨림 (정지 시 0이어야 이상적)
  IDsw   : ID 스왑 횟수 (가해자/피해자 안 섞이는지)
  Recover: 폐색 관절 복원 성공률
"""
import sys, os, argparse
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import synthetic
from edge.pose import SkeletonEngine
from edge.pose.skeleton import torso_size

rng = np.random.default_rng(7)


def degrade(kp, jitter=0.006, drop_p=0.15, outlier_p=0.03):
    """깨끗한 GT(17,3) → 실제 모델 출력처럼 열화된 검출(17,3)."""
    out = kp.copy().astype(np.float32)
    n = out.shape[0]
    # 1) 고주파 지터
    out[:, :2] += rng.normal(0, jitter, size=(n, 2))
    out[:, 2] = 0.9
    # 2) 폐색: 일부 관절 신뢰도 급락 + 위치 망가짐
    drop = rng.random(n) < drop_p
    out[drop, :2] += rng.normal(0, 0.05, size=(drop.sum(), 2))
    out[drop, 2] = rng.uniform(0.0, 0.18, size=drop.sum())
    # 3) 이상치: 모델이 가끔 엉뚱한 데 찍음
    out_mask = rng.random(n) < outlier_p
    out[out_mask, :2] += rng.normal(0, 0.18, size=(out_mask.sum(), 2))
    return out, drop


def centroid(kp, conf_min=0.2):
    v = kp[kp[:, 2] > conf_min]
    return v[:, :2].mean(0) if len(v) else np.array([0.5, 0.5])


def match_gt(persons, gts):
    """엔진 출력 person을 GT(id->kp)에 최근접 매칭. 반환 dict person.id -> gt_id."""
    mapping = {}
    for p in persons:
        pc = centroid(p.kp)
        best, bd = None, 1e9
        for gid, gkp in gts.items():
            d = np.hypot(*(pc - centroid(gkp)))
            if d < bd:
                best, bd = gid, d
        if best is not None and bd < 0.15:
            mapping[p.id] = best
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--jitter", type=float, default=0.006)
    args = ap.parse_args()

    engine = SkeletonEngine(default_fps=args.fps, aerial=True)

    raw_err, eng_err = [], []
    raw_jit, eng_jit = [], []
    recover_hit, recover_tot = 0, 0
    id_switch = 0
    prev_map = {}      # gt_id -> engine_id (직전)
    prev_raw = {}      # gt_id -> raw kp (지터 측정)
    prev_eng = {}      # gt_id -> eng kp
    n_frames = 0

    for t, frame, gt_persons in synthetic.frames(fps=args.fps):
        gts = {gid: gkp for gid, gkp in gt_persons}
        # 열화된 검출 만들기
        detections, drops = [], {}
        for gid, gkp in gt_persons:
            dkp, drop = degrade(gkp, args.jitter, args.drop)
            detections.append({"kp": dkp, "_gid": gid})
            drops[gid] = (dkp, drop)

        people = engine.process([{"kp": d["kp"]} for d in detections], t)
        if not people:
            continue
        n_frames += 1

        # 엔진출력 ↔ GT 매칭
        p_by_id = {p.id: p for p in people}
        m = match_gt(people, gts)        # engine_id -> gt_id
        gt_to_engine = {g: e for e, g in m.items()}

        # ID 스왑 카운트
        for gid, eid in gt_to_engine.items():
            if gid in prev_map and prev_map[gid] != eid:
                id_switch += 1
        prev_map = dict(gt_to_engine)

        sc = 0.30  # 정규화 스케일 기준(몸통)
        for gid, gkp in gts.items():
            # raw 검출(열화) 오차
            dkp, drop = drops[gid]
            re = np.linalg.norm(dkp[:, :2] - gkp[:, :2], axis=1).mean()
            raw_err.append(re)
            # 엔진 오차
            if gid in gt_to_engine:
                p = p_by_id[gt_to_engine[gid]]
                ee = np.linalg.norm(p.kp[:, :2] - gkp[:, :2], axis=1).mean()
                eng_err.append(ee)
                # 폐색 복원 성공률: 드롭된 관절이 GT 근처(<0.04)로 복원됐나
                for j in np.where(drop)[0]:
                    recover_tot += 1
                    if np.hypot(*(p.kp[j, :2] - gkp[j, :2])) < 0.05:
                        recover_hit += 1
                # 지터(프레임간 변화)
                if gid in prev_eng:
                    eng_jit.append(np.linalg.norm(p.kp[:, :2] - prev_eng[gid][:, :2], axis=1).mean())
                prev_eng[gid] = p.kp.copy()
            if gid in prev_raw:
                raw_jit.append(np.linalg.norm(dkp[:, :2] - prev_raw[gid][:, :2], axis=1).mean())
            prev_raw[gid] = dkp.copy()

    def mm(a): return (np.mean(a) if a else float("nan"))
    print("=" * 64)
    print(" SKELETON ENGINE — QUALITY REPORT")
    print(f" frames evaluated: {n_frames},  injected drop={args.drop:.0%}, jitter={args.jitter}")
    print("-" * 64)
    print(f" MPJPE  raw → engine : {mm(raw_err):.5f} → {mm(eng_err):.5f}  "
          f"({(1 - mm(eng_err)/mm(raw_err))*100:.1f}% better)")
    print(f" Jitter raw → engine : {mm(raw_jit):.5f} → {mm(eng_jit):.5f}  "
          f"({(1 - mm(eng_jit)/mm(raw_jit))*100:.1f}% smoother)")
    print(f" Occlusion recovery  : {recover_hit}/{recover_tot} "
          f"({(recover_hit/max(1,recover_tot))*100:.1f}% of dropped joints recovered to GT)")
    print(f" ID switches         : {id_switch}  (lower=better; 0 ideal)")
    print("=" * 64)


if __name__ == "__main__":
    main()
