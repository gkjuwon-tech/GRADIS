"""
2D→3D temporal lifting (VideoPose3D) + 시점-적응 재투영.

왜 3D인가 (핵심 아키텍처 결정):
  단일 프레임 2D pose는 시점 모호성을 못 푼다. 탑뷰에서 팔이 몸 위로 겹쳐
  보이면 2D상 어깨≈팔꿈치≈손목이 한 점에 뭉쳐 정면 학습 모델은 '옆모습 굽힌
  팔'로 환각한다. 깊이가 사라진 평면에선 누가 봐도 못 푼다. 그래서 시간축
  맥락으로 2D 시퀀스를 3D로 들어올린다(lift). 3D에선 뼈 길이가 '상수'다 —
  이게 우리가 그토록 찾던 시점 불변식이다.

파이프라인 (트랙 1개, 오프라인):
  1. COCO-17 2D 시퀀스 수집 (엔진 출력, 이미지좌표)
  2. VideoPose3D TemporalModel → H36M-17 3D[T] (루트상대, 뼈길이 일정)
  3. 프레임마다 약투영 카메라 M(2x3)를 관측 2D에 최소제곱으로 피팅
     (orthogonal/affine Procrustes — intrinsics 불필요, 각도 하드코딩 없음)
  4. 3D 전체를 M으로 재투영 → 보정된 2D
     → 시간 일관 + 해부학 유효 + 폐색 복원 + '그 프레임의 실제 시점' 반영

가중치: VideoPose3D h36m/detectron-coco 사전학습 (입력 COCO-17 순서 = 우리와
동일). 가중치는 _external/weights/videopose3d_h36m_coco.bin (git 미포함).
"""
from __future__ import annotations

import os

import numpy as np

from .skeleton import (LANK, LEAR, LELB, LEYE, LHIP, LKNE, LSHO, LWRI, NOSE,
                       RANK, REAR, RELB, REYE, RHIP, RKNE, RSHO, RWRI)

# H36M 17-joint 인덱스 (VideoPose3D 출력 순서)
H_HIP, H_RHIP, H_RKNE, H_RANK = 0, 1, 2, 3
H_LHIP, H_LKNE, H_LANK = 4, 5, 6
H_SPINE, H_THORAX, H_NECKNOSE, H_HEAD = 7, 8, 9, 10
H_LSHO, H_LELB, H_LWRI = 11, 12, 13
H_RSHO, H_RELB, H_RWRI = 14, 15, 16

# COCO-17 -> H36M-17 매핑. 파생 관절(hip-center, spine, thorax, head)은 COCO에
# 직접 대응이 없어 lift 입력에선 안 쓰고, 출력 H36M을 COCO로 되돌릴 때만 쓴다.
H2C = {
    H_RHIP: RHIP, H_RKNE: RKNE, H_RANK: RANK,
    H_LHIP: LHIP, H_LKNE: LKNE, H_LANK: LANK,
    H_LSHO: LSHO, H_LELB: LELB, H_LWRI: LWRI,
    H_RSHO: RSHO, H_RELB: RELB, H_RWRI: RWRI,
    H_NECKNOSE: NOSE,
}

# H36M 뼈 (시점 불변 길이 측정용)
H_BONES = [
    (H_HIP, H_RHIP), (H_RHIP, H_RKNE), (H_RKNE, H_RANK),
    (H_HIP, H_LHIP), (H_LHIP, H_LKNE), (H_LKNE, H_LANK),
    (H_HIP, H_SPINE), (H_SPINE, H_THORAX), (H_THORAX, H_NECKNOSE),
    (H_NECKNOSE, H_HEAD),
    (H_THORAX, H_LSHO), (H_LSHO, H_LELB), (H_LELB, H_LWRI),
    (H_THORAX, H_RSHO), (H_RSHO, H_RELB), (H_RELB, H_RWRI),
]

_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "_external", "weights", "videopose3d_h36m_coco.bin")


# ---------------------------------------------------------------------------
# VideoPose3D TemporalModel (filter_widths=[3,3,3,3,3], channels=1024, RF=243)
# ---------------------------------------------------------------------------
def _build_model():
    import torch.nn as nn

    class TemporalModel(nn.Module):
        def __init__(self, num_joints_in=17, in_features=2, num_joints_out=17,
                     filter_widths=(3, 3, 3, 3, 3), dropout=0.25, channels=1024):
            super().__init__()
            self.num_joints_in = num_joints_in
            self.in_features = in_features
            self.num_joints_out = num_joints_out
            self.pad = [filter_widths[0] // 2]
            self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels,
                                         filter_widths[0], bias=False)
            self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
            self.drop = nn.Dropout(dropout)
            self.relu = nn.ReLU(inplace=True)
            layers_conv, layers_bn = [], []
            next_dilation = filter_widths[0]
            for i in range(1, len(filter_widths)):
                self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
                layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i],
                                             dilation=next_dilation, bias=False))
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
                layers_conv.append(nn.Conv1d(channels, channels, 1, bias=False))
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
                next_dilation *= filter_widths[i]
            self.layers_conv = nn.ModuleList(layers_conv)
            self.layers_bn = nn.ModuleList(layers_bn)
            self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

        def receptive_field(self):
            return 1 + 2 * sum(self.pad)

        def forward(self, x):  # x: [B,T,J,2]
            sz = x.shape[:3]
            x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
            for i in range(len(self.pad) - 1):
                pad = self.pad[i + 1]
                res = x[:, :, pad: x.shape[2] - pad]
                x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
                x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
            x = self.shrink(x)
            x = x.permute(0, 2, 1).view(sz[0], -1, self.num_joints_out, 3)
            return x

    return TemporalModel()


class Lifter3D:
    _shared = None

    def __init__(self, weights=None, device="cpu"):
        import torch

        self.torch = torch
        self.device = device
        path = weights or _DEFAULT_WEIGHTS
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"VideoPose3D weights not found: {path}\n"
                "Download: curl -o _external/weights/videopose3d_h36m_coco.bin "
                "https://dl.fbaipublicfiles.com/video-pose-3d/"
                "pretrained_h36m_detectron_coco.bin")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_pos"] if "model_pos" in ck else ck
        self.model = _build_model()
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(device)
        self.rf = self.model.receptive_field()

    @classmethod
    def shared(cls, weights=None):
        if cls._shared is None:
            cls._shared = cls(weights)
        return cls._shared

    def lift(self, seq2d, w, h, prenorm=False):
        """seq2d: [T,17,2]. prenorm=False면 이미지좌표(px)로 보고 전체이미지 screen
        normalization. prenorm=True면 이미 [-1,1] 정규화된 것으로 보고 그대로 사용.
        반환 3d: [T,17,3] H36M 루트상대."""
        torch = self.torch
        X = seq2d.astype(np.float32).copy()
        if not prenorm:
            # VideoPose3D screen normalization: x in [-1,1] by width, y keeps aspect.
            X[..., 0] = X[..., 0] / w * 2 - 1
            X[..., 1] = X[..., 1] / w * 2 - h / w
        pad = self.rf // 2
        Xp = np.pad(X, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        with torch.no_grad():
            inp = torch.from_numpy(Xp[None]).to(self.device)
            out = self.model(inp).cpu().numpy()[0]  # [T,17,3]
        return out


def _person_normalize(seq_px, conf, min_conf=0.3):
    """프레임별 root(midhip) 중심화 + 트랙 고정 스케일로 사람이 정규화 프레임을
    채우게 한다(H36M 학습분포 정렬). 반환 [T,17,2] (~[-1,1])."""
    T = seq_px.shape[0]
    # root: midhip(보이면) 아니면 보이는 관절 평균
    roots = np.zeros((T, 2), np.float32)
    spans = []
    for t in range(T):
        vis = conf[t] > min_conf
        if (conf[t, LHIP] > min_conf) and (conf[t, RHIP] > min_conf):
            roots[t] = (seq_px[t, LHIP] + seq_px[t, RHIP]) / 2
        elif vis.any():
            roots[t] = seq_px[t, vis].mean(0)
        if vis.sum() >= 2:
            p = seq_px[t, vis]
            spans.append(max(p[:, 1].ptp(), p[:, 0].ptp()))
    scale = float(np.median(spans)) if spans else 1.0
    scale = max(scale, 1e-3)
    out = (seq_px - roots[:, None, :]) / scale * 0.8
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 시점-적응 재투영: 3D를 관측 2D에 affine(약투영)으로 맞춰 되돌린다.
# ---------------------------------------------------------------------------
def _fit_affine(X3, x2, wts):
    """min_M || (X3·M^T) - x2 ||_w,  M:(2,4) [3x rot/scale + translation].
    X3:(N,3), x2:(N,2), wts:(N,). 가중 최소제곱."""
    N = X3.shape[0]
    A = np.concatenate([X3, np.ones((N, 1), np.float32)], axis=1)  # (N,4)
    W = wts[:, None]
    AtA = (A * W).T @ A  # (4,4)
    AtA += np.eye(4, dtype=np.float32) * 1e-6
    Atb = (A * W).T @ x2  # (4,2)
    M = np.linalg.solve(AtA, Atb)  # (4,2)
    return M  # x2 ≈ A @ M


def refine_track(kp_seq, w, h, lifter=None, min_conf=0.3, blend=0.85):
    """
    kp_seq: [T,>=17,3] 정규화([0,1]) COCO-17 (+파생). 이미지좌표가 아니라 정규화면
            w,h로 px 변환해 lift, 다시 정규화로 반환.
    반환: (kp_refined[T,>=17,3] 정규화, info dict)
      - 보정된 0..16 좌표를 관측과 blend (관측 신뢰 높으면 관측 쪽 유지).
      - 17.. 파생관절은 호출측에서 extend_keypoints로 재계산 권장.
    """
    if lifter is None:
        lifter = Lifter3D.shared()
    T = kp_seq.shape[0]
    out = kp_seq.copy()
    info = {"bone_cv3d": None}
    if T < 5:
        return out, info

    seq2d_px = kp_seq[:, :17, :2].copy()
    seq2d_px[..., 0] *= w
    seq2d_px[..., 1] *= h
    conf = kp_seq[:, :17, 2]

    seq_norm = _person_normalize(seq2d_px, conf)
    p3d = lifter.lift(seq_norm, w, h, prenorm=True)  # [T,17,3] H36M

    # 시점 불변 3D 뼈길이 변동(품질 지표 — 작을수록 일관)
    cvs = []
    for a, b in H_BONES:
        L = np.linalg.norm(p3d[:, a] - p3d[:, b], axis=1)
        if L.mean() > 1e-6:
            cvs.append(L.std() / L.mean())
    info["bone_cv3d"] = float(np.mean(cvs)) if cvs else None

    # 프레임별 affine 피팅: H36M 3D -> 관측 COCO 2D(px). 매핑된 관절만 사용.
    h_idx = list(H2C.keys())
    c_idx = [H2C[i] for i in h_idx]
    refined_px = seq2d_px.copy()
    for t in range(T):
        wts = conf[t, c_idx].astype(np.float32)
        good = wts > min_conf
        if good.sum() < 6:
            continue
        X3 = p3d[t, np.array(h_idx)[good]]
        x2 = seq2d_px[t, np.array(c_idx)[good]]
        M = _fit_affine(X3, x2, wts[good])
        # 전체 H36M 재투영 후 COCO 자리에 기입
        A = np.concatenate([p3d[t], np.ones((17, 1), np.float32)], axis=1)
        proj = A @ M  # (17,2) px
        for hi, ci in H2C.items():
            refined_px[t, ci] = proj[hi]

    # 관측과 blend: 관측 신뢰 높은 곳은 관측 유지, 낮은 곳은 보정 우위.
    for t in range(T):
        for ci in range(17):
            c = conf[t, ci]
            a = blend * (1.0 - min(1.0, c / 0.6))  # conf 낮을수록 보정 비중↑
            mixed = (1 - a) * seq2d_px[t, ci] + a * refined_px[t, ci]
            out[t, ci, 0] = mixed[0] / w
            out[t, ci, 1] = mixed[1] / h
            # 보정으로 살아난 폐색 관절은 최소 신뢰 부여
            if c < min_conf:
                out[t, ci, 2] = max(c, 0.3)
    return out, info


def bone_cv3d_only(kp_seq, w, h, lifter=None):
    """좌표는 안 건드리고 3D 일관성 지표만 반환(측정용)."""
    if lifter is None:
        lifter = Lifter3D.shared()
    if kp_seq.shape[0] < 5:
        return None
    seq = kp_seq[:, :17, :2].copy()
    seq[..., 0] *= w
    seq[..., 1] *= h
    seq = _person_normalize(seq, kp_seq[:, :17, 2])
    p3d = lifter.lift(seq, w, h, prenorm=True)
    cvs = []
    for a, b in H_BONES:
        L = np.linalg.norm(p3d[:, a] - p3d[:, b], axis=1)
        if L.mean() > 1e-6:
            cvs.append(L.std() / L.mean())
    return float(np.mean(cvs)) if cvs else None
