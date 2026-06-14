"""
Avatar Layer (v5 FK) — 추출 좌표(졸라맨)를 '회색 마네킹(형상)'으로 빙의시켜 VLM에 보여준다.

왜 마네킹인가: VLM에 점·선 스켈레톤만 주면 "이 별자리 뭐임?" 상태가 된다.
사람 '형상'이 있어야 장면을 사람답게 이해한다. 기본 리그 = Mixamo X Bot
(assets3d/Xbot.glb, 오프라인 박제). 얼굴 없음·신원 없음. 프라이버시 그대로:
마네킹은 좌표에서 '생성'된 형상이지 원본 픽셀이 아니다.

리타깃 = v5 FK 계층(아래 _bone_mats): ①깊이 복원 → ②루트(골반)만 월드 앵커
(몸통 up + 골반 side로 정면 방향까지) → ③나머지는 부모 기준 로컬 스윙 회전만
(G_child=G_parent·L0·R_swing) → ④미관측 본은 레스트 유지. 본별 독립 글로벌
변환(v3/v4)은 살(가중치 블렌드)이 찢어져 폐기됨.

데모 품질 안정화 (v5.2):
  · 프레임 간 키포인트 적응형 EMA 평활(_smooth_kp) — 빠른 동작엔 덜, 떨림엔 더.
    저신뢰로 떨어진 관절은 직전 위치 유지 → 다리 깜빡임·순간이동 차단.
  · 머리/목 본 스윙 각도 클램프(_clamp_rot) → '대가리 회전' 방지.
  · head 외삽 축소 + 다리 깊이 EMA 강화 → 머리 흔들림·다리 앞뒤 떨림 억제.
로드 1회: GLB 파싱 → 리그 자동감지 → 데시메이션 → 레스트 행렬 캐시.
"""
import os
import numpy as np
import cv2
from PIL import Image

# GRADIS-25 인덱스 (edge.pose.skeleton과 동일)
NOSE = 0
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16
NECK, PELVIS, SPINE, HEAD = 17, 18, 19, 20
LHAND, RHAND, LFOOT, RFOOT = 21, 22, 23, 24

_CONF = 0.15   # 이 미만 관절은 '안 보임' 취급

_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
           5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_accessor(g, blob, idx):
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    n = _NCOMP[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    arr = np.frombuffer(blob, dtype=_DTYPES[a.componentType],
                        count=a.count * n, offset=off)
    return arr.reshape(a.count, n) if n > 1 else arr


def _node_local(n):
    if n.matrix:
        return np.array(n.matrix, np.float32).reshape(4, 4).T
    M = np.eye(4, dtype=np.float32)
    if n.rotation:
        x, y, z, w = n.rotation
        M[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], np.float32)
    if n.scale:
        M[:3, :3] *= np.array(n.scale, np.float32)
    if n.translation:
        M[:3, 3] = n.translation
    return M


def _decimate(verts, faces, vjoint, vweight, grid):
    """버텍스 클러스터링 데시메이션 — grid(m) 셀 단위로 버텍스 병합.
    스킨 가중치는 클러스터 대표 버텍스 것을 쓰고, 위치는 클러스터 평균."""
    key = np.floor(verts / grid).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    nc = counts.shape[0]
    # 클러스터 평균 위치
    pos = np.zeros((nc, 3), np.float64)
    np.add.at(pos, inv, verts)
    pos = (pos / counts[:, None]).astype(np.float32)
    # 대표 버텍스(첫 등장)의 조인트/가중치
    rep = np.full(nc, -1, np.int64)
    first_seen = np.unique(inv, return_index=True)[1]
    rep[inv[first_seen]] = first_seen
    nf = inv[faces]
    keep = ((nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2]))
    return pos, nf[keep].astype(np.int32), vjoint[rep], vweight[rep]


class _RigProfile:
    """본 체인 → GRADIS 타깃 키 매핑. (본이름, 시작타깃키, 끝본이름|None, 끝타깃키|None)"""

    MIXAMO = [
        ("mixamorig:Hips", "pelvis", "mixamorig:Spine", "spine_03"),
        ("mixamorig:Spine", "spine_03", "mixamorig:Spine1", "spine_05"),
        ("mixamorig:Spine1", "spine_05", "mixamorig:Spine2", "spine_08"),
        ("mixamorig:Spine2", "spine_08", "mixamorig:Neck", "neck"),
        ("mixamorig:Neck", "neck", "mixamorig:Head", "head_low"),
        ("mixamorig:Head", "head_low", "mixamorig:HeadTop_End", "head_top"),
        # 쇄골(L/RShoulder)은 의도적으로 미매핑 — 레스트 유지.
        # 좁게 관측된 어깨폭에 쇄골을 맞추면 어깨가 안쪽으로 접힌다(기형).
        # 팔 움직임은 Arm 본부터 시작해도 충분하다.
        ("mixamorig:LeftArm", "l_sho", "mixamorig:LeftForeArm", "l_elb"),
        ("mixamorig:LeftForeArm", "l_elb", "mixamorig:LeftHand", "l_wri"),
        ("mixamorig:LeftHand", "l_wri", None, "l_hand"),
        ("mixamorig:RightArm", "r_sho", "mixamorig:RightForeArm", "r_elb"),
        ("mixamorig:RightForeArm", "r_elb", "mixamorig:RightHand", "r_wri"),
        ("mixamorig:RightHand", "r_wri", None, "r_hand"),
        ("mixamorig:LeftUpLeg", "l_hip", "mixamorig:LeftLeg", "l_kne"),
        ("mixamorig:LeftLeg", "l_kne", "mixamorig:LeftFoot", "l_ank"),
        ("mixamorig:LeftFoot", "l_ank", "mixamorig:LeftToeBase", "l_foot"),
        ("mixamorig:LeftToeBase", "l_foot", None, None),
        ("mixamorig:RightUpLeg", "r_hip", "mixamorig:RightLeg", "r_kne"),
        ("mixamorig:RightLeg", "r_kne", "mixamorig:RightFoot", "r_ank"),
        ("mixamorig:RightFoot", "r_ank", "mixamorig:RightToeBase", "r_foot"),
        ("mixamorig:RightToeBase", "r_foot", None, None),
    ]

    RIGGEDFIGURE = [
        ("torso_joint_1", "pelvis", "torso_joint_2", "spine_05"),
        ("torso_joint_2", "spine_05", "torso_joint_3", "neck"),
        ("torso_joint_3", "neck", "neck_joint_1", "head_low"),
        ("neck_joint_1", "head_low", "neck_joint_2", "head_mid"),
        ("neck_joint_2", "head_mid", None, "head_top"),
        ("arm_joint_L_1", "l_sho", "arm_joint_L_2", "l_elb"),
        ("arm_joint_L_2", "l_elb", "arm_joint_L_3", "l_wri"),
        ("arm_joint_L_3", "l_wri", None, "l_hand"),
        ("arm_joint_R_1", "r_sho", "arm_joint_R_2", "r_elb"),
        ("arm_joint_R_2", "r_elb", "arm_joint_R_3", "r_wri"),
        ("arm_joint_R_3", "r_wri", None, "r_hand"),
        ("leg_joint_L_1", "l_hip", "leg_joint_L_2", "l_kne"),
        ("leg_joint_L_2", "l_kne", "leg_joint_L_3", "l_ank"),
        ("leg_joint_L_3", "l_ank", "leg_joint_L_5", "l_foot"),
        ("leg_joint_L_5", "l_foot", None, None),
        ("leg_joint_R_1", "r_hip", "leg_joint_R_2", "r_kne"),
        ("leg_joint_R_2", "r_kne", "leg_joint_R_3", "r_ank"),
        ("leg_joint_R_3", "r_ank", "leg_joint_R_5", "r_foot"),
        ("leg_joint_R_5", "r_foot", None, None),
    ]

    @staticmethod
    def detect(joint_names):
        if any(n.startswith("mixamorig") for n in joint_names):
            return _RigProfile.MIXAMO
        return _RigProfile.RIGGEDFIGURE


def _targets(kp, W, H):
    """kp: (25,3) 정규화 → dict[타깃키] -> (x,y) px | None"""
    K = kp.shape[0]

    def P(i):
        if i < K and kp[i, 2] >= _CONF:
            return np.array([kp[i, 0] * W, kp[i, 1] * H], np.float32)
        return None

    def mid(a, b):
        pa, pb = P(a), P(b)
        if pa is None or pb is None:
            return pa if pb is None else pb
        return (pa + pb) / 2

    def lerp(a, b, t):
        if a is None or b is None:
            return None
        return a * (1 - t) + b * t

    pelvis = P(PELVIS)
    if pelvis is None:
        pelvis = mid(LHIP, RHIP)
    neck = P(NECK)
    if neck is None:
        neck = mid(LSHO, RSHO)
    head = P(HEAD)
    if head is None:
        head = P(NOSE)
    if head is None and neck is not None and pelvis is not None:
        head = neck + (neck - pelvis) * 0.35

    return {
        "pelvis": pelvis,
        "spine_03": lerp(pelvis, neck, 0.3),
        "spine_05": lerp(pelvis, neck, 0.52),
        "spine_08": lerp(pelvis, neck, 0.78),
        "neck": neck,
        "head_low": lerp(neck, head, 0.55),
        "head_mid": lerp(neck, head, 0.85),
        "head_top": lerp(neck, head, 1.3),
        "clav_l": lerp(neck, P(LSHO), 0.45),
        "clav_r": lerp(neck, P(RSHO), 0.45),
        "l_sho": P(LSHO), "l_elb": P(LELB), "l_wri": P(LWRI), "l_hand": P(LHAND),
        "r_sho": P(RSHO), "r_elb": P(RELB), "r_wri": P(RWRI), "r_hand": P(RHAND),
        "l_hip": P(LHIP), "l_kne": P(LKNE), "l_ank": P(LANK), "l_foot": P(LFOOT),
        "r_hip": P(RHIP), "r_kne": P(RKNE), "r_ank": P(RANK), "r_foot": P(RFOOT),
    }


def _stabilize_head_targets(tg, kp):
    """Keep the mannequin head inside a sane cone around the torso axis.

    Pose models often hallucinate nose/head points during falls, backlight, or
    self-occlusion. If we feed that directly to FK, the neck/head chain can do
    a full horror-movie spin. The VLM cares about body motion far more than
    exact skull yaw, so we clamp head direction and length aggressively.
    """
    pelvis, neck, head_top = tg.get("pelvis"), tg.get("neck"), tg.get("head_top")
    if pelvis is None or neck is None or head_top is None:
        return tg
    body = neck - pelvis
    torso = float(np.linalg.norm(body))
    if torso < 1e-6:
        return tg
    body_dir = body / torso

    raw = (head_top - neck) / 1.3
    raw_len = float(np.linalg.norm(raw))
    head_conf = 0.0
    for i in (HEAD, NOSE):
        if i < len(kp):
            head_conf = max(head_conf, float(kp[i, 2]))

    default_len = 0.32 * torso
    if raw_len < 0.10 * torso or head_conf < 0.28:
        v = body_dir * default_len
    else:
        raw_dir = raw / raw_len
        max_ang = np.deg2rad(55.0 if head_conf >= 0.45 else 38.0)
        dot = float(np.clip(raw_dir @ body_dir, -1.0, 1.0))
        if dot < float(np.cos(max_ang)):
            perp = raw_dir - dot * body_dir
            pn = float(np.linalg.norm(perp))
            if pn < 1e-6:
                raw_dir = body_dir
            else:
                raw_dir = (np.cos(max_ang) * body_dir
                           + np.sin(max_ang) * (perp / pn))
        raw_len = float(np.clip(raw_len, 0.18 * torso, 0.42 * torso))
        v = raw_dir * raw_len

    head = neck + v
    tg = dict(tg)
    tg["head_low"] = neck + v * 0.55
    tg["head_mid"] = neck + v * 0.85
    tg["head_top"] = neck + v * 1.12
    return tg


class MannequinRenderer:
    def __init__(self, glb_path, max_tris=9000, light=(0.22, -0.3, 0.93)):
        from pygltflib import GLTF2
        g = GLTF2().load(glb_path)
        blob = g.binary_blob()

        # ---- 노드 트리 / 레스트 글로벌 ----
        parent = {}
        world = {}

        def walk(i, par, M):
            parent[i] = par
            world[i] = M @ _node_local(g.nodes[i])
            for c in (g.nodes[i].children or []):
                walk(c, i, world[i])

        for r in g.scenes[g.scene].nodes:
            walk(r, None, np.eye(4, dtype=np.float32))

        skin = g.skins[0]
        self.joints = list(skin.joints)
        ibm = _read_accessor(g, blob, skin.inverseBindMatrices) \
            .reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float32)
        self.joint_index = {j: k for k, j in enumerate(self.joints)}
        names = [g.nodes[j].name or f"j{j}" for j in self.joints]
        self.name_to_node = {n: j for n, j in zip(names, self.joints)}

        conv = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)   # y-up → y-down
        self.rest_screen = {n: (conv @ world[j])[:3, 3]
                            for n, j in zip(names, self.joints)}
        # FK 리타깃용: 레스트 글로벌(스크린계), IBM, 부모 기준 레스트 로컬
        self.g_rest = np.stack([conv @ world[j] for j in self.joints])   # (J,4,4)
        self.ibm = ibm                                                   # (J,4,4)

        # 부모 상속용: 조인트를 계층 순서로 (부모 먼저)
        order, par_k = [], []
        seen = set()

        def add(j):
            if j in seen or j not in self.joint_index:
                return
            p = parent.get(j)
            while p is not None and p not in self.joint_index:
                p = parent.get(p)
            if p is not None and p not in seen:
                add(p)
            seen.add(j)
            order.append(self.joint_index[j])
            par_k.append(self.joint_index[p] if p is not None else -1)

        for j in self.joints:
            add(j)
        self.hier_order = np.array(order, np.int64)
        self.hier_parent = np.array(par_k, np.int64)
        self._facing = {}                            # pid -> facing EMA (+toward / -away)
        self._facingsgn = {}                         # pid -> last hard facing sign
        self._facing_fix = os.environ.get("GRADIS_NO_FACING_FIX") != "1"
        self._jparent = dict(zip(order, par_k))      # joint idx -> 부모 joint idx
        self._zstate = {}                            # pid -> {타깃키: z_prev} (3D 리프팅 상태)
        self._kpstate = {}                           # pid -> 평활된 kp (프레임 간 안정화)

        # 레스트 로컬(부모 기준) — FK의 뼈대. L0[j] = G_rest(p)⁻¹ · G_rest(j)
        self.l0 = np.zeros_like(self.g_rest)
        for k, pk in zip(order, par_k):
            if pk < 0:
                self.l0[k] = self.g_rest[k]
            else:
                self.l0[k] = np.linalg.inv(self.g_rest[pk]) @ self.g_rest[k]

        # ---- 메시 수집 + 데시메이션 ----
        prims = []
        total_tris = 0
        for node in g.nodes:
            if node.mesh is None or node.skin is None:
                continue
            for p in g.meshes[node.mesh].primitives:
                v = _read_accessor(g, blob, p.attributes.POSITION).astype(np.float32)
                f = _read_accessor(g, blob, p.indices).reshape(-1, 3).astype(np.int32)
                vj = _read_accessor(g, blob, p.attributes.JOINTS_0).astype(np.int64)
                vw = _read_accessor(g, blob, p.attributes.WEIGHTS_0).astype(np.float32)
                vw = vw / np.maximum(vw.sum(1, keepdims=True), 1e-8)
                mat_name = (g.materials[p.material].name or "") if p.material is not None else ""
                if "joint" in mat_name.lower():
                    # Xbot의 볼조인트 장식 메시 — 2D 리타깃에서 구체가 늘어나
                    # '검은 디스크'가 된다. 서피스 메시만으로 전신이 완전하므로 제외.
                    continue
                prims.append([v, f, vj, vw, 0.80])
                total_tris += len(f)

        # 트라이 예산 맞춰 그리드 키우며 데시메이션
        if total_tris > max_tris:
            height = max(p[0][:, 1].max() - p[0][:, 1].min() for p in prims)
            grid = height / 110.0
            for _ in range(6):
                tris = 0
                dec = []
                for v, f, vj, vw, gray in prims:
                    dv, df, dj, dw = _decimate(v, f, vj, vw, grid)
                    dec.append([dv, df, dj, dw, gray])
                    tris += len(df)
                if tris <= max_tris:
                    break
                grid *= 1.35
            prims = dec
            total_tris = tris

        # 단일 버퍼로 병합 (페인터 정렬을 사람 단위로 한 번에)
        vs, fs, js, ws, grays = [], [], [], [], []
        off = 0
        for v, f, vj, vw, gray in prims:
            vs.append(v); fs.append(f + off); js.append(vj); ws.append(vw)
            grays.append(np.full(len(f), gray, np.float32))
            off += len(v)
        self.verts = np.concatenate(vs)
        self.faces = np.concatenate(fs)
        self.vjoint = np.concatenate(js)
        self.vweight = np.concatenate(ws)
        self.face_gray = np.concatenate(grays)
        self.n_tris = total_tris

        self.profile = _RigProfile.detect(names)
        L = np.array(light, np.float32)
        self.light = L / np.linalg.norm(L)
        print(f"[avatar] rig loaded: {glb_path} joints={len(self.joints)} "
              f"tris={self.n_tris} profile="
              f"{'mixamo' if self.profile is _RigProfile.MIXAMO else 'riggedfigure'}")

    # ---- 본별 변환 ----------------------------------------------------------
    @staticmethod
    def _similarity(p0, p1, q0, q1, s_clamp):
        v0, v1 = p1 - p0, q1 - q0
        n0 = float(np.hypot(*v0))
        if n0 < 1e-6:
            return None
        s = float(np.clip(float(np.hypot(*v1)) / n0, s_clamp[0], s_clamp[1]))
        th = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        c, si = np.cos(th) * s, np.sin(th) * s
        M = np.eye(4, dtype=np.float32)
        M[0, 0], M[0, 1] = c, -si
        M[1, 0], M[1, 1] = si, c
        M[2, 2] = s
        M[:2, 3] = q0 - (M[:2, :2] @ p0)
        return M

    @staticmethod
    def _align(u, v):
        """단위벡터 u→v 최단호 회전 3x3 (Rodrigues). 3D 본 정렬의 핵심."""
        c = float(np.dot(u, v))
        w = np.cross(u, v)
        n2 = float(w @ w)
        if n2 < 1e-12:
            if c > 0:
                return np.eye(3, dtype=np.float32)
            a = np.array([0.0, 1.0, 0.0], np.float32) \
                if abs(u[0]) > 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
            axis = np.cross(u, a)
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], np.float32)
            return (np.eye(3, dtype=np.float32) + 2.0 * (K @ K))
        K = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]], np.float32)
        return (np.eye(3, dtype=np.float32) + K + (K @ K) * ((1 - c) / n2))

    @staticmethod
    def _mat4(R3, s, P0, Q0):
        """M = T(Q0)·(s·R3)·T(-P0) — 레스트 P0(3D)를 타깃 Q0(3D)에 앵커."""
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R3 * s
        M[:3, 3] = Q0 - (R3 * s) @ P0
        return M

    def _body_scale(self, tg, r0):
        """robust 전신 스케일 — 몸통/양다리 투영 중 '최대'를 쓴다.
        (원근 단축된 세그먼트는 작게 투영되므로, 최대 투영이 실스케일에 가장 가깝다)"""
        cands = []

        def seg(k0, k1, b0, b1):
            a, b = tg.get(k0), tg.get(k1)
            if a is None or b is None or b0 not in r0 or b1 not in r0:
                return
            rest = float(np.hypot(*(r0[b0][:2] - r0[b1][:2])))
            if rest > 1e-6:
                cands.append(float(np.hypot(*(a - b))) / rest)

        P = {k: b for b, k, *_ in self.profile}   # 타깃키 -> 본이름
        if "pelvis" in P and "neck" in P:
            seg("pelvis", "neck", P["pelvis"], P["neck"])
        for side in ("l", "r"):
            if f"{side}_hip" in P and f"{side}_ank" in P:
                seg(f"{side}_hip", f"{side}_ank", P[f"{side}_hip"], P[f"{side}_ank"])
        return max(cands) if cands else None

    def _rest_segment(self, bone, end_bone, chain_parent):
        """본의 레스트 3D 세그먼트 (P0, P1). 리프는 부모 방향으로 연장."""
        r0 = self.rest_screen
        P0 = r0[bone]
        if end_bone is not None and end_bone in r0:
            return P0, r0[end_bone]
        par = chain_parent.get(bone)
        ref = r0[par] if par and par in r0 else P0 - 1.0
        return P0, P0 + (P0 - ref)

    def _seed_z(self, bone, t3, key_of_joint):
        """체인 시작키의 z 시드 — 리그 계층을 올라가며 이미 리프팅된 키의 z."""
        ji = self.joint_index.get(self.name_to_node.get(bone, -1), None)
        while ji is not None and ji >= 0:
            k = key_of_joint.get(ji)
            if k and k in t3:
                return float(t3[k][2])
            ji = self._jparent.get(ji, -1)
        return 0.0

    def _facing_sign(self, kp, pid):
        """Estimate body facing (toward vs away from camera) and lock it in time.

        Signal: screen-x ordering of L/R shoulders & hips. A body facing the
        camera has its anatomical-left on the viewer's right (x_L > x_R); a body
        turned away has that flipped. RTMPose's L/R labels are self-consistent
        enough on top/oblique views to read facing, whereas facial-keypoint
        confidence is NOT (the net hallucinates a face on the back of the head).
        Returns +1.0 (toward camera) or -1.0 (facing away)."""
        kp = np.asarray(kp, np.float32)
        raw = 0.0
        wsum = 0.0
        for li, ri, w in ((LSHO, RSHO, 1.0), (LHIP, RHIP, 0.6)):
            cl, cr = float(kp[li, 2]), float(kp[ri, 2])
            if cl < _CONF or cr < _CONF:
                continue
            cw = w * min(cl, cr)
            raw += cw * float(kp[li, 0] - kp[ri, 0])     # >0 toward, <0 away
            wsum += cw
        if len(self._facing) > 64:
            self._facing.clear()
        ema = self._facing.get(pid)
        if wsum > 1e-6:
            obs = raw / wsum
            ema = obs if ema is None else 0.7 * ema + 0.3 * obs
            self._facing[pid] = ema
        if ema is None:
            return 1.0
        return 1.0 if ema >= 0.0 else -1.0

    def _lift(self, tg, S, st, chain_parent, key_of_joint, facing=1.0):
        """2D 타깃 → 3D 타깃. 빠진 깊이는 본 길이 보존으로 복원한다.

        dz = sqrt(L² − |proj|²)   (L = 레스트 본길이 × S, proj = 2D 관측 세그먼트)
        부호(앞/뒤)는 프레임 간 연속성으로 잠그고(z 상태 st), 첫 관측은
        '카메라 쪽(+z)' 기본 — 사람 사지는 대부분 몸 앞에서 움직인다.
        z는 EMA 평활(0.55)로 떨림 억제. 골반 z=0 기준(사람 내부 상대 깊이).
        """
        t3 = {"pelvis": np.array([tg["pelvis"][0], tg["pelvis"][1], 0.0], np.float32)}
        for bone, k0, end_bone, k1 in self.profile:
            if k1 is None or bone not in self.name_to_node:
                continue
            q1_2 = tg.get(k1)
            if q1_2 is None:
                continue
            if k0 not in t3:
                q0_2 = tg.get(k0)
                if q0_2 is None:
                    continue
                t3[k0] = np.array([q0_2[0], q0_2[1],
                                   self._seed_z(bone, t3, key_of_joint)], np.float32)
            base = t3[k0]
            P0, P1 = self._rest_segment(bone, end_bone, chain_parent)
            L = float(np.linalg.norm(P1 - P0)) * S
            proj = q1_2 - base[:2]
            dz = float(np.sqrt(max(0.0, L * L - float(proj @ proj))))
            leg = k1 in ("l_kne", "r_kne", "l_ank", "r_ank", "l_foot", "r_foot")
            if leg:
                dz *= 0.7    # 다리 깊이 댐핑 — 과추정이 골반을 찢는다
            zp, zm = base[2] + dz, base[2] - dz
            prev = st.get(k1)
            if prev is None:
                # 첫 관측 기본 부호: 팔=카메라 쪽(몸 앞), 다리=몸 평면에 가까운 쪽
                z = base[2] + facing * dz             # pop limbs to the body-front side
            else:
                z = zp if abs(zp - prev) <= abs(zm - prev) else zm
                beta = 0.72 if leg else 0.55              # 다리는 더 끈적하게(앞뒤 떨림 억제)
                z = beta * prev + (1.0 - beta) * z        # 시간 평활
            st[k1] = z
            t3[k1] = np.array([q1_2[0], q1_2[1], z], np.float32)
        return t3

    def _root_rotation(self, t3, r0, facing=1.0):
        """루트(골반) 3D 자세 — 몸통 up축 + 골반 좌우축으로 정면 방향까지 복원.

        side 벡터(왼엉덩이−오른엉덩이)가 '몸이 어딜 보는지'를 알려준다.
        등을 돌린 사람은 화면상 좌우가 뒤집히는데, 그게 그대로 side에 반영돼
        루트가 180° 돌아간다 → 팔다리 좌우 꼬임이 루트에서 풀린다.
        """
        up_t = t3["neck"] - t3["pelvis"]
        n = float(np.linalg.norm(up_t))
        if n < 1e-6:
            return None
        up_t = up_t / n

        side_t = None
        for a, b in (("l_hip", "r_hip"), ("l_sho", "r_sho")):
            if a in t3 and b in t3:
                v = t3[a] - t3[b]
                v -= up_t * float(v @ up_t)          # up에 직교화
                vn = float(np.linalg.norm(v))
                if vn > 1e-3:
                    side_t = v / vn
                    break
        if side_t is None:
            side_t = np.cross(up_t, np.array([0, 0, 1], np.float32))
            side_t /= max(np.linalg.norm(side_t), 1e-6)
        fwd_t = np.cross(side_t, up_t)
        # lock forward depth sign (+z = toward camera) to the facing estimate;
        # flipping side_t rotates the root 180 deg about up = front/back swap.
        if facing * fwd_t[2] < 0.0:
            side_t = -side_t
            fwd_t = np.cross(side_t, up_t)

        # 레스트 기준 프레임 (스크린계)
        rb = {k: b for b, k, *_ in self.profile}
        up_r = r0[rb["neck"]] - r0[rb["pelvis"]]
        up_r /= max(np.linalg.norm(up_r), 1e-6)
        side_r = r0[rb["l_hip"]] - r0[rb["r_hip"]]
        side_r -= up_r * float(side_r @ up_r)
        side_r /= max(np.linalg.norm(side_r), 1e-6)
        fwd_r = np.cross(side_r, up_r)

        B_t = np.stack([side_t, up_t, fwd_t], axis=1)
        B_r = np.stack([side_r, up_r, fwd_r], axis=1)
        return (B_t @ B_r.T).astype(np.float32)

    # ---- 안정화 ------------------------------------------------------------
    def _smooth_kp(self, kp, pid):
        """프레임 간 키포인트 평활. 떨림은 죽이고 빠른 동작은 따라간다(적응형 EMA).
        저신뢰로 떨어진 관절은 직전 위치를 유지 → 다리/팔 깜빡임·순간이동 차단."""
        cur = np.asarray(kp, np.float32)
        if cur.ndim != 2 or cur.shape[1] < 3:
            return cur
        if len(self._kpstate) > 64:
            self._kpstate.clear()
        prev = self._kpstate.get(pid)
        out = cur.copy()
        if prev is not None and prev.shape == cur.shape:
            for i in range(cur.shape[0]):
                cc = float(cur[i, 2])
                if cc >= _CONF and prev[i, 2] >= _CONF:
                    mv = float(np.hypot(cur[i, 0] - prev[i, 0],
                                        cur[i, 1] - prev[i, 1]))
                    a = float(np.clip(0.65 - mv * 4.0, 0.2, 0.65))  # prev 가중(이동↑→평활↓)
                    out[i, 0] = a * prev[i, 0] + (1.0 - a) * cur[i, 0]
                    out[i, 1] = a * prev[i, 1] + (1.0 - a) * cur[i, 1]
                    out[i, 2] = cc
                elif cc < _CONF and prev[i, 2] >= _CONF:
                    out[i, 0], out[i, 1] = prev[i, 0], prev[i, 1]   # 관측 끊김: 직전 유지
                    out[i, 2] = float(prev[i, 2]) * 0.7             # 신뢰만 감쇠
        self._kpstate[pid] = out.copy()
        return out

    @staticmethod
    def _clamp_rot(R, max_rad):
        """3x3 회전의 각도 크기를 max_rad로 제한(축 보존). 머리/목 스윙 폭주 방지."""
        c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
        ang = float(np.arccos(c))
        if ang <= max_rad or ang < 1e-6:
            return R
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                       R[1, 0] - R[0, 1]], np.float32)
        s = float(np.linalg.norm(ax))
        if s < 1e-8:
            return R
        ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]], np.float32)
        return (np.eye(3, dtype=np.float32) + np.sin(max_rad) * K
                + (1.0 - np.cos(max_rad)) * (K @ K)).astype(np.float32)

    def _bone_mats(self, kp, W, H, pid=0):
        """리타깃 v5 — FK(Forward Kinematics) 계층 리타깃. '진짜 리깅'.

        v3/v4의 본별 독립 글로벌 변환은 본 사이 살(가중치 블렌드 영역)을
        찢었다 — 골반 실종, 몸통-다리 분리의 근본 원인. v5는 리깅의 정석:
          1) 깊이 복원(v4의 lift)으로 관절 3D 타깃 확보
          2) 루트(골반)만 월드에 앵커 — up축(몸통) + side축(골반 라인)으로
             '정면 방향'까지 포함한 3D 자세 (등 돌림 = 좌우 꼬임 해결)
          3) 나머지 전부 부모 기준 '로컬 스윙 회전'만: 관측된 본 방향을
             부모 프레임으로 가져와 레스트 로컬 방향에 최단호 정렬
          4) 미관측 본은 로컬 회전 0 = 레스트 포즈 유지 (자연스러운 손/발)
        자식은 수학적으로 부모에 붙어 있다(G_child = G_parent·L0·R_swing).
        골반이 사라지거나 허리가 끊기는 것은 '구조적으로' 불가능하다.
        """
        kp = self._smooth_kp(kp, pid)            # 프레임 간 평활(떨림·깜빡임 억제)
        tg = _stabilize_head_targets(_targets(kp, W, H), kp)
        pelvis, neck = tg["pelvis"], tg["neck"]
        if pelvis is None or neck is None:
            return None
        r0 = self.rest_screen
        S = self._body_scale(tg, r0)
        if S is None or S < 1e-6:
            return None

        chain_parent = {e: b for b, _, e, _2 in self.profile if e is not None}
        key_of_joint = {}
        for b, k, *_ in self.profile:
            node = self.name_to_node.get(b)
            if node is not None:
                key_of_joint[self.joint_index[node]] = k

        if len(self._zstate) > 64:
            self._zstate.clear()
        st = self._zstate.setdefault(pid, {})
        f = self._facing_sign(kp, pid) if self._facing_fix else 1.0
        if self._facingsgn.get(pid) not in (None, f):
            st.clear()                                # facing flipped -> re-seed depth signs
        self._facingsgn[pid] = f
        t3 = self._lift(tg, S, st, chain_parent, key_of_joint, facing=f)
        if "neck" not in t3:
            return None

        R_root = self._root_rotation(t3, r0, facing=f)
        if R_root is None:
            return None

        # 본 → (관측 시작키, 관측 끝키, 레스트 끝 3D) 룩업
        seg_of = {}
        for bone, k0, end_bone, k1 in self.profile:
            if bone in self.name_to_node:
                P0, P1 = self._rest_segment(bone, end_bone, chain_parent)
                seg_of[self.joint_index[self.name_to_node[bone]]] = (k0, k1, P0, P1)

        root_name = next(b for b, k, *_ in self.profile if k == "pelvis")
        root_ji = self.joint_index[self.name_to_node[root_name]]

        J = len(self.joints)
        G = np.zeros((J, 4, 4), np.float32)
        # 주먹: 손가락 본은 세그먼트마다 0.5배 축소(누적 0.5³≈0.13) —
        # 벌어진 레스트 손가락('안녕하살법' 포즈)을 손안으로 말아 넣는다.
        fist = np.diag([0.5, 0.5, 0.5, 1.0]).astype(np.float32)
        idx_name = {self.joint_index[n]: name
                    for name, n in self.name_to_node.items()}
        _FINGER = ("Thumb", "Index", "Middle", "Ring", "Pinky")

        for ji, pi in zip(self.hier_order, self.hier_parent):
            if ji == root_ji or pi < 0:
                # 루트: 레스트 위에 '델타' 합성 — G = Δ·g_rest.
                # (리그 노드의 자체 스케일(예: Mixamo 0.01 cm단위)을 교체하면
                # 자식 오프셋이 폭주한다. 레스트는 보존하고 위에 얹는다.)
                A3 = (R_root * S).astype(np.float32)
                Gj = np.eye(4, dtype=np.float32)
                Gj[:3, :3] = A3 @ self.g_rest[ji][:3, :3]
                Gj[:3, 3] = t3["pelvis"] + A3 @ (self.g_rest[ji][:3, 3]
                                                 - self.g_rest[root_ji][:3, 3])
                G[ji] = Gj
                continue

            base = G[pi] @ self.l0[ji]
            if any(f in idx_name.get(ji, "") for f in _FINGER):
                G[ji] = base @ fist
                continue
            seg = seg_of.get(ji)
            if seg is not None:
                k0, k1, P0, P1 = seg
                Q0, Q1 = (t3.get(k0), t3.get(k1) if k1 else None)
                if Q0 is not None and Q1 is not None:
                    d_w = Q1 - Q0
                    dn = float(np.linalg.norm(d_w))
                    if dn > 1e-3:
                        # 관측 방향을 이 본의 로컬 프레임으로 → 레스트 로컬
                        # 본 방향에 최단호 스윙 (비틀림 최소 = 캔디랩 꼬임 방지)
                        Rb = base[:3, :3]
                        v_l = np.linalg.inv(Rb) @ d_w
                        v_l /= max(float(np.linalg.norm(v_l)), 1e-9)
                        u_l = np.linalg.inv(self.g_rest[ji][:3, :3]) @ (P1 - P0)
                        u_l /= max(float(np.linalg.norm(u_l)), 1e-9)
                        Re = np.eye(4, dtype=np.float32)
                        Rsw = self._align(u_l.astype(np.float32),
                                          v_l.astype(np.float32))
                        nm = idx_name.get(ji, "")
                        if any(h in nm for h in ("Head", "head")):
                            Rsw = self._clamp_rot(Rsw, 0.35)   # ≈20°: head spin lock
                        elif any(h in nm for h in ("Neck", "neck")):
                            Rsw = self._clamp_rot(Rsw, 0.52)   # ≈30°: neck still follows torso
                        Re[:3, :3] = Rsw
                        base = base @ Re
            G[ji] = base

        return G @ self.ibm

    # ---- 스키닝 + 렌더 ------------------------------------------------------
    def _skin(self, kp, W, H, pid=0):
        Mskin = self._bone_mats(kp, W, H, pid)           # (J,4,4) = G_new·IBM
        if Mskin is None:
            return None
        # 버텍스당 블렌디드 매트릭스: Mv = Σ w_k · Mskin[joint_k]
        Mv = np.einsum("vk,vkab->vab", self.vweight, Mskin[self.vjoint])
        vh = np.concatenate([self.verts,
                             np.ones((len(self.verts), 1), np.float32)], 1)
        return np.einsum("vab,vb->va", Mv, vh)[:, :3]

    @staticmethod
    def _person_depth(kp):
        """장면 깊이 프록시 — 값이 클수록 카메라에 가깝다(앞).

        수직/비스듬 위 CCTV에서는 발·뒤꿈치 접지점이 낮을수록(화면 y 클수록)
        카메라에 가깝다. 발 좌표가 신뢰 있으면 그걸, 아니면 보이는 최하단 관측점,
        그것도 없으면 평균 y로 폴백. 정규화 [0,1] y."""
        kp = np.asarray(kp, np.float32)
        if kp.ndim != 2 or kp.shape[1] < 3:
            return 0.0
        conf = kp[:, 2] >= _CONF
        foot = np.zeros(kp.shape[0], bool)
        for i in (LANK, RANK, LFOOT, RFOOT):
            if i < kp.shape[0]:
                foot[i] = True
        fc = conf & foot
        if fc.any():
            return float(kp[fc, 1].max())
        if conf.any():
            return float(kp[conf, 1].max())
        return float(kp[:, 1].mean())

    def render_into(self, canvas_bgr, persons, supersample=1):
        """canvas_bgr: np.uint8[H,W,3] (in-place 합성). persons: [(id, kp25)].
        supersample>1: 오프라인 고품질 — sc배로 렌더 후 INTER_AREA 축소(계단·시임 제거).
        시임 메움: 삼각형을 중심에서 ~0.7px 팽창 + LINE_AA로 인접 면 1px 틈을 덮는다.
        정렬: 면의 '가장 먼 정점'(min-z) 기준 뒤→앞 — mean-z보다 상호관통 아티팩트 적음."""
        H, W = canvas_bgr.shape[:2]
        sc = max(1, int(supersample))
        work = (cv2.resize(canvas_bgr, (W * sc, H * sc), interpolation=cv2.INTER_LINEAR)
                if sc > 1 else canvas_bgr)
        # 사람을 장면 깊이로 far→near 정렬 후 합성 — 트랙 리스트 순서는 깊이와
        # 무관해 뒤 사람이 앞 사람 위에 덮였다(가림 반대). 가까운 사람을 마지막에
        # 그려 앞 사람이 뒤 사람을 자연히 가리게 한다(사람별 공유 z-buffer 대용).
        persons = sorted(persons, key=lambda pk: self._person_depth(pk[1]))
        for pid, kp in persons:
            v = self._skin(np.asarray(kp, np.float32), W, H, pid)
            if v is None:
                continue
            if sc > 1:
                v = v.copy(); v[:, :2] *= sc
            tri = v[self.faces]                              # (F,3,3)
            order = np.argsort(tri[:, :, 2].min(1))          # 먼 면부터(뒤→앞)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
            lam = np.abs(nrm @ self.light)                   # 양면 램버트
            shade = np.clip(self.face_gray * (0.66 + 0.42 * lam), 0, 1)
            cen = tri[:, :, :2].mean(1, keepdims=True)       # 시임 메움: 중심서 팽창
            d = tri[:, :, :2] - cen
            dn = np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-6)
            pts = np.round(tri[:, :, :2] + d / dn * (0.7 * sc)).astype(np.int32)
            for f in order:
                gv = int(shade[f] * 255)
                cv2.fillConvexPoly(work, pts[f], (gv, gv, gv), cv2.LINE_AA)
        if sc > 1:
            canvas_bgr[:, :, :] = cv2.resize(work, (W, H), interpolation=cv2.INTER_AREA)
        return canvas_bgr

    def compose(self, frame, persons, supersample=1):
        """redact된 PIL 프레임 위에 마네킹 합성 → PIL RGB. (오프라인 GIF는 supersample=2 권장)"""
        bgr = cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.render_into(bgr, persons, supersample=supersample)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
