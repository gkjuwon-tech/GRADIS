"""
Mask-assisted pose validation.

Pose remains the primary signal for behavior analysis. A segmentation mask is
used only as a second AI opinion: keypoints and limb segments that fall outside
the visible person silhouette lose confidence before temporal tracking.
"""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import (
    LANK,
    LELB,
    LHIP,
    LKNE,
    LSHO,
    LWRI,
    RANK,
    RELB,
    RHIP,
    RKNE,
    RSHO,
    RWRI,
)


CHECK_LIMBS = [
    (LSHO, LELB),
    (LELB, LWRI),
    (RSHO, RELB),
    (RELB, RWRI),
    (LHIP, LKNE),
    (LKNE, LANK),
    (RHIP, RKNE),
    (RKNE, RANK),
]


def fuse_pose_with_mask(kp_norm, mask, min_conf=0.20, sample_count=9):
    """Return a keypoint copy whose confidence is gated by the mask.

    Args:
        kp_norm: np.ndarray[17,3], normalized x/y/conf.
        mask: np.ndarray[H,W], float/bool person mask from a segmentation model.

    The function never fabricates new joints. It only lowers confidence when a
    pose joint or limb is inconsistent with the model-observed silhouette.
    """
    kp = np.asarray(kp_norm, np.float32).copy()
    if mask is None or mask.size == 0:
        return kp, {"outside_points": 0, "bad_limbs": 0, "mask_area": 0}

    mask01 = (mask.astype(np.float32) > 0.35).astype(np.uint8)
    h, w = mask01.shape[:2]
    area = int(mask01.sum())
    if area < 20:
        return kp, {"outside_points": 0, "bad_limbs": 0, "mask_area": area}

    outside_dist = cv2.distanceTransform(1 - mask01, cv2.DIST_L2, 3)
    tol = max(2.5, float(np.sqrt(area)) * 0.045)
    outside_points = 0

    for j in range(min(len(kp), 17)):
        if kp[j, 2] < min_conf:
            continue
        x = int(np.clip(round(kp[j, 0] * (w - 1)), 0, w - 1))
        y = int(np.clip(round(kp[j, 1] * (h - 1)), 0, h - 1))
        d = float(outside_dist[y, x])
        if d <= tol:
            continue
        outside_points += 1
        if d > tol * 2.2:
            kp[j, 2] = 0.0
        else:
            kp[j, 2] *= 0.25

    bad_limbs = 0
    for a, b in CHECK_LIMBS:
        if kp[a, 2] < min_conf or kp[b, 2] < min_conf:
            continue
        xs = np.linspace(kp[a, 0] * (w - 1), kp[b, 0] * (w - 1), sample_count)
        ys = np.linspace(kp[a, 1] * (h - 1), kp[b, 1] * (h - 1), sample_count)
        xs = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
        ys = np.clip(np.round(ys).astype(np.int32), 0, h - 1)
        inside_ratio = float(mask01[ys, xs].mean())
        if inside_ratio < 0.38:
            bad_limbs += 1
            kp[b, 2] *= 0.20
        elif inside_ratio < 0.55:
            bad_limbs += 1
            kp[b, 2] *= 0.55

    return kp, {
        "outside_points": outside_points,
        "bad_limbs": bad_limbs,
        "mask_area": area,
    }


def bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = aa + bb - inter
    return inter / den if den > 1e-6 else 0.0


def match_masks_to_poses(pose_boxes, masks, mask_boxes, min_iou=0.15):
    matches = []
    used = set()
    for pbox in pose_boxes:
        best_i, best = None, 0.0
        for i, mbox in enumerate(mask_boxes):
            if i in used:
                continue
            score = bbox_iou(pbox, mbox)
            if score > best:
                best, best_i = score, i
        if best_i is not None and best >= min_iou:
            used.add(best_i)
            matches.append(best_i)
        else:
            matches.append(None)
    return matches

