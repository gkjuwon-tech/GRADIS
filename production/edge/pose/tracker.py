"""
Multi-person pose tracker.

The tracker accepts detections produced by the new top-view pipeline:
YOLO detector/tracker boxes plus MMPose keypoints. If BoT-SORT/ByteTrack gives
an external track id, that id wins. If not, geometry matching keeps the old
behavior for offline clips and simple tests.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .filters import OneEuroArray
from .occlusion import recover, update_bone_lengths
from .skeleton import N_KPT, anatomical_validity, torso_size

HIGH_CONF = 0.50
LOW_CONF = 0.20


def _centroid(kp):
    v = kp[kp[:, 2] > 0.2]
    return v[:, :2].mean(0) if len(v) else np.array([0.5, 0.5], np.float32)


class Track:
    _next_id = 1

    def __init__(self, det, t):
        ext_id = det.get("track_id")
        if ext_id is None:
            self.id = Track._next_id
            Track._next_id += 1
            self.state = "tentative"
        else:
            self.id = int(ext_id)
            Track._next_id = max(Track._next_id, self.id + 1)
            self.state = "confirmed"

        self.filter = OneEuroArray(min_cutoff=1.2, beta=0.25)
        self.bone_len = {}
        self.kp = det["kp"].copy()
        self.inferred = np.zeros(N_KPT, bool)
        self.history = deque(maxlen=64)
        self.centroid = _centroid(det["kp"])
        self.vel = np.zeros(2, np.float32)
        self.scale = torso_size(det["kp"]) or 0.2
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.t_last = t
        self._ingest(det, t, first=True)

    def predict(self, dt):
        self.centroid = self.centroid + self.vel * dt
        self.age += 1
        self.time_since_update += 1

    def update(self, det, t):
        self._ingest(det, t)
        self.hits += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"

    def _ingest(self, det, t, first=False):
        dt = max(1e-3, t - self.t_last) if not first else 1 / 12
        raw = det["kp"].astype(np.float32)
        new_c = _centroid(raw)
        shift = new_c - self.centroid if not first else np.zeros(2, np.float32)

        prev = self.kp if not first else None
        filled, inferred = recover(raw, prev, shift, self.bone_len)

        valid = filled[:, 2] > 0.15
        sm = self.filter(filled[:, :2], dt, valid)

        kp = np.zeros((N_KPT, 3), np.float32)
        kp[:, :2] = sm
        kp[:, 2] = filled[:, 2]
        kp[~valid, 2] = 0.0

        update_bone_lengths(self.bone_len, kp)
        self.kp = kp
        self.inferred = inferred
        sc = torso_size(kp)
        if sc:
            self.scale = 0.7 * self.scale + 0.3 * sc
        if not first:
            self.vel = 0.6 * self.vel + 0.4 * (new_c - self.centroid) / dt
        self.centroid = new_c
        self.t_last = t
        self.history.append((t, kp.copy()))


class MultiPersonTracker:
    def __init__(self, max_age=12, gate=0.14):
        self.tracks = []
        self.max_age = max_age
        self.gate = gate

    def update(self, detections, t, dt):
        dets = self._valid_detections(detections)

        for tr in self.tracks:
            tr.predict(dt)

        used_det = set()
        used_track = set()
        id_to_track = {tr.id: i for i, tr in enumerate(self.tracks)}
        for di, det in enumerate(dets):
            tid = det.get("track_id")
            if tid is None or int(tid) not in id_to_track:
                continue
            ti = id_to_track[int(tid)]
            self.tracks[ti].update(det, t)
            used_det.add(di)
            used_track.add(ti)

        remaining = [d for i, d in enumerate(dets) if i not in used_det]
        high = [d for d in remaining if d["score"] >= HIGH_CONF]
        low = [d for d in remaining if LOW_CONF <= d["score"] < HIGH_CONF]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_track]

        m1, ut, ud_high = self._match(self.tracks, high, unmatched_tracks)
        for ti, di in m1:
            self.tracks[ti].update(high[di], t)

        m2, _, _ = self._match(self.tracks, low, ut)
        for ti, di in m2:
            self.tracks[ti].update(low[di], t)

        for di in ud_high:
            self.tracks.append(Track(high[di], t))

        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]
        return [
            tr
            for tr in self.tracks
            if tr.state == "confirmed" and tr.time_since_update <= 2
        ]

    def _valid_detections(self, detections):
        dets = []
        for src in detections:
            kp = src["kp"]
            if anatomical_validity(kp) < 0.25:
                continue
            d = dict(src)
            d["centroid"] = _centroid(kp)
            d["score"] = float(src.get("score", kp[:, 2].mean()))
            if d.get("track_id") is not None:
                d["track_id"] = int(d["track_id"])
            dets.append(d)
        return dets

    def _match(self, tracks, dets, track_idxs):
        if not dets or not track_idxs:
            return [], track_idxs, list(range(len(dets)))

        big = 1e6
        cost = np.full((len(track_idxs), len(dets)), big, np.float32)
        for row, ti in enumerate(track_idxs):
            tr = tracks[ti]
            sc = max(tr.scale, 0.08)
            gate = self.gate + 2.0 * float(np.hypot(*tr.vel)) * (1.0 / 12)
            for di, det in enumerate(dets):
                norm = float(np.hypot(*(tr.centroid - det["centroid"])))
                if norm <= gate:
                    cost[row, di] = norm / sc

        matches, used_t, used_d = [], set(), set()
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(cost)
            for row, di in zip(rows, cols):
                if cost[row, di] < big:
                    ti = track_idxs[row]
                    used_t.add(ti)
                    used_d.add(di)
                    matches.append((ti, di))
        except ImportError:
            pairs = [
                (cost[row, di], track_idxs[row], di)
                for row in range(cost.shape[0])
                for di in range(cost.shape[1])
                if cost[row, di] < big
            ]
            pairs.sort()
            for _, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                matches.append((ti, di))

        unmatched_tracks = [ti for ti in track_idxs if ti not in used_t]
        unmatched_dets = [di for di in range(len(dets)) if di not in used_d]
        return matches, unmatched_tracks, unmatched_dets

