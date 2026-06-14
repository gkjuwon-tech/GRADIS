#!/usr/bin/env python3
"""
Fetch the multi-angle top-view benchmark clips from GitHub.

The raw CCTV clips are too large to commit, but the benchmark must be
reproducible WITHOUT anyone manually uploading a zip. This script pulls a small,
curated set of CASIA action clips (re-hosted in the public SPHAR-Dataset repo)
straight from raw.githubusercontent.com into ``_media/sphar_bench/``.

Why these clips: CASIA filmed the SAME actor doing the SAME action from three
camera angles — horizontal, oblique ("angle"), and top-down. That makes them a
direct test of the completion criterion: does the mannequin overlay survive and
preserve behavior from ANY angle? It includes falling (faint) and fighting,
which are exactly the behaviors anonymization must not destroy.

Usage:
  python tools/fetch_bench_videos.py            # fetch the default curated set
  python tools/fetch_bench_videos.py --list     # just print what would be fetched
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/AlexanderMelde/SPHAR-Dataset/master/videos"

# (source path in repo) -> kept as flattened name in _media/sphar_bench
CLIPS = [
    # falling / faint — full 3-angle trio (the hardest, most important behavior)
    "falling/casia_topdownview_p01_faint_a1.mp4",
    "falling/casia_angleview_p01_faint_a1.mp4",
    "falling/casia_horizontalview_p01_faint_a1.mp4",
    # fighting — two people, top-down + oblique
    "hitting/casia_topdownview_p01p02_fight_a1.mp4",
    "hitting/casia_angleview_p01p02_fight_a1.mp4",
    # walking / running — the everyday baseline, oblique + horizontal
    "walking/casia_angleview_p01_walk_a1.mp4",
    "walking/casia_horizontalview_p01_walk_a1.mp4",
    "running/casia_angleview_p01_run_a1.mp4",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "_media", "sphar_bench"))
    ap.add_argument("--list", action="store_true", help="print URLs and exit")
    args = ap.parse_args()

    if args.list:
        for c in CLIPS:
            print(f"{RAW_BASE}/{c}")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"[fetch] -> {args.out}")
    ok = 0
    for c in CLIPS:
        name = c.replace("/", "_")
        dst = os.path.join(args.out, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            print(f"  have  {name} ({os.path.getsize(dst)} B)")
            ok += 1
            continue
        url = f"{RAW_BASE}/{c}"
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  got   {name} ({os.path.getsize(dst)} B)")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}", file=sys.stderr)
    print(f"[fetch] {ok}/{len(CLIPS)} clips ready in {args.out}")
    if ok < len(CLIPS):
        sys.exit(1)


if __name__ == "__main__":
    main()
