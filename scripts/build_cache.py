"""
Overnight batch: build the track cache for every clip in Videos/.

Resumable — clips whose tracks are already cached are skipped. Run from the repo root:

    python -m scripts.build_cache            # all clips, checkpoint-resumable
    python -m scripts.build_cache --force    # rebuild everything
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import time

from person_vehicle_interactions.config import (
    DetectionConfig,
    target_class_ids_for_clip,
)
from person_vehicle_interactions.tracker_engine import cache_clip_tracks

DEFAULT_VIDEOS_DIR = Path("Videos")


def main() -> None:
    """Build the track cache for every mp4 in the videos directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if already cached."
    )
    args = parser.parse_args()

    base_config = DetectionConfig()
    clips = sorted(args.videos_dir.glob("*.mp4"))
    print(f"Found {len(clips)} clip(s) in {args.videos_dir}", flush=True)

    for index, video_path in enumerate(clips, start=1):
        clip_id = video_path.stem
        # Per-clip keep-set so clips with detector quirks get their extra vehicle classes.
        config = dataclasses.replace(
            base_config, target_class_ids=target_class_ids_for_clip(clip_id)
        )
        print(f"[{index}/{len(clips)}] {clip_id} ...", flush=True)
        started = time.perf_counter()
        tracks_path, _ = cache_clip_tracks(
            video_path, clip_id, config, force=args.force
        )
        elapsed = time.perf_counter() - started
        print(f"    -> {tracks_path}  ({elapsed:.1f}s)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
