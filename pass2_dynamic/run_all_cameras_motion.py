#!/usr/bin/env python3
"""Run single-video motion tracking for all configured cameras.

This is a thin driver around SingleVideoMotionTracker that iterates over
all cameras listed in the config and runs pass2 tracking for each video.
"""

import sys
from pathlib import Path

# Make project utilities and pass2 modules importable
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from utils import load_config, setup_logger  # type: ignore
from pass2_dynamic.single_video_motion import SingleVideoMotionTracker  # type: ignore


def main() -> None:
    config = load_config()

    logger = setup_logger(
        name="RunAllCamerasMotion",
        log_dir=config["data"]["log_dir"],
        level=config["logging"]["level"],
        save_to_file=config["logging"]["save_logs"],
    )

    cameras = list(config["data"]["cameras"])
    logger.info("=== PASS 2: Per-camera motion tracking for all cameras ===")
    logger.info(f"Cameras: {cameras}")

    for cam_id in cameras:
        logger.info("\n--- Processing camera: %s ---", cam_id)
        tracker = SingleVideoMotionTracker(config, logger, camera_id=cam_id)
        tracker.track_video()

    logger.info("\n✓ Completed motion tracking for all cameras")


if __name__ == "__main__":
    main()
