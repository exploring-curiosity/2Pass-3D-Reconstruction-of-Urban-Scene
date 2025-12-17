import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root
OUT_DIR = ROOT / "outputs" / "pass2_dynamic"

# NOTE: we use multi_camera_primitives.json since objects_3d.json
# is not present in this repo.
PRIMS_PATH = OUT_DIR / "multi_camera_primitives.json"

# use one of the motion summary files to get fps
SUMMARY_PATH = OUT_DIR / "s1-left_motion_summary.json"

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RESULTS_DIR / "multicam_association.csv"

# how many overlapping frames to call "well aligned"
MIN_OVERLAP_FRAMES = 5


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_fps():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find motion summary file: {SUMMARY_PATH}")
    with open(SUMMARY_PATH, "r") as f:
        data = json.load(f)
    fps = float(data["fps"])
    print(f"[info] Using FPS = {fps:.4f} from {SUMMARY_PATH.name}")
    return fps


def load_primitives():
    """Load the multi-camera 3D primitives (objects)."""
    if not PRIMS_PATH.exists():
        raise FileNotFoundError(f"Cannot find multi-camera primitives file: {PRIMS_PATH}")

    with open(PRIMS_PATH, "r") as f:
        data = json.load(f)

    # handle both [ {...}, {...} ] and {"objects": [...]}
    if isinstance(data, dict):
        primitives = data.get("objects", [])
    else:
        primitives = data

    print(f"[info] Loaded {len(primitives)} 3D objects from {PRIMS_PATH.name}")
    if len(primitives) == 0:
        print("[warn] No objects found – multicam_association.csv would be empty.")
    return primitives


def compute_overlap_frames(cam_ranges, fps: float) -> int:
    """
    cam_ranges: dict[camera_id] -> (start_sec, end_sec)
    Returns the intersection length in frames (integer, >=0).
    """
    if len(cam_ranges) < 2:
        return 0

    max_start = max(s for (s, _) in cam_ranges.values())
    min_end = min(e for (_, e) in cam_ranges.values())

    overlap_sec = max(0.0, min_end - max_start)
    overlap_frames = int(round(overlap_sec * fps))
    return overlap_frames


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    fps = load_fps()
    primitives = load_primitives()

    rows = []

    num_objects = len(primitives)
    num_ge2 = 0
    num_ge3 = 0
    num_well_aligned = 0

    for obj in primitives:
        obj_id = obj.get("id")
        cls = obj.get("class_name", "unknown")
        cat = obj.get("category", "unknown")
        members = obj.get("members", [])

        # aggregate time ranges per camera for this 3D object
        cam_ranges = {}  # camera_id -> [start_sec, end_sec]

        for m in members:
            cid = m["camera_id"]
            s = float(m.get("start_time_sec", obj.get("start_time_sec", 0.0)))
            e = float(m.get("end_time_sec", obj.get("end_time_sec", s)))

            if cid not in cam_ranges:
                cam_ranges[cid] = [s, e]
            else:
                cam_ranges[cid][0] = min(cam_ranges[cid][0], s)
                cam_ranges[cid][1] = max(cam_ranges[cid][1], e)

        num_views = len(cam_ranges)
        overlap_frames = compute_overlap_frames(cam_ranges, fps)
        well_aligned = int(num_views >= 2 and overlap_frames >= MIN_OVERLAP_FRAMES)

        if num_views >= 2:
            num_ge2 += 1
            if num_views >= 3:
                num_ge3 += 1
            if well_aligned:
                num_well_aligned += 1

        rows.append(
            {
                "object_id": obj_id,
                "class_name": cls,
                "category": cat,
                "num_views": num_views,
                "overlap_frames": overlap_frames,
                "well_aligned": well_aligned,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[info] Saved per-object metrics to: {OUT_CSV}")

    # -----------------------------------------------------------------
    # Global summary (for the paper write-up)
    # -----------------------------------------------------------------
    if num_objects > 0:
        median_views = df["num_views"].median()
        max_views = df["num_views"].max()
        frac_ge2 = num_ge2 / num_objects
        frac_ge3 = num_ge3 / num_objects
        frac_well = num_well_aligned / max(num_ge2, 1)

        print("\n=== Multi-Camera Association Summary ===")
        print(f"Total 3D objects      : {num_objects}")
        print(f"Objects with ≥2 views : {num_ge2} ({frac_ge2:.3f})")
        print(f"Objects with ≥3 views : {num_ge3} ({frac_ge3:.3f})")
        print(f"Median #views         : {median_views:.2f}")
        print(f"Max #views            : {max_views}")
        print(
            f"Well-aligned (≥{MIN_OVERLAP_FRAMES} frame overlap) among ≥2-view objects: "
            f"{num_well_aligned}/{num_ge2} ({frac_well:.3f})"
        )

        # suggest a small random subset of multi-view objects to inspect manually
        multi_view_ids = df[df["num_views"] >= 2]["object_id"].tolist()
        if multi_view_ids:
            sample_size = min(30, len(multi_view_ids))
            sample_ids = random.sample(multi_view_ids, sample_size)
            print("\nSample object IDs (num_views ≥ 2) for manual visual inspection:")
            print(sample_ids)
            print(
                "\nUse these IDs with your visualizer (e.g. visualize_tracking.py) "
                "to check whether tracks are correctly merged across cameras."
            )


if __name__ == "__main__":
    main()
