import json
import csv
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]   # repo root
SUMMARY_DIR = ROOT / "outputs" / "pass2_dynamic"
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RESULTS_DIR / "per_camera_tracking.csv"


def safe_mean(xs):
    xs = list(xs)
    return float(mean(xs)) if xs else 0.0


def safe_median(xs):
    xs = list(xs)
    return float(median(xs)) if xs else 0.0


def process_camera(summary_path: Path):
    with open(summary_path, "r") as f:
        data = json.load(f)

    camera_id = data.get("camera_id", summary_path.stem.replace("_motion_summary", ""))
    fps = float(data.get("fps", 0.0))
    total_frames = int(data.get("total_frames", 0))

    tracks = data.get("tracks", [])
    num_tracks = len(tracks)

    # Basic track lengths
    lengths_frames = [int(t.get("num_detections", 0)) for t in tracks]
    lengths_sec = [float(t.get("duration_sec", 0.0)) for t in tracks]

    # Detection completeness proxy:
    # completeness = (#detections) / (fps * duration_sec)
    completeness_list = []
    for t in tracks:
        num_det = float(t.get("num_detections", 0))
        dur = float(t.get("duration_sec", 0.0))
        if fps > 0 and dur > 0:
            ideal_frames = fps * dur
            if ideal_frames > 0:
                completeness = max(0.0, min(1.0, num_det / ideal_frames))
                completeness_list.append(completeness)

    # Stationary / moving, per category
    moving_counts = data.get("counts", {}).get("moving", {})
    stationary_counts = data.get("counts", {}).get("stationary", {})

    moving_person = int(moving_counts.get("person", 0))
    moving_vehicle = int(moving_counts.get("vehicle", 0))
    moving_other = int(moving_counts.get("other", 0))

    stationary_person = int(stationary_counts.get("person", 0))
    stationary_vehicle = int(stationary_counts.get("vehicle", 0))
    stationary_other = int(stationary_counts.get("other", 0))

    total_labeled_tracks = (
        moving_person
        + moving_vehicle
        + moving_other
        + stationary_person
        + stationary_vehicle
        + stationary_other
    )

    if total_labeled_tracks > 0:
        stationary_fraction_overall = (
            stationary_person + stationary_vehicle + stationary_other
        ) / total_labeled_tracks
    else:
        stationary_fraction_overall = 0.0

    # Fractions by category (vehicles vs people)
    vehicle_total = moving_vehicle + stationary_vehicle
    person_total = moving_person + stationary_person

    stationary_fraction_vehicle = (
        stationary_vehicle / vehicle_total if vehicle_total > 0 else 0.0
    )
    stationary_fraction_person = (
        stationary_person / person_total if person_total > 0 else 0.0
    )

    row = {
        "camera_id": camera_id,
        "fps": fps,
        "total_frames": total_frames,
        "num_tracks": num_tracks,
        "mean_track_len_frames": safe_mean(lengths_frames),
        "median_track_len_frames": safe_median(lengths_frames),
        "mean_track_len_sec": safe_mean(lengths_sec),
        "median_track_len_sec": safe_median(lengths_sec),
        "mean_completeness": safe_mean(completeness_list),
        "stationary_fraction_overall": stationary_fraction_overall,
        "stationary_fraction_vehicle": stationary_fraction_vehicle,
        "stationary_fraction_person": stationary_fraction_person,
        "moving_person": moving_person,
        "stationary_person": stationary_person,
        "moving_vehicle": moving_vehicle,
        "stationary_vehicle": stationary_vehicle,
        "moving_other": moving_other,
        "stationary_other": stationary_other,
    }

    return row


def main():
    summary_files = sorted(SUMMARY_DIR.glob("*_motion_summary.json"))

    if not summary_files:
        print(f"[ERROR] No *_motion_summary.json files found in {SUMMARY_DIR}")
        return

    print(f"Found {len(summary_files)} motion summary files:")
    for p in summary_files:
        print("  -", p.name)

    rows = []
    for p in summary_files:
        print(f"\nProcessing {p.name} ...")
        row = process_camera(p)
        rows.append(row)
        print(
            f"  camera={row['camera_id']} | "
            f"tracks={row['num_tracks']} | "
            f"mean_len_frames={row['mean_track_len_frames']:.1f} | "
            f"mean_completeness={row['mean_completeness']:.2f} | "
            f"stationary_fraction={row['stationary_fraction_overall']:.2f}"
        )

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Wrote per-camera tracking metrics to {OUT_CSV}")


if __name__ == "__main__":
    main()
