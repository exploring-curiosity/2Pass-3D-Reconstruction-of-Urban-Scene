import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

# Repo root: 2 levels up from this file (experiments/scripts/...)
ROOT = Path(__file__).resolve().parents[2]

# Dynamic primitives (3D objects) – prefer objects_3d.json if exists
OUT_DIR = ROOT / "outputs" / "pass2_dynamic"
OBJ3D_PATH1 = OUT_DIR / "objects_3d" / "objects_3d.json"
OBJ3D_PATH2 = OUT_DIR / "multi_camera_primitives.json"

if OBJ3D_PATH1.exists():
    PRIMS_PATH = OBJ3D_PATH1
else:
    PRIMS_PATH = OBJ3D_PATH2

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMITIVE_CSV = RESULTS_DIR / "primitive_metrics.csv"

# Rough canonical sizes (meters)
CANONICAL = {
    "car":        {"length": 4.5, "width": 1.8, "height": 1.5},
    "truck":      {"length": 8.0, "width": 2.5, "height": 3.0},
    "bus":        {"length": 12.0, "width": 2.5, "height": 3.0},
    "bicycle":    {"length": 2.0, "width": 0.6, "height": 1.5},
    "motorcycle": {"length": 2.0, "width": 0.7, "height": 1.4},
    "person":     {"height": 1.7},
}


# ---------------------------------------------------------------------------
# Helper: load static point cloud (DUSt3R backbone)
# ---------------------------------------------------------------------------

def load_pointcloud() -> np.ndarray:
    """Load static DUSt3R point cloud from pass1_static."""
    import open3d as o3d

    pcd_path = ROOT / "outputs" / "pass1_static" / "dust3r_pointcloud.ply"
    if not pcd_path.exists():
        raise FileNotFoundError(f"Cannot find static point cloud: {pcd_path}")

    print(f"[info] Loading static point cloud from {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    print(f"[info] Loaded point cloud with {pts.shape[0]} points")
    return pts


# ---------------------------------------------------------------------------
# Helper: get bbox for one primitive
# ---------------------------------------------------------------------------

def get_bbox_for_primitive(obj: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return axis-aligned [min, max] box for a primitive.

    Priority:
      1. explicit bbox_3d.min / bbox_3d.max if present
      2. center + size_lwh (box)
      3. radius / height (cylinder → approximate as a box)
      4. fallback small cube (1m)
    """
    # 1) explicit bbox_3d
    bbox = obj.get("bbox_3d")
    if isinstance(bbox, dict) and "min" in bbox and "max" in bbox:
        bmin = np.array(bbox["min"], dtype=np.float32)
        bmax = np.array(bbox["max"], dtype=np.float32)
        return bmin, bmax

    center = np.array(obj.get("center", [0, 0, 0]), dtype=np.float32)

    # 2) size_lwh
    if "size_lwh" in obj:
        size = np.array(obj["size_lwh"], dtype=np.float32)

    # 3) cylinders (persons etc.)
    elif obj.get("type") == "cylinder":
        r = float(obj.get("radius", 0.4))
        h = float(obj.get("height", CANONICAL.get("person", {}).get("height", 1.7)))
        size = np.array([2 * r, 2 * r, h], dtype=np.float32)

    # 4) fallback
    else:
        size = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    half = size / 2.0
    bmin = center - half
    bmax = center + half
    return bmin, bmax


# ---------------------------------------------------------------------------
# Experiment 4: Primitive–point overlap & scale stats
# ---------------------------------------------------------------------------

def eval_primitive_scale(primitives: List[dict], pts: np.ndarray):
    """
    Exp. 4: Primitive–point overlap and scale calibration.
    For each class_name, compute:
      - mean/std inside_ratio
      - mean/std length/width/height
      - mean abs / relative error vs canonical length/height (if available).
    """
    # epsilon expansion around each primitive box to define "local" neighborhood
    eps = 0.5  # meters

    per_class_inside: Dict[str, List[float]] = {}
    per_class_len: Dict[str, List[float]] = {}
    per_class_width: Dict[str, List[float]] = {}
    per_class_height: Dict[str, List[float]] = {}

    for obj in primitives:
        cls = obj.get("class_name", "unknown")
        bmin, bmax = get_bbox_for_primitive(obj)

        # Expanded box for local neighborhood
        bmin_exp = bmin - eps
        bmax_exp = bmax + eps

        # mask for local neighborhood
        mask_local = np.all((pts >= bmin_exp) & (pts <= bmax_exp), axis=1)
        total_local = int(mask_local.sum())
        if total_local == 0:
            inside_ratio = np.nan
        else:
            mask_inside = np.all((pts >= bmin) & (pts <= bmax), axis=1)
            inside = int(mask_inside.sum())
            inside_ratio = inside / total_local if total_local > 0 else np.nan

        if not np.isnan(inside_ratio):
            per_class_inside.setdefault(cls, []).append(inside_ratio)

        # geometric size from bbox (length, width, height)
        size = np.abs(bmax - bmin)
        length, width, height = float(size[0]), float(size[1]), float(size[2])

        per_class_len.setdefault(cls, []).append(length)
        per_class_width.setdefault(cls, []).append(width)
        per_class_height.setdefault(cls, []).append(height)

    # Aggregate per class into rows
    rows = []
    all_classes = sorted(set(list(per_class_len.keys()) + list(per_class_inside.keys())))

    for cls in all_classes:
        inside_list = per_class_inside.get(cls, [])
        len_list = per_class_len.get(cls, [])
        width_list = per_class_width.get(cls, [])
        h_list = per_class_height.get(cls, [])

        mean_inside = float(np.mean(inside_list)) if inside_list else np.nan
        std_inside = float(np.std(inside_list)) if inside_list else np.nan

        mean_len = float(np.mean(len_list)) if len_list else np.nan
        std_len = float(np.std(len_list)) if len_list else np.nan

        mean_width = float(np.mean(width_list)) if width_list else np.nan
        std_width = float(np.std(width_list)) if width_list else np.nan

        mean_h = float(np.mean(h_list)) if h_list else np.nan
        std_h = float(np.std(h_list)) if h_list else np.nan

        canon = CANONICAL.get(cls, {})
        canon_len = canon.get("length")
        canon_h = canon.get("height")

        # length errors
        if canon_len is not None and len_list:
            len_abs_errs = [abs(x - canon_len) for x in len_list]
            mean_len_abs_err = float(np.mean(len_abs_errs))
            mean_len_rel_err = float(np.mean([e / canon_len for e in len_abs_errs]))
        else:
            mean_len_abs_err = np.nan
            mean_len_rel_err = np.nan

        # height errors
        if canon_h is not None and h_list:
            h_abs_errs = [abs(x - canon_h) for x in h_list]
            mean_h_abs_err = float(np.mean(h_abs_errs))
            mean_h_rel_err = float(np.mean([e / canon_h for e in h_abs_errs]))
        else:
            mean_h_abs_err = np.nan
            mean_h_rel_err = np.nan

        rows.append(
            {
                "class_name": cls,
                "num_primitives": len(len_list),
                "mean_inside_ratio": mean_inside,
                "std_inside_ratio": std_inside,
                "mean_length_m": mean_len,
                "std_length_m": std_len,
                "mean_width_m": mean_width,
                "std_width_m": std_width,
                "mean_height_m": mean_h,
                "std_height_m": std_h,
                "mean_length_abs_err_m": mean_len_abs_err,
                "mean_length_rel_err": mean_len_rel_err,
                "mean_height_abs_err_m": mean_h_abs_err,
                "mean_height_rel_err": mean_h_rel_err,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(PRIMITIVE_CSV, index=False)
    print(f"[info] Saved primitive/scale metrics to: {PRIMITIVE_CSV}")

    # Pretty console summary for the paper text
    print("\n=== Primitive Scale / Overlap Summary (Exp. 4) ===")
    for _, r in df.iterrows():
        cls = r["class_name"]
        n = int(r["num_primitives"])
        inside = r["mean_inside_ratio"]
        len_err = r["mean_length_abs_err_m"]
        h_err = r["mean_height_abs_err_m"]

        print(
            f"{cls:10s}  n={n:3d}  inside≈{inside:5.2f}  "
            f"|ΔL|≈{len_err if not np.isnan(len_err) else float('nan'):.2f} m  "
            f"|ΔH|≈{h_err if not np.isnan(h_err) else float('nan'):.2f} m"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PRIMS_PATH.exists():
        raise FileNotFoundError(f"Cannot find primitives file: {PRIMS_PATH}")

    print(f"[info] Loading primitives from {PRIMS_PATH}")
    with open(PRIMS_PATH, "r") as f:
        primitives = json.load(f)

    pts = load_pointcloud()
    eval_primitive_scale(primitives, pts)


if __name__ == "__main__":
    main()
