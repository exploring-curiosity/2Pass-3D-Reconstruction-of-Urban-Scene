import json
import math
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]  # repo root = 2 levels up
PCD_PATH = ROOT / "outputs" / "pass1_static" / "dust3r_pointcloud.ply"
CAM_PATH = ROOT / "outputs" / "pass1_static" / "cameras.json"
OUT_CSV = ROOT / "experiments" / "results" / "static_metrics.csv"


# ---------- helpers for cameras.json ----------

def parse_cameras_json(path):
    """
    Parse cameras.json and return a list of camera records with:

        {
          "camera_id": str,
          "K": (3,3) np.array,
          "R": (3,3) np.array,
          "t": (3,)  np.array,
          "width": int,
          "height": int,
          "frame_idx": int or None
        }

    You MAY need to adjust this function depending on the actual
    format of your cameras.json. Open that file once and line up the
    keys with the code below.
    """
    with open(path, "r") as f:
        data = json.load(f)

    cams = []

    # ---- CASE 1: cameras.json is a dict of camera_id -> params ----
    if isinstance(data, dict) and "cameras" not in data:
        # e.g. { "s1-left_0000": {"K": [...], "R": [...], "t": [...], ...}, ... }
        for cam_name, params in data.items():
            K = np.array(params["K"], dtype=np.float32)
            R = np.array(params["R"], dtype=np.float32)
            t = np.array(params["t"], dtype=np.float32).reshape(3)

            width = int(params.get("width", 1920))
            height = int(params.get("height", 1080))

            # try to parse frame index from name, e.g. "s1-left_0000"
            frame_idx = None
            for token in cam_name.replace("-", "_").split("_"):
                if token.isdigit():
                    frame_idx = int(token)
            cams.append(
                dict(
                    camera_id=cam_name,
                    K=K,
                    R=R,
                    t=t,
                    width=width,
                    height=height,
                    frame_idx=frame_idx,
                )
            )

    # ---- CASE 2: cameras.json has an explicit "cameras" list ----
    else:
        # expect structure like {"cameras": [ { "camera_id": ..., ... }, ... ]}
        cam_list = data.get("cameras", [])
        for c in cam_list:
            K = np.array(c["K"], dtype=np.float32)
            R = np.array(c["R"], dtype=np.float32)
            t = np.array(c["t"], dtype=np.float32).reshape(3)
            width = int(c.get("width", 1920))
            height = int(c.get("height", 1080))
            frame_idx = c.get("frame_idx", None)

            cams.append(
                dict(
                    camera_id=c.get("camera_id", "cam"),
                    K=K,
                    R=R,
                    t=t,
                    width=width,
                    height=height,
                    frame_idx=frame_idx,
                )
            )

    if not cams:
        raise RuntimeError(
            "Could not parse any cameras from cameras.json. "
            "Open the file and adjust parse_cameras_json() to match its structure."
        )

    return cams


# ---------- 5.4.1 (a) point cloud density & ground coverage ----------

def compute_density_and_coverage(pcd, cell_size=0.5, min_pts_per_cell=10):
    pts = np.asarray(pcd.points)  # (N,3)

    # basic stats
    num_points = pts.shape[0]
    min_xyz = pts.min(axis=0)
    max_xyz = pts.max(axis=0)
    extents = max_xyz - min_xyz
    volume = float(extents[0] * extents[1] * extents[2])
    density = num_points / volume if volume > 0 else float("nan")

    # ground grid on x–y
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    # shift so min is 0
    x0 = x_coords.min()
    y0 = y_coords.min()
    x_idx = ((x_coords - x0) / cell_size).astype(int)
    y_idx = ((y_coords - y0) / cell_size).astype(int)

    grid = {}
    for xi, yi in zip(x_idx, y_idx):
        grid[(xi, yi)] = grid.get((xi, yi), 0) + 1

    num_cells = len(grid)
    num_covered = sum(1 for v in grid.values() if v >= min_pts_per_cell)
    coverage = num_covered / num_cells if num_cells > 0 else float("nan")

    return {
        "num_points": int(num_points),
        "volume_m3": float(volume),
        "density_pts_per_m3": float(density),
        "coverage_fraction": float(coverage),
    }


# ---------- 5.4.1 (b) reprojection metrics ----------

def project_points(K, R, t, points_world):
    """
    points_world: (M,3) np array in world coords
    returns: (M,2) pixel coords and (M,) depth values
    """
    # world -> camera
    Pc = (R @ points_world.T + t.reshape(3, 1))  # (3,M)
    z = Pc[2, :]
    # avoid divide-by-zero
    z_safe = np.where(z == 0, 1e-6, z)
    pts_cam_norm = Pc / z_safe
    pts_img = (K @ pts_cam_norm).T  # (M,3)
    u = pts_img[:, 0]
    v = pts_img[:, 1]
    return np.stack([u, v], axis=1), z


def compute_reprojection_metrics(pcd, cameras, num_sample_points=5000):
    pts = np.asarray(pcd.points)
    N = pts.shape[0]
    if N == 0:
        return {
            "mean_reproj_err_px": float("nan"),
            "p95_reproj_err_px": float("nan"),
            "mean_visible_fraction": float("nan"),
        }

    # sample subset of points
    idx = np.random.choice(N, size=min(num_sample_points, N), replace=False)
    sample_pts = pts[idx]

    all_errors = []
    visible_fractions = []

    for cam in cameras:
        K, R, t = cam["K"], cam["R"], cam["t"]
        width, height = cam["width"], cam["height"]

        pix, depth = project_points(K, R, t, sample_pts)
        u, v = pix[:, 0], pix[:, 1]

        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (depth > 0)
        visible_fraction = inside.sum() / len(sample_pts)
        visible_fractions.append(visible_fraction)

        # NOTE: we don't have original 2D keypoints, so we treat
        # "how far outside the image" as a pseudo-error.
        # For points outside, measure distance to nearest image border.
        err = np.zeros_like(u)
        # distance to image box in pixels
        err += np.where(u < 0, -u, 0)
        err += np.where(u >= width, u - (width - 1), 0)
        err += np.where(v < 0, -v, 0)
        err += np.where(v >= height, v - (height - 1), 0)

        all_errors.extend(np.abs(err))

    all_errors = np.array(all_errors)
    if len(all_errors) == 0:
        mean_err = float("nan")
        p95_err = float("nan")
    else:
        mean_err = float(all_errors.mean())
        p95_err = float(np.percentile(all_errors, 95))

    mean_visible_fraction = float(np.mean(visible_fractions)) if visible_fractions else float("nan")

    return {
        "mean_reproj_err_px": mean_err,
        "p95_reproj_err_px": p95_err,
        "mean_visible_fraction": mean_visible_fraction,
    }


# ---------- 5.4.1 (c) pose stability ----------

def rotation_matrix_to_euler(R):
    """
    Convert 3x3 rotation matrix to yaw-pitch-roll (Z-Y-X) in radians.
    We just need relative differences, so exact convention is not crucial.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z], dtype=np.float32)


def compute_pose_stability(cameras):
    """
    If cameras.json has multiple entries per physical camera (over time),
    measure how much the extrinsics wiggle.

    We group by camera_id with frame_idx stripped, e.g.
    "s1-left_0000", "s1-left_0010" -> "s1-left".
    """
    groups = {}
    for cam in cameras:
        name = cam["camera_id"]
        # strip trailing _###### if present
        base = name.split("_")[0]
        groups.setdefault(base, []).append(cam)

    trans_stds = []
    rot_stds = []

    for base_name, cams in groups.items():
        if len(cams) < 2:
            continue  # nothing to measure

        Ts = np.stack([c["t"] for c in cams], axis=0)  # (T,3)
        Rs_euler = np.stack([rotation_matrix_to_euler(c["R"]) for c in cams], axis=0)  # (T,3)

        trans_std = np.linalg.norm(Ts.std(axis=0))  # magnitude of std dev (m)
        rot_std_deg = float(np.linalg.norm(Rs_euler.std(axis=0)) * 180.0 / math.pi)  # degrees

        trans_stds.append(trans_std)
        rot_stds.append(rot_std_deg)

    if not trans_stds:
        return {
            "pose_std_trans_m": float("nan"),
            "pose_std_rot_deg": float("nan"),
        }

    return {
        "pose_std_trans_m": float(np.mean(trans_stds)),
        "pose_std_rot_deg": float(np.mean(rot_stds)),
    }


# ---------- main ----------

def main():
    print("Loading point cloud from", PCD_PATH)
    if not PCD_PATH.exists():
        raise FileNotFoundError(PCD_PATH)
    pcd = o3d.io.read_point_cloud(str(PCD_PATH))

    print("Loading cameras from", CAM_PATH)
    if not CAM_PATH.exists():
        raise FileNotFoundError(CAM_PATH)
    cameras = parse_cameras_json(CAM_PATH)

    print(f"Loaded {len(np.asarray(pcd.points))} points and {len(cameras)} cameras.")

    density_stats = compute_density_and_coverage(pcd)
    print("Density / coverage:", density_stats)

    reproj_stats = compute_reprojection_metrics(pcd, cameras)
    print("Reprojection stats:", reproj_stats)

    pose_stats = compute_pose_stability(cameras)
    print("Pose stability:", pose_stats)

    # combine everything into one row
    row = {}
    row.update(density_stats)
    row.update(reproj_stats)
    row.update(pose_stats)

    df = pd.DataFrame([row])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Saved metrics to", OUT_CSV)


if __name__ == "__main__":
    main()
