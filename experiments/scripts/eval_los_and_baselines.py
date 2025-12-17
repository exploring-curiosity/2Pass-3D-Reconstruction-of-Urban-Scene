# experiments/scripts/eval_los_and_baselines.py

import json
import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import open3d as o3d
import pandas as pd

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PCD_PATH = BASE_DIR / "outputs" / "pass1_static" / "dust3r_pointcloud.ply"
CAM_PATH = BASE_DIR / "outputs" / "pass1_static" / "cameras.json"

# Try a few reasonable locations / names for the 3D objects file
OBJ3D_CANDIDATES = [
    BASE_DIR / "outputs" / "pass2_dynamic" / "objects_3d" / "objects_3d.json",
    BASE_DIR / "outputs" / "pass2_dynamic" / "objects_3d.json",
    BASE_DIR / "outputs" / "pass2_dynamic" / "multicam" / "objects_3d.json",
    BASE_DIR / "outputs" / "pass2_dynamic" / "multi_camera_primitives.json",
]

RESULTS_DIR = BASE_DIR / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOS_CSV_PATH = RESULTS_DIR / "los_metrics.csv"

# FPS for the dynamic sequence (from tracking metadata)
FPS = 14.402503672478764


# -------------------------------------------------------------------------
# Utility: load static point cloud
# -------------------------------------------------------------------------

def load_static_point_cloud(voxel_size: float = 0.05) -> np.ndarray:
    """
    Load DUSt3R static point cloud and (optionally) voxel-downsample.

    Returns
    -------
    pts : (N, 3) ndarray in world coordinates.
    """
    if not PCD_PATH.exists():
        raise FileNotFoundError(f"Static point cloud not found at {PCD_PATH}")

    print(f"[info] Loading static point cloud from {PCD_PATH}")
    pcd = o3d.io.read_point_cloud(str(PCD_PATH))
    print(f"[info] Loaded {np.asarray(pcd.points).shape[0]} raw points")

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[info] Downsampled to {np.asarray(pcd.points).shape[0]} points "
              f"(voxel_size={voxel_size} m)")

    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


# -------------------------------------------------------------------------
# Utility: load cameras.json (intrinsics + extrinsics)
# -------------------------------------------------------------------------

def _extract_c2w(cam: Dict[str, Any]) -> np.ndarray:
    """Try to recover a 4x4 cam-to-world matrix from various key names."""
    if "c2w" in cam:
        mat = np.array(cam["c2w"], dtype=float).reshape(4, 4)
        return mat
    if "cam2world" in cam:
        mat = np.array(cam["cam2world"], dtype=float).reshape(4, 4)
        return mat
    if "w2c" in cam:
        mat = np.array(cam["w2c"], dtype=float).reshape(4, 4)
        return np.linalg.inv(mat)
    if "world2cam" in cam:
        mat = np.array(cam["world2cam"], dtype=float).reshape(4, 4)
        return np.linalg.inv(mat)
    if "extrinsics" in cam:
        mat = np.array(cam["extrinsics"], dtype=float).reshape(4, 4)
        # Heuristic: assume extrinsics is world->cam
        return np.linalg.inv(mat)

    raise ValueError("Could not find a cam-to-world or world-to-cam matrix "
                     f"for camera entry: {cam.keys()}")


def load_cameras() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load cameras.json and return a dict:
      camera_id -> {'R', 't', 'center'}
    where R, t are world->cam and center is camera center in world coords.
    Assumes DUSt3R-style entries with keys: K, R, t, width, height, camera_id, id.
    """
    if not CAM_PATH.exists():
        raise FileNotFoundError(f"cameras.json not found at {CAM_PATH}")

    print(f"[info] Loading cameras from {CAM_PATH}")
    with open(CAM_PATH, "r") as f:
        data = json.load(f)

    # cameras.json can be:
    #   - {"cameras": [ {...}, {...}, ... ]}
    #   - [ {...}, {...}, ... ]
    #   - { cam_id: {...}, ... }
    if isinstance(data, dict) and "cameras" in data:
        cam_list = data["cameras"]
    elif isinstance(data, list):
        cam_list = data
    else:
        # dict of {cam_id: entry}
        cam_list = []
        for cid, entry in data.items():
            e = dict(entry)
            e.setdefault("camera_id", cid)
            cam_list.append(e)

    cams: Dict[str, Dict[str, np.ndarray]] = {}

    for cam in cam_list:
        cam_id = cam.get("camera_id") or cam.get("id") or cam.get("name")
        if cam_id is None:
            continue

        if "R" not in cam or "t" not in cam:
            # Skip anything weird – but in your file every camera has R,t.
            print(f"[warn] Camera {cam_id} has no R/t, skipping")
            continue

        # DUSt3R stores R, t as world->cam
        R = np.asarray(cam["R"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(cam["t"], dtype=np.float32).reshape(3)

        # Camera center in world coordinates: C = -R^T t
        center = -R.T @ t

        cams[cam_id] = {
            "id": cam_id,
            "R": R,
            "t": t,
            "center": center.astype(np.float32),
        }

    print(f"[info] Loaded {len(cams)} cameras with valid centers")
    return cams



# -------------------------------------------------------------------------
# Precompute static points per camera (camera coordinates + angular ratios)
# -------------------------------------------------------------------------

def precompute_static_for_cameras(
    pts_world: np.ndarray,
    cameras: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """
    For each camera, attach:
        'pts_cam'  : (M, 3) points in that camera's frame (z>0),
        'x_over_z' : (M,),
        'y_over_z' : (M,)
    """
    for cam_id, cam in cameras.items():
        R = cam["R"]
        t = cam["t"]
        pts_cam = (R @ pts_world.T + t[:, None]).T  # (N, 3)
        # Keep only points in front of the camera
        mask = pts_cam[:, 2] > 0.0
        pts_cam = pts_cam[mask]

        cam["pts_cam"] = pts_cam
        cam["x_over_z"] = pts_cam[:, 0] / pts_cam[:, 2]
        cam["y_over_z"] = pts_cam[:, 1] / pts_cam[:, 2]

        print(f"[info] Camera {cam_id}: {pts_cam.shape[0]} static points after z>0")


# -------------------------------------------------------------------------
# Load pedestrian objects and approximate per-frame instances
# -------------------------------------------------------------------------

def load_objects_with_instances() -> List[Dict[str, Any]]:
    """
    Load multi-camera objects (or primitives) and build a list of
    pedestrian objects with synthetic per-frame instances.

    Each returned object has:
        {
          'id': int,
          'class_name': str,
          'category': str,
          'center': (3,),
          'members': [
              {
                 'camera_id': str,
                 'frames': [0, 1, ..., F-1],  # local frame indices
              }, ...
          ]
        }
    """
    obj_path = None
    for cand in OBJ3D_CANDIDATES:
        if cand.exists():
            obj_path = cand
            break

    if obj_path is None:
        raise FileNotFoundError(
            "objects_3d.json not found. Tried:\n" +
            "\n".join(str(p) for p in OBJ3D_CANDIDATES)
        )

    print(f"[info] Loading objects/primitives from {obj_path}")
    with open(obj_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "objects" in data:
        obj_list = data["objects"]
    elif isinstance(data, dict) and "primitives" in data:
        obj_list = data["primitives"]
    elif isinstance(data, list):
        obj_list = data
    else:
        raise ValueError(
            "Unrecognized objects_3d / multi_camera_primitives format. "
            "Expected list or dict with 'objects'/'primitives'."
        )

    ped_objects: List[Dict[str, Any]] = []
    for obj in obj_list:
        category = obj.get("category", obj.get("type", ""))
        class_name = obj.get("class_name", "")

        # We are interested in pedestrians
        if category != "person" and class_name != "person":
            continue

        obj_id = obj.get("id", obj.get("object_id", -1))
        center = np.array(obj.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)

        start_t = float(obj.get("start_time_sec", 0.0))
        end_t = float(obj.get("end_time_sec", start_t))
        duration = max(0.0, end_t - start_t)
        # approximate number of frames this object is alive
        n_frames = max(1, int(round(duration * FPS)))
        frames = list(range(n_frames))

        members = []
        for m in obj.get("members", []):
            cam_id = m.get("camera_id")
            if cam_id is None:
                continue
            members.append({
                "camera_id": cam_id,
                "frames": frames,  # same frame count per camera (approx.)
            })

        if not members:
            continue

        ped_objects.append({
            "id": obj_id,
            "class_name": class_name,
            "category": category,
            "center": center,
            "members": members,
        })

    print(f"[info] Found {len(ped_objects)} pedestrian objects with members")
    return ped_objects


# -------------------------------------------------------------------------
# Line-of-Sight check
# -------------------------------------------------------------------------

def is_occluded_for_camera(
    center_world: np.ndarray,
    cam: Dict[str, Any],
    delta_x: float = 0.02,
    delta_y: float = 0.02,
    depth_margin: float = 0.2,
) -> bool:
    """
    Simple ray-based occlusion test.

    Parameters
    ----------
    center_world : (3,) pedestrian center in world coords.
    cam          : camera dict with precomputed 'pts_cam', 'x_over_z', 'y_over_z'.

    Returns
    -------
    occluded : bool
        True if any static points lie in front of the pedestrian along the
        approximate viewing ray; False if clear LoS.
    """
    R = cam["R"]
    t = cam["t"]

    # transform pedestrian center into camera coordinates
    P_cam = R @ center_world + t
    if P_cam[2] <= 0:
        # behind camera – treat as not visible in usual sense
        return True

    px_over_z = P_cam[0] / P_cam[2]
    py_over_z = P_cam[1] / P_cam[2]

    pts_cam = cam["pts_cam"]
    x_over_z = cam["x_over_z"]
    y_over_z = cam["y_over_z"]

    # angular window around ray
    mask = (
        (np.abs(x_over_z - px_over_z) < delta_x) &
        (np.abs(y_over_z - py_over_z) < delta_y) &
        (pts_cam[:, 2] < P_cam[2] - depth_margin)
    )

    return bool(np.any(mask))


def eval_los():
    """
    Main LoS evaluation:
      - loads static cloud, cameras, pedestrian objects
      - computes per (object, camera) visibility curves
      - writes experiments/results/los_metrics.csv
    """
    # 1. Static cloud + cameras
    pts_world = load_static_point_cloud(voxel_size=0.05)
    cams = load_cameras()
    precompute_static_for_cameras(pts_world, cams)

    # 2. Pedestrian objects (multi-camera)
    ped_objs = load_objects_with_instances()

    rows = []
    for obj in ped_objs:
        obj_id = obj["id"]
        center = obj["center"]

        for mem in obj["members"]:
            cam_id = mem["camera_id"]
            frames = mem["frames"]

            if cam_id not in cams:
                continue

            cam = cams[cam_id]
            vis_curve: List[int] = []

            for _f in frames:
                occluded = is_occluded_for_camera(center, cam)
                vis_curve.append(0 if occluded else 1)

            vis_curve = np.array(vis_curve, dtype=np.int32)
            num_frames = len(vis_curve)

            if num_frames == 0:
                continue

            # First frame where visible==1
            visible_indices = np.where(vis_curve == 1)[0]
            if visible_indices.size > 0:
                first_visible_frame = int(visible_indices[0])
            else:
                first_visible_frame = -1  # never visible

            visibility_fraction = float(vis_curve.mean())

            # Count occlusion/visibility transitions (optional baseline metric)
            transitions = np.sum(vis_curve[1:] != vis_curve[:-1])
            num_occlusion_events = int(transitions // 2)  # approx.

            rows.append({
                "object_id": obj_id,
                "camera_id": cam_id,
                "num_frames": num_frames,
                "first_visible_frame": first_visible_frame,
                "visibility_fraction": visibility_fraction,
                "num_occlusion_events": num_occlusion_events,
            })

    df = pd.DataFrame(rows)
    df.to_csv(LOS_CSV_PATH, index=False)
    print(f"[info] Wrote LoS metrics to {LOS_CSV_PATH}")
    if not df.empty:
        print(df.describe())


# -------------------------------------------------------------------------
# Baselines stub (you can extend later for §5.4.6)
# -------------------------------------------------------------------------

def eval_baselines():
    """
    Placeholder for §5.4.6 baseline comparison logic.
    Currently just prints a message so the script doesn't error if called.
    """
    print("[info] Baseline metrics not implemented yet in this script.")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main():
    eval_los()
    # If/when you add baseline metrics, call eval_baselines() here too.
    # eval_baselines()


if __name__ == "__main__":
    main()
