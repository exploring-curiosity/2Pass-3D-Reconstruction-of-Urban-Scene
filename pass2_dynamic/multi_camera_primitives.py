#!/usr/bin/env python3
"""Multi-camera primitive representation for dynamic objects.

This script:
  1. Loads per-camera motion summaries produced by single_video_motion.py
  2. Associates tracks across cameras in 3D + time into global objects
  3. Assigns canonical 3D primitives:
       - Vehicles  -> axis-aligned boxes with class-specific sizes
       - Persons   -> vertical cylinders with canonical radius/height
  4. Writes:
       - multi_camera_primitives.json with primitive parameters
       - multi_camera_primitives.ply with sampled points for visualization

All coordinates are expressed in the same world frame as pass1_static
(dust3r_pointcloud.ply), since per-camera tracking already projects
positions to that frame.
"""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Make project utilities importable
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from utils import load_config, setup_logger, save_ply, load_json  # type: ignore


VEHICLE_BOX_SIZES = {
    "car": (4.5, 1.8, 1.5),        # L, W, H in meters (approx sedan)
    "truck": (8.0, 2.5, 3.0),      # generic box truck
    "bus": (10.0, 2.5, 3.2),       # city bus
    "motorcycle": (2.0, 0.8, 1.2),
    "bicycle": (2.0, 0.6, 1.5),
}

DEFAULT_VEHICLE_BOX = (4.0, 1.7, 1.5)

PERSON_CYLINDER_RADIUS = 0.4
PERSON_CYLINDER_HEIGHT = 1.7


@dataclass
class TrackSummary:
    camera_id: str
    track_id: int
    class_name: str
    category: str
    is_stationary: bool
    avg_pos3d: np.ndarray  # [3]
    start_time: float
    end_time: float


def _load_all_tracks(config, logger) -> List[TrackSummary]:
    """Load per-camera motion summaries and collect usable tracks.

    We require avg_position_3d to be present so that tracks can be
    associated across cameras in a shared 3D frame.
    """
    output_root = Path(config["data"]["output_dir"]) / "pass2_dynamic"
    cameras = list(config["data"]["cameras"])

    tracks: List[TrackSummary] = []

    for cam_id in cameras:
        summary_path = output_root / f"{cam_id}_motion_summary.json"
        if not summary_path.exists():
            logger.warning("Motion summary not found for %s at %s", cam_id, summary_path)
            continue

        try:
            summary = load_json(str(summary_path))
        except Exception as e:  # pragma: no cover - debug path
            logger.error("Failed to load %s: %s", summary_path, e)
            continue

        for t in summary.get("tracks", []):
            avg_pos = t.get("avg_position_3d", None)
            if avg_pos is None:
                continue

            category = t.get("category", "other")
            if category not in ("person", "vehicle"):
                continue

            try:
                pos3d = np.asarray(avg_pos, dtype=float).reshape(3,)
            except Exception:
                continue

            start_time = float(t.get("start_time_sec", 0.0))
            end_time = float(t.get("end_time_sec", start_time))

            tracks.append(
                TrackSummary(
                    camera_id=summary.get("camera_id", cam_id),
                    track_id=int(t.get("track_id", -1)),
                    class_name=str(t.get("class_name", "unknown")),
                    category=category,
                    is_stationary=bool(t.get("is_stationary", False)),
                    avg_pos3d=pos3d,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    logger.info("Loaded %d tracks with 3D positions from %d cameras", len(tracks), len(cameras))
    return tracks


def _associate_tracks(tracks: List[TrackSummary]) -> List[List[int]]:
    """Associate tracks across cameras in 3D + time.

    Returns a list of clusters, each a list of indices into `tracks`.
    """
    n = len(tracks)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Thresholds (meters and seconds)
    dist_thresh_vehicle = 2.0
    dist_thresh_person = 1.0
    min_time_overlap = 0.3
    min_overlap_ratio = 0.3

    for i in range(n):
        ti = tracks[i]
        for j in range(i + 1, n):
            tj = tracks[j]

            if ti.category != tj.category:
                continue

            # 3D distance between average positions
            d = float(np.linalg.norm(ti.avg_pos3d - tj.avg_pos3d))
            if ti.category == "vehicle":
                if d > dist_thresh_vehicle:
                    continue
            else:  # person
                if d > dist_thresh_person:
                    continue

            # Temporal overlap
            start = max(ti.start_time, tj.start_time)
            end = min(ti.end_time, tj.end_time)
            overlap = max(0.0, end - start)
            if overlap < min_time_overlap:
                continue

            dur_i = max(1e-3, ti.end_time - ti.start_time)
            dur_j = max(1e-3, tj.end_time - tj.start_time)
            if overlap / min(dur_i, dur_j) < min_overlap_ratio:
                continue

            union(i, j)

    # Build clusters
    clusters_dict: Dict[int, List[int]] = {}
    for idx in range(n):
        r = find(idx)
        clusters_dict.setdefault(r, []).append(idx)

    return list(clusters_dict.values())


def _canonical_vehicle_box(class_name: str) -> np.ndarray:
    size = VEHICLE_BOX_SIZES.get(class_name, DEFAULT_VEHICLE_BOX)
    return np.asarray(size, dtype=float)


def _sample_box_points(center: np.ndarray, size_lwh: np.ndarray, num_points_per_dim=(4, 4, 3)) -> np.ndarray:
    """Sample a regular grid of points inside an axis-aligned box.

    center: [x, y, z] of box center
    size_lwh: [L, W, H]
    """
    L, W, H = size_lwh
    nx, ny, nz = num_points_per_dim

    xs = np.linspace(-L / 2.0, L / 2.0, nx)
    ys = np.linspace(-W / 2.0, W / 2.0, ny)
    zs = np.linspace(0.0, H, nz)  # base at z=0, height H

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="xy"), axis=-1).reshape(-1, 3)

    # Interpret center as ground-contact center: shift box so that its base
    # rests on the ground at center[2].
    world_center = np.asarray(center, dtype=float).reshape(3,)
    base = world_center.copy()
    base[2] = world_center[2]

    grid[:, 2] += base[2]
    grid[:, 0] += base[0]
    grid[:, 1] += base[1]
    return grid


def _sample_cylinder_points(center: np.ndarray, radius: float, height: float,
                            num_theta: int = 16, num_z: int = 6) -> np.ndarray:
    """Sample points on the surface of a vertical cylinder."""
    thetas = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    zs = np.linspace(0.0, height, num_z)

    pts = []
    for z in zs:
        for th in thetas:
            x = radius * np.cos(th)
            y = radius * np.sin(th)
            pts.append([x, y, z])

    pts = np.asarray(pts, dtype=float)

    world_center = np.asarray(center, dtype=float).reshape(3,)
    base = world_center.copy()
    base[2] = world_center[2]

    pts[:, 2] += base[2]
    pts[:, 0] += base[0]
    pts[:, 1] += base[1]
    return pts


def run_multi_camera_primitives() -> None:
    config = load_config()
    logger = setup_logger(
        name="MultiCameraPrimitives",
        log_dir=config["data"]["log_dir"],
        level=config["logging"]["level"],
        save_to_file=config["logging"]["save_logs"],
    )

    logger.info("=== PASS 2: Multi-camera primitive generation ===")

    tracks = _load_all_tracks(config, logger)
    if not tracks:
        logger.error("No tracks with 3D positions available; run per-camera motion tracking first")
        return

    clusters = _associate_tracks(tracks)
    logger.info("Formed %d multi-camera clusters", len(clusters))

    primitives: List[Dict[str, Any]] = []
    all_points = []
    all_colors = []

    for obj_id, inds in enumerate(clusters):
        if not inds:
            continue

        members = [tracks[i] for i in inds]
        categories = {m.category for m in members}
        if len(categories) != 1:
            # Mixed categories; skip for safety
            continue

        category = members[0].category

        # Aggregate attributes
        pos_stack = np.stack([m.avg_pos3d for m in members], axis=0)
        center = pos_stack.mean(axis=0)

        is_stationary = bool(np.mean([1.0 if m.is_stationary else 0.0 for m in members]) >= 0.5)

        start_time = float(min(m.start_time for m in members))
        end_time = float(max(m.end_time for m in members))

        class_names = [m.class_name for m in members]
        # Choose the most frequent class name within the cluster
        class_name = max(set(class_names), key=class_names.count)

        member_meta = [
            {
                "camera_id": m.camera_id,
                "track_id": int(m.track_id),
                "class_name": m.class_name,
                "category": m.category,
                "is_stationary": bool(m.is_stationary),
                "start_time_sec": float(m.start_time),
                "end_time_sec": float(m.end_time),
            }
            for m in members
        ]

        if category == "vehicle":
            prim_type = "box"
            size = _canonical_vehicle_box(class_name)
            pts = _sample_box_points(center, size)
            color = np.array([255, 0, 0], dtype=np.uint8)

            prim = {
                "id": int(obj_id),
                "type": prim_type,
                "class_name": class_name,
                "category": category,
                "center": center.tolist(),
                "size_lwh": size.tolist(),
                "orientation_quat": [0.0, 0.0, 0.0, 1.0],  # identity (axis-aligned)
                "is_stationary": is_stationary,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "members": member_meta,
            }

        elif category == "person":
            prim_type = "cylinder"
            radius = PERSON_CYLINDER_RADIUS
            height = PERSON_CYLINDER_HEIGHT
            pts = _sample_cylinder_points(center, radius, height)
            color = np.array([0, 255, 0], dtype=np.uint8)

            prim = {
                "id": int(obj_id),
                "type": prim_type,
                "class_name": class_name,
                "category": category,
                "center": center.tolist(),
                "radius": float(radius),
                "height": float(height),
                "axis": [0.0, 0.0, 1.0],
                "is_stationary": is_stationary,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "members": member_meta,
            }

        else:
            # Should not happen because we filtered earlier
            continue

        primitives.append(prim)
        all_points.append(pts.astype(np.float32))
        all_colors.append(np.tile(color[None, :], (pts.shape[0], 1)))

    output_root = Path(config["data"]["output_dir"]) / "pass2_dynamic"
    output_root.mkdir(parents=True, exist_ok=True)

    prim_json_path = output_root / "multi_camera_primitives.json"
    with open(prim_json_path, "w") as f:
        json.dump(primitives, f, indent=2)

    if all_points:
        pts_all = np.concatenate(all_points, axis=0)
        cols_all = np.concatenate(all_colors, axis=0)
        prim_ply_path = output_root / "multi_camera_primitives.ply"
        save_ply(str(prim_ply_path), pts_all, cols_all.astype(np.uint8))
        logger.info(
            "Saved %d primitives and %d sampled points to %s and %s",
            len(primitives),
            pts_all.shape[0],
            prim_json_path,
            prim_ply_path,
        )
    else:
        logger.warning("No primitives were created; multi_camera_primitives.json is empty")


def main() -> None:
    run_multi_camera_primitives()


if __name__ == "__main__":
    main()
