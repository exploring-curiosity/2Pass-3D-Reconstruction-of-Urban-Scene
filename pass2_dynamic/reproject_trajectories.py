#!/usr/bin/env python3
"""
Reproject 2D trajectories to 3D using Pi3 camera calibrations.

The original trajectories were computed using DUSt3R calibrations.
This script reprojects them using the corrected Pi3 cameras.
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, setup_logger


def load_pi3_cameras(cameras_path: Path) -> Dict:
    """Load corrected Pi3 cameras."""
    with open(cameras_path) as f:
        return json.load(f)


def estimate_ground_plane_z(points_ply_path: Path) -> float:
    """Estimate ground plane Z from point cloud."""
    from utils import load_ply
    
    points, _, _ = load_ply(str(points_ply_path))
    
    # Ground is approximately at Z=0 after correction
    # Use 5th percentile as ground level
    ground_z = np.percentile(points[:, 2], 5)
    
    return ground_z


def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates to normalized ray direction in camera frame."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Normalized image coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    
    # Ray direction (not normalized)
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


def ray_ground_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    ground_z: float = 0.0
) -> Optional[np.ndarray]:
    """Find intersection of ray with ground plane (Z = ground_z).
    
    Args:
        ray_origin: [3] ray origin in world coordinates
        ray_dir: [3] ray direction in world coordinates
        ground_z: Z coordinate of ground plane
        
    Returns:
        intersection: [3] intersection point or None if no intersection
    """
    # Ground plane: Z = ground_z
    # Ray: P = origin + t * dir
    # Intersection: origin[2] + t * dir[2] = ground_z
    
    if abs(ray_dir[2]) < 1e-6:
        return None  # Ray parallel to ground
    
    t = (ground_z - ray_origin[2]) / ray_dir[2]
    
    if t < 0:
        return None  # Intersection behind camera
    
    intersection = ray_origin + t * ray_dir
    return intersection


def reproject_detection_to_3d(
    center_px: List[float],
    K: np.ndarray,
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
    ground_z: float,
    original_width: int,
    original_height: int,
    calibration_width: int,
    calibration_height: int
) -> Optional[np.ndarray]:
    """Reproject a 2D detection center to 3D on ground plane.
    
    Args:
        center_px: [u, v] pixel coordinates in original image
        K: [3, 3] intrinsic matrix (for calibration resolution)
        R_c2w: [3, 3] camera-to-world rotation
        t_c2w: [3] camera position in world
        ground_z: Z coordinate of ground plane
        original_width/height: Original image resolution
        calibration_width/height: Resolution used for calibration
        
    Returns:
        position_3d: [3] world coordinates or None
    """
    # Scale pixel coordinates to calibration resolution
    scale_x = calibration_width / original_width
    scale_y = calibration_height / original_height
    
    u = center_px[0] * scale_x
    v = center_px[1] * scale_y
    
    # Get ray in camera frame
    ray_cam = pixel_to_ray(u, v, K)
    
    # Transform ray to world frame
    ray_world = R_c2w @ ray_cam
    
    # Camera position in world
    cam_pos = t_c2w
    
    # Find ground intersection
    intersection = ray_ground_intersection(cam_pos, ray_world, ground_z)
    
    return intersection


def reproject_trajectories(
    traj_path: Path,
    cameras: Dict,
    ground_z: float,
    original_width: int = 2592,
    original_height: int = 1944
) -> Dict:
    """Reproject all trajectories in a file to 3D."""
    
    with open(traj_path) as f:
        data = json.load(f)
    
    cam_id = data['camera_id']
    
    if cam_id not in cameras:
        print(f"Warning: Camera {cam_id} not found in Pi3 cameras")
        return data
    
    cam_data = cameras[cam_id]
    K = np.array(cam_data['K'], dtype=np.float32)
    pose_c2w = np.array(cam_data['pose_c2w'], dtype=np.float32)
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    
    cal_width = cam_data['width']
    cal_height = cam_data['height']
    
    # Reproject each trajectory
    for traj in data['trajectories']:
        for frame in traj['frames']:
            center_px = frame.get('center_px')
            
            if center_px is None:
                continue
            
            pos_3d = reproject_detection_to_3d(
                center_px, K, R_c2w, t_c2w, ground_z,
                original_width, original_height,
                cal_width, cal_height
            )
            
            if pos_3d is not None:
                frame['position_3d'] = pos_3d.tolist()
            else:
                frame['position_3d'] = None
    
    return data


def main():
    config = load_config()
    logger = setup_logger(
        name="ReprojectTrajectories",
        log_dir=config['data']['log_dir'],
        level="INFO",
        save_to_file=False
    )
    
    base_dir = Path(config['data']['output_dir'])
    pass1_dir = base_dir / "pass1_static"
    pass2_dir = base_dir / "pass2_dynamic"
    
    # Load Pi3 cameras
    cameras_path = pass1_dir / "pi3_cameras_corrected.json"
    if not cameras_path.exists():
        logger.error(f"Pi3 cameras not found: {cameras_path}")
        return
    
    cameras = load_pi3_cameras(cameras_path)
    logger.info(f"Loaded {len(cameras)} Pi3 cameras")
    
    # Estimate ground plane
    ply_path = pass1_dir / "pi3_pointcloud_corrected.ply"
    ground_z = estimate_ground_plane_z(ply_path)
    logger.info(f"Estimated ground Z: {ground_z:.2f}")
    
    # Find all trajectory files
    traj_files = list(pass2_dir.glob("*_trajectories.json"))
    logger.info(f"Found {len(traj_files)} trajectory files")
    
    # Reproject each file
    for traj_path in traj_files:
        logger.info(f"Processing {traj_path.name}...")
        
        data = reproject_trajectories(traj_path, cameras, ground_z)
        
        # Save reprojected trajectories
        output_path = pass2_dir / f"{traj_path.stem}_pi3.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Count valid 3D positions
        n_valid = sum(
            1 for traj in data['trajectories']
            for frame in traj['frames']
            if frame.get('position_3d') is not None
        )
        n_total = sum(len(traj['frames']) for traj in data['trajectories'])
        
        logger.info(f"  Reprojected {n_valid}/{n_total} detections")
    
    logger.info("\n=== Done! ===")
    logger.info(f"Reprojected trajectories saved to {pass2_dir}/*_pi3.json")


if __name__ == "__main__":
    main()
