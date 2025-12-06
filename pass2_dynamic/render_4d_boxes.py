#!/usr/bin/env python3
"""
4D Scene Renderer with Solid 3D Boxes for Dynamic Objects.

This script renders the static scene (from Pi3) with dynamic objects
represented as solid 3D boxes, not point clouds.

Uses PyTorch3D or Open3D for proper 3D rendering.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_config, setup_logger, load_ply


# Canonical sizes for different object types (in meters)
VEHICLE_SIZES = {
    'car': (4.5, 1.8, 1.5),      # length, width, height
    'truck': (8.0, 2.5, 3.0),
    'bus': (12.0, 2.5, 3.5),
    'motorcycle': (2.2, 0.8, 1.2),
    'bicycle': (1.8, 0.6, 1.1),
}

PERSON_SIZE = (0.5, 0.5, 1.7)  # width, depth, height


@dataclass
class Track:
    """A tracked object across frames."""
    track_id: int
    class_name: str
    category: str
    is_stationary: bool
    frames: Dict[int, np.ndarray]  # frame_idx -> 3D position


@dataclass
class Box3D:
    """A 3D bounding box."""
    center: np.ndarray  # [3]
    size: np.ndarray    # [3] (length, width, height)
    rotation: float     # yaw angle in radians
    color: Tuple[int, int, int]  # RGB


def load_pi3_data(output_dir: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load corrected Pi3 point cloud and cameras."""
    
    ply_path = output_dir / "pi3_pointcloud_corrected.ply"
    cameras_path = output_dir / "pi3_cameras_corrected.json"
    transform_path = output_dir / "pi3_transform.json"
    
    if not ply_path.exists():
        raise FileNotFoundError(f"Corrected point cloud not found: {ply_path}")
    
    points, colors, _ = load_ply(str(ply_path))
    
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    with open(transform_path) as f:
        transform = json.load(f)
    
    return points, colors, cameras, transform


def load_trajectories(pass2_dir: Path, transform: dict, 
                      min_track_length: int = 10,
                      scene_bounds: Tuple[float, float] = (-30, 30)) -> List[Track]:
    """Load Pi3-reprojected trajectories with filtering.
    
    Args:
        pass2_dir: Directory with trajectory files
        transform: Transform parameters (for fallback)
        min_track_length: Minimum number of frames for a valid track
        scene_bounds: (min, max) bounds for X and Y coordinates
    """
    
    tracks = {}
    
    # Find Pi3-reprojected trajectory files
    traj_files = list(pass2_dir.glob("*_trajectories_pi3.json"))
    
    if not traj_files:
        print("Warning: No Pi3 trajectory files found, using original with transform")
        traj_files = list(pass2_dir.glob("*_trajectories.json"))
        use_transform = True
        center = np.array(transform['center'])
        R_total = np.array(transform['R_total'])
        scale = transform['scale']
        z_offset = transform['z_offset']
    else:
        use_transform = False
    
    for traj_file in traj_files:
        with open(traj_file) as f:
            data = json.load(f)
        
        cam_id = data['camera_id']
        
        for traj in data['trajectories']:
            track_id = traj['track_id']
            
            # Skip short tracks (likely noise)
            if traj['num_frames'] < min_track_length:
                continue
            
            # Create or get track
            if track_id not in tracks:
                tracks[track_id] = Track(
                    track_id=track_id,
                    class_name=traj['class_name'],
                    category=traj['category'],
                    is_stationary=traj['is_stationary'],
                    frames={}
                )
            
            track = tracks[track_id]
            
            # Add frame positions
            for frame_data in traj['frames']:
                frame_idx = frame_data['frame_idx']
                pos_3d = frame_data.get('position_3d')
                
                if pos_3d is None:
                    continue
                
                pos = np.array(pos_3d)
                
                # Apply transform only if using original files
                if use_transform:
                    pos_centered = pos - center
                    pos_rotated = R_total @ pos_centered
                    pos = pos_rotated * scale
                    pos[2] -= z_offset
                
                # Filter out-of-bounds positions
                if not (scene_bounds[0] <= pos[0] <= scene_bounds[1] and
                        scene_bounds[0] <= pos[1] <= scene_bounds[1]):
                    continue
                
                # Store position
                if frame_idx not in track.frames:
                    track.frames[frame_idx] = pos
    
    # Filter tracks with too few valid positions
    valid_tracks = [t for t in tracks.values() if len(t.frames) >= min_track_length]
    
    return valid_tracks


def get_box_size(class_name: str, category: str) -> np.ndarray:
    """Get canonical box size for object class."""
    if category == 'vehicle':
        size = VEHICLE_SIZES.get(class_name.lower(), VEHICLE_SIZES['car'])
    else:
        size = PERSON_SIZE
    return np.array(size)


def get_box_color(category: str, is_stationary: bool) -> Tuple[int, int, int]:
    """Get color for object based on type and motion state."""
    if category == 'vehicle':
        if is_stationary:
            return (255, 165, 0)  # Orange
        else:
            return (255, 0, 0)    # Red
    else:  # person
        if is_stationary:
            return (0, 255, 255)  # Cyan
        else:
            return (0, 255, 0)    # Green


def create_box_vertices(center: np.ndarray, size: np.ndarray, 
                        yaw: float = 0.0) -> np.ndarray:
    """Create 8 vertices of a 3D box.
    
    Args:
        center: [3] center position
        size: [3] (length, width, height)
        yaw: rotation around Z axis
        
    Returns:
        vertices: [8, 3] box corners
    """
    l, w, h = size / 2
    
    # Box corners in local frame (centered at origin)
    corners = np.array([
        [-l, -w, 0],
        [+l, -w, 0],
        [+l, +w, 0],
        [-l, +w, 0],
        [-l, -w, h*2],
        [+l, -w, h*2],
        [+l, +w, h*2],
        [-l, +w, h*2],
    ])
    
    # Apply yaw rotation
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])
    
    corners = (R @ corners.T).T
    
    # Translate to center (bottom of box at center height)
    corners += center
    
    return corners


def project_box_to_image(vertices: np.ndarray, K: np.ndarray, 
                         R: np.ndarray, t: np.ndarray,
                         width: int, height: int) -> Optional[np.ndarray]:
    """Project 3D box vertices to 2D image coordinates.
    
    Returns:
        projected: [8, 2] 2D coordinates or None if box is behind camera
    """
    # Transform to camera coordinates
    pts_cam = (R @ vertices.T).T + t
    
    # Check if any point is in front of camera
    if np.all(pts_cam[:, 2] <= 0.1):
        return None
    
    # Project to image
    pts_proj = (K @ pts_cam.T).T
    
    # Handle points behind camera
    valid = pts_cam[:, 2] > 0.1
    pts_2d = np.zeros((8, 2))
    pts_2d[valid] = pts_proj[valid, :2] / pts_proj[valid, 2:3]
    
    # Clamp invalid points
    pts_2d[~valid] = np.nan
    
    return pts_2d


def draw_box_3d(img: np.ndarray, vertices_2d: np.ndarray, 
                color: Tuple[int, int, int], thickness: int = 2):
    """Draw a 3D box on the image.
    
    Box edges:
    - Bottom face: 0-1-2-3-0
    - Top face: 4-5-6-7-4
    - Vertical edges: 0-4, 1-5, 2-6, 3-7
    """
    # Define edges
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # BGR color for OpenCV
    color_bgr = (color[2], color[1], color[0])
    
    for i, j in edges:
        pt1 = vertices_2d[i]
        pt2 = vertices_2d[j]
        
        # Skip if either point is invalid
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue
        
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int))
        
        # Check bounds
        h, w = img.shape[:2]
        if not (-1000 < pt1[0] < w + 1000 and -1000 < pt1[1] < h + 1000):
            continue
        if not (-1000 < pt2[0] < w + 1000 and -1000 < pt2[1] < h + 1000):
            continue
        
        cv2.line(img, pt1, pt2, color_bgr, thickness)
    
    # Fill bottom face for visibility
    bottom_pts = vertices_2d[:4]
    if not np.isnan(bottom_pts).any():
        pts = bottom_pts.astype(np.int32).reshape((-1, 1, 2))
        # Semi-transparent fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)


def render_frame_with_boxes(
    background: np.ndarray,
    tracks: List[Track],
    frame_idx: int,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Render a frame with 3D boxes for dynamic objects."""
    
    img = background.copy()
    h, w = img.shape[:2]
    
    boxes_drawn = 0
    
    for track in tracks:
        if frame_idx not in track.frames:
            continue
        
        pos = track.frames[frame_idx]
        
        # Get box parameters
        size = get_box_size(track.class_name, track.category)
        color = get_box_color(track.category, track.is_stationary)
        
        # Estimate yaw from trajectory (if we have multiple frames)
        yaw = 0.0
        frame_indices = sorted(track.frames.keys())
        idx = frame_indices.index(frame_idx)
        if idx > 0:
            prev_pos = track.frames[frame_indices[idx - 1]]
            direction = pos[:2] - prev_pos[:2]
            if np.linalg.norm(direction) > 0.1:
                yaw = np.arctan2(direction[1], direction[0])
        
        # Create box vertices
        vertices = create_box_vertices(pos, size, yaw)
        
        # Project to image
        vertices_2d = project_box_to_image(vertices, K, R, t, w, h)
        
        if vertices_2d is None:
            continue
        
        # Draw box
        draw_box_3d(img, vertices_2d, color, thickness=2)
        boxes_drawn += 1
    
    return img, boxes_drawn


def render_static_background(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    point_size: int = 3
) -> np.ndarray:
    """Render static point cloud as background."""
    
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    
    # Transform to camera coordinates
    pts_cam = (R @ points.T).T + t
    
    # Filter points in front of camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    colors_valid = colors[valid]
    
    if len(pts_cam) == 0:
        return img
    
    # Project to image
    pts_proj = (K @ pts_cam.T).T
    pts_2d = pts_proj[:, :2] / pts_proj[:, 2:3]
    depths = pts_cam[:, 2]
    
    # Filter points in image bounds
    in_bounds = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < width) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < height)
    )
    pts_2d = pts_2d[in_bounds]
    colors_valid = colors_valid[in_bounds]
    depths = depths[in_bounds]
    
    # Sort by depth (far to near)
    sort_idx = np.argsort(-depths)
    pts_2d = pts_2d[sort_idx]
    colors_sorted = colors_valid[sort_idx]
    
    # Draw points
    for pt, col in zip(pts_2d, colors_sorted):
        x, y = int(pt[0]), int(pt[1])
        color_bgr = (int(col[2]), int(col[1]), int(col[0]))
        cv2.circle(img, (x, y), point_size, color_bgr, -1)
    
    return img


def render_4d_video(config, logger, camera_name: str = "s1-right"):
    """Main rendering function."""
    
    base_dir = Path(config['data']['output_dir'])
    pass1_dir = base_dir / "pass1_static"
    pass2_dir = base_dir / "pass2_dynamic"
    
    # Load Pi3 data
    logger.info("Loading Pi3 static scene...")
    points, colors, cameras, transform = load_pi3_data(pass1_dir)
    logger.info(f"  Loaded {len(points)} points")
    
    # Subsample for faster rendering
    max_points = 300000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        logger.info(f"  Subsampled to {len(points)} points")
    
    # Load trajectories
    logger.info("Loading trajectories...")
    tracks = load_trajectories(pass2_dir, transform)
    logger.info(f"  Loaded {len(tracks)} tracks")
    
    # Get frame range
    all_frames = set()
    for track in tracks:
        all_frames.update(track.frames.keys())
    
    if not all_frames:
        logger.error("No trajectory frames found!")
        return
    
    min_frame = min(all_frames)
    max_frame = max(all_frames)
    total_frames = max_frame - min_frame + 1
    logger.info(f"  Frame range: {min_frame} to {max_frame}")
    
    # Choose a camera for rendering
    cam_name = camera_name
    if cam_name not in cameras:
        cam_name = list(cameras.keys())[0]
        logger.warning(f"Camera {camera_name} not found, using {cam_name}")
    
    cam_data = cameras[cam_name]
    K = np.array(cam_data['K'], dtype=np.float32)
    
    # Get world-to-camera transform
    pose_c2w = np.array(cam_data['pose_c2w'], dtype=np.float32)
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    
    width = cam_data['width']
    height = cam_data['height']
    
    logger.info(f"Rendering from camera: {cam_name} ({width}x{height})")
    
    # Pre-render static background
    logger.info("Rendering static background...")
    static_bg = render_static_background(
        points, colors, K, R_w2c, t_w2c, width, height, point_size=3
    )
    
    # Setup video writer
    output_path = pass2_dir / f"4d_boxes_{cam_name}.mp4"
    fps = 15.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Render frames
    logger.info(f"Rendering {total_frames} frames...")
    
    for frame_idx in tqdm(range(min_frame, max_frame + 1), desc="Rendering"):
        # Render boxes on static background
        frame, n_boxes = render_frame_with_boxes(
            static_bg, tracks, frame_idx, K, R_w2c, t_w2c
        )
        
        # Add overlay
        time_sec = frame_idx / fps
        cv2.putText(frame, f"Frame: {frame_idx} | Time: {time_sec:.2f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Dynamic Objects: {n_boxes}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        legend_y = height - 100
        cv2.putText(frame, "Legend:", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (10, legend_y + 10), (30, legend_y + 25), (0, 0, 255), -1)
        cv2.putText(frame, "Moving Vehicle", (35, legend_y + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (10, legend_y + 30), (30, legend_y + 45), (0, 165, 255), -1)
        cv2.putText(frame, "Stationary Vehicle", (35, legend_y + 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (10, legend_y + 50), (30, legend_y + 65), (0, 255, 0), -1)
        cv2.putText(frame, "Moving Person", (35, legend_y + 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (10, legend_y + 70), (30, legend_y + 85), (255, 255, 0), -1)
        cv2.putText(frame, "Stationary Person", (35, legend_y + 82), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        writer.write(frame)
    
    writer.release()
    logger.info(f"✓ Saved video to {output_path}")
    
    return str(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='s1-right', 
                        help='Camera to render from')
    parser.add_argument('--all', action='store_true',
                        help='Render from all cameras')
    args = parser.parse_args()
    
    config = load_config()
    logger = setup_logger(
        name="Render4DBoxes",
        log_dir=config['data']['log_dir'],
        level="INFO",
        save_to_file=True
    )
    
    logger.info("=== 4D Scene Renderer with 3D Boxes ===")
    
    if args.all:
        # Render from all cameras
        base_dir = Path(config['data']['output_dir'])
        pass1_dir = base_dir / "pass1_static"
        cameras_path = pass1_dir / "pi3_cameras_corrected.json"
        with open(cameras_path) as f:
            cameras = json.load(f)
        
        for cam_name in cameras.keys():
            logger.info(f"\n--- Rendering from {cam_name} ---")
            render_4d_video(config, logger, camera_name=cam_name)
    else:
        render_4d_video(config, logger, camera_name=args.camera)


if __name__ == "__main__":
    main()
