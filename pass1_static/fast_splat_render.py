#!/usr/bin/env python3
"""
Fast Gaussian-like Point Cloud Renderer.

Uses OpenCV for efficient rendering with variable-size circles.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_ply(filepath: str):
    """Load PLY file."""
    from plyfile import PlyData
    
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    colors = None
    if 'red' in vertex.data.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    
    return points, colors


def render_splat_view(
    points: np.ndarray,  # [N, 3]
    colors: np.ndarray,  # [N, 3] RGB 0-255
    K: np.ndarray,       # [3, 3]
    R: np.ndarray,       # [3, 3] world-to-camera rotation
    t: np.ndarray,       # [3] world-to-camera translation
    width: int,
    height: int,
    base_radius: float = 0.1,  # meters
) -> np.ndarray:
    """Render points with depth-based size for splat-like appearance."""
    
    # Transform to camera coordinates
    pts_cam = (R @ points.T).T + t  # [N, 3]
    
    # Filter points in front of camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    colors_valid = colors[valid]
    
    if len(pts_cam) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Project to image
    pts_proj = (K @ pts_cam.T).T  # [N, 3]
    pts_2d = pts_proj[:, :2] / pts_proj[:, 2:3]  # [N, 2]
    depths = pts_cam[:, 2]  # [N]
    
    # Filter points in image bounds (with margin)
    margin = 50
    in_bounds = (
        (pts_2d[:, 0] >= -margin) & (pts_2d[:, 0] < width + margin) &
        (pts_2d[:, 1] >= -margin) & (pts_2d[:, 1] < height + margin)
    )
    pts_2d = pts_2d[in_bounds]
    colors_valid = colors_valid[in_bounds]
    depths = depths[in_bounds]
    
    if len(pts_2d) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Compute projected radius
    fx = K[0, 0]
    radii_px = np.clip(base_radius * fx / depths, 1, 30).astype(np.int32)
    
    # Sort by depth (far to near)
    sort_idx = np.argsort(-depths)
    pts_2d = pts_2d[sort_idx]
    colors_valid = colors_valid[sort_idx]
    radii_px = radii_px[sort_idx]
    
    # Create output image with dark background
    img = np.full((height, width, 3), 20, dtype=np.uint8)
    
    # Draw points as filled circles with anti-aliasing
    for i in range(len(pts_2d)):
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        r = radii_px[i]
        color = (int(colors_valid[i, 2]), int(colors_valid[i, 1]), int(colors_valid[i, 0]))  # BGR
        
        # Draw filled circle
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
    
    return img


def render_all_views(points: np.ndarray, colors: np.ndarray, 
                     cameras: dict, output_dir: Path, scale: float = 0.5):
    """Render from all camera viewpoints."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cam_name, cam_info in tqdm(cameras.items(), desc="Rendering views"):
        K = np.array(cam_info['K'], dtype=np.float32)
        
        # Get world-to-camera transform
        pose_c2w = np.array(cam_info['pose_c2w'], dtype=np.float32)
        R_c2w = pose_c2w[:3, :3]
        t_c2w = pose_c2w[:3, 3]
        
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        
        # Scale for rendering
        width = int(cam_info['width'] * scale)
        height = int(cam_info['height'] * scale)
        K_scaled = K.copy()
        K_scaled[0, :] *= scale
        K_scaled[1, :] *= scale
        
        # Render
        img = render_splat_view(
            points, colors, K_scaled, R_w2c, t_w2c,
            width, height, base_radius=0.15
        )
        
        # Save
        cv2.imwrite(str(output_dir / f"{cam_name}_splat.png"), img)
    
    print(f"\nSaved renders to {output_dir}")


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs" / "pass1_static"
    
    # Load point cloud
    ply_path = output_dir / "pi3_pointcloud_corrected.ply"
    if not ply_path.exists():
        print(f"Error: {ply_path} not found")
        return
    
    print("Loading point cloud...")
    points, colors = load_ply(str(ply_path))
    print(f"  Loaded {len(points)} points")
    
    # Subsample for faster rendering
    max_points = 300000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"  Subsampled to {len(points)} points")
    
    # Load cameras
    cameras_path = output_dir / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    print(f"  Loaded {len(cameras)} cameras")
    
    # Render
    render_dir = output_dir / "gaussian_splatting" / "renders"
    render_all_views(points, colors, cameras, render_dir, scale=0.5)
    
    print("\n=== Done! ===")
    print(f"Renders saved to {render_dir}")


if __name__ == "__main__":
    main()
