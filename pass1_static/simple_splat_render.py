#!/usr/bin/env python3
"""
Simple Gaussian Splatting Renderer using pure PyTorch.

This is a simplified renderer that doesn't require CUDA compilation.
It renders the point cloud with Gaussian-like appearance using alpha blending.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
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


def render_gaussians_pytorch(
    points: torch.Tensor,  # [N, 3]
    colors: torch.Tensor,  # [N, 3]
    K: torch.Tensor,       # [3, 3]
    R: torch.Tensor,       # [3, 3] world-to-camera rotation
    t: torch.Tensor,       # [3] world-to-camera translation
    width: int,
    height: int,
    point_radius: float = 0.1,  # meters
    device: str = "cuda"
) -> np.ndarray:
    """Render points as soft Gaussians using PyTorch."""
    
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
    
    # Compute projected radius (approximate)
    fx = K[0, 0]
    radii_px = (point_radius * fx / depths).clamp(min=1, max=50)  # [N]
    
    # Sort by depth (far to near for proper blending)
    sort_idx = torch.argsort(depths, descending=True)
    pts_2d = pts_2d[sort_idx]
    colors_valid = colors_valid[sort_idx]
    radii_px = radii_px[sort_idx]
    
    # Create output image
    img = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
    alpha = torch.zeros((height, width), device=device, dtype=torch.float32)
    
    # Create coordinate grids
    y_coords = torch.arange(height, device=device).float()
    x_coords = torch.arange(width, device=device).float()
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Render each point as a Gaussian splat
    # Process in batches for memory efficiency
    batch_size = 1000
    n_points = len(pts_2d)
    
    for i in range(0, n_points, batch_size):
        end_i = min(i + batch_size, n_points)
        
        for j in range(i, end_i):
            cx, cy = pts_2d[j]
            r = radii_px[j]
            color = colors_valid[j]
            
            # Skip if center is way outside image
            if cx < -r*2 or cx > width + r*2 or cy < -r*2 or cy > height + r*2:
                continue
            
            # Compute Gaussian weights for nearby pixels
            # Use a bounding box for efficiency
            x_min = max(0, int(cx - r * 3))
            x_max = min(width, int(cx + r * 3) + 1)
            y_min = max(0, int(cy - r * 3))
            y_max = min(height, int(cy + r * 3) + 1)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Local coordinates
            local_xx = xx[y_min:y_max, x_min:x_max]
            local_yy = yy[y_min:y_max, x_min:x_max]
            
            # Gaussian weight
            dist_sq = (local_xx - cx)**2 + (local_yy - cy)**2
            sigma = r / 2
            weight = torch.exp(-dist_sq / (2 * sigma**2))
            
            # Alpha blending
            local_alpha = alpha[y_min:y_max, x_min:x_max]
            blend = weight * (1 - local_alpha)
            
            img[y_min:y_max, x_min:x_max] += blend.unsqueeze(-1) * color
            alpha[y_min:y_max, x_min:x_max] += blend
    
    # Normalize by alpha
    mask = alpha > 0.001
    img[mask] = img[mask] / alpha[mask].unsqueeze(-1)
    
    # Convert to uint8
    img = (img.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    
    return img


def render_all_views(points: np.ndarray, colors: np.ndarray, 
                     cameras: dict, output_dir: Path, device: str = "cuda"):
    """Render from all camera viewpoints."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to tensors
    points_t = torch.tensor(points, dtype=torch.float32, device=device)
    colors_t = torch.tensor(colors / 255.0, dtype=torch.float32, device=device)
    
    for cam_name, cam_info in tqdm(cameras.items(), desc="Rendering views"):
        K = torch.tensor(cam_info['K'], dtype=torch.float32, device=device)
        
        # Get world-to-camera transform
        pose_c2w = np.array(cam_info['pose_c2w'], dtype=np.float32)
        R_c2w = pose_c2w[:3, :3]
        t_c2w = pose_c2w[:3, 3]
        
        R_w2c = torch.tensor(R_c2w.T, dtype=torch.float32, device=device)
        t_w2c = torch.tensor(-R_c2w.T @ t_c2w, dtype=torch.float32, device=device)
        
        # Scale down for faster rendering
        scale = 0.25
        width = int(cam_info['width'] * scale)
        height = int(cam_info['height'] * scale)
        K_scaled = K.clone()
        K_scaled[0, :] *= scale
        K_scaled[1, :] *= scale
        
        # Render
        img = render_gaussians_pytorch(
            points_t, colors_t, K_scaled, R_w2c, t_w2c,
            width, height, point_radius=0.15, device=device
        )
        
        # Save
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{cam_name}_splat.png"), img_bgr)
    
    print(f"Saved renders to {output_dir}")


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
    max_points = 200000
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    render_dir = output_dir / "gaussian_splatting" / "renders"
    render_all_views(points, colors, cameras, render_dir, device)
    
    print("\n=== Done! ===")
    print(f"Renders saved to {render_dir}")


if __name__ == "__main__":
    main()
