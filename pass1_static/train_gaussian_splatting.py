#!/usr/bin/env python3
"""
Train 3D Gaussian Splatting on the Pi3 static scene reconstruction.

Uses gsplat library for efficient Gaussian splatting.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional
import math

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_ply

# gsplat imports
from gsplat import rasterization


@dataclass
class GaussianParams:
    """Parameters for 3D Gaussians."""
    means: torch.Tensor      # [N, 3] positions
    scales: torch.Tensor     # [N, 3] scales (log space)
    quats: torch.Tensor      # [N, 4] quaternions
    opacities: torch.Tensor  # [N] opacities (logit space)
    colors: torch.Tensor     # [N, 3] RGB colors


class GaussianSplatModel(nn.Module):
    """3D Gaussian Splatting model."""
    
    def __init__(self, points: np.ndarray, colors: np.ndarray, device: str = "cuda"):
        super().__init__()
        
        n_points = len(points)
        print(f"Initializing {n_points} Gaussians...")
        
        # Initialize means from point cloud
        self.means = nn.Parameter(torch.tensor(points, dtype=torch.float32, device=device))
        
        # Initialize scales (small spheres)
        initial_scale = 0.1  # meters
        self.scales = nn.Parameter(
            torch.full((n_points, 3), math.log(initial_scale), dtype=torch.float32, device=device)
        )
        
        # Initialize quaternions (identity rotation)
        quats = torch.zeros((n_points, 4), dtype=torch.float32, device=device)
        quats[:, 0] = 1.0  # w=1, x=y=z=0
        self.quats = nn.Parameter(quats)
        
        # Initialize opacities (sigmoid^-1(0.5) = 0)
        self.opacities = nn.Parameter(
            torch.zeros(n_points, dtype=torch.float32, device=device)
        )
        
        # Initialize colors from point cloud
        colors_normalized = colors.astype(np.float32) / 255.0
        self.colors = nn.Parameter(
            torch.tensor(colors_normalized, dtype=torch.float32, device=device)
        )
        
        self.device = device
    
    def get_params(self) -> GaussianParams:
        """Get current Gaussian parameters."""
        return GaussianParams(
            means=self.means,
            scales=torch.exp(self.scales),
            quats=torch.nn.functional.normalize(self.quats, dim=-1),
            opacities=torch.sigmoid(self.opacities),
            colors=torch.sigmoid(self.colors)
        )
    
    def render(self, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
               width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render from a given camera viewpoint.
        
        Args:
            K: [3, 3] intrinsic matrix
            R: [3, 3] rotation matrix (world-to-camera)
            t: [3] translation vector
            width: image width
            height: image height
            
        Returns:
            rgb: [H, W, 3] rendered image
            alpha: [H, W] alpha channel
        """
        params = self.get_params()
        
        # Build camera-to-world matrix (gsplat convention)
        # gsplat expects viewmat as world-to-camera [4, 4]
        viewmat = torch.eye(4, device=self.device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        
        # gsplat rasterization
        renders, alphas, meta = rasterization(
            means=params.means,
            quats=params.quats,
            scales=params.scales,
            opacities=params.opacities,
            colors=params.colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            near_plane=0.1,
            far_plane=100.0,
            render_mode="RGB",
        )
        
        return renders[0], alphas[0]


def load_training_data(config_path: Path, images_dir: Path) -> list:
    """Load camera parameters and images for training."""
    
    with open(config_path) as f:
        cameras = json.load(f)
    
    training_data = []
    for cam_name, cam_data in cameras.items():
        img_path = images_dir / f"{cam_name}_bg.png"
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        K = np.array(cam_data['K'], dtype=np.float32)
        
        # Get world-to-camera transform
        pose_c2w = np.array(cam_data['pose_c2w'], dtype=np.float32)
        R_c2w = pose_c2w[:3, :3]
        t_c2w = pose_c2w[:3, 3]
        
        # Invert to get world-to-camera
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        
        training_data.append({
            'name': cam_name,
            'image': img,
            'K': K,
            'R': R_w2c,
            't': t_w2c,
            'width': cam_data['width'],
            'height': cam_data['height']
        })
    
    return training_data


def train_gaussian_splatting(
    points: np.ndarray,
    colors: np.ndarray,
    training_data: list,
    output_dir: Path,
    num_iterations: int = 3000,
    lr: float = 0.01,
    device: str = "cuda"
):
    """Train Gaussian Splatting model."""
    
    print(f"\n=== Training Gaussian Splatting ===")
    print(f"  Points: {len(points)}")
    print(f"  Training views: {len(training_data)}")
    print(f"  Iterations: {num_iterations}")
    
    # Subsample points for faster training
    max_points = 100000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"  Subsampled to {len(points)} points")
    
    # Initialize model
    model = GaussianSplatModel(points, colors, device)
    
    # Optimizer
    optimizer = Adam([
        {'params': [model.means], 'lr': lr * 0.1},
        {'params': [model.scales], 'lr': lr * 0.5},
        {'params': [model.quats], 'lr': lr * 0.1},
        {'params': [model.opacities], 'lr': lr},
        {'params': [model.colors], 'lr': lr},
    ])
    
    # Prepare training images
    train_images = []
    train_cameras = []
    for data in training_data:
        # Resize images for faster training
        scale = 0.25
        h, w = data['image'].shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(data['image'], (new_w, new_h))
        img_tensor = torch.tensor(img_resized, dtype=torch.float32, device=device) / 255.0
        
        # Scale intrinsics
        K = data['K'].copy()
        K[0, :] *= scale
        K[1, :] *= scale
        
        train_images.append(img_tensor)
        train_cameras.append({
            'K': torch.tensor(K, device=device),
            'R': torch.tensor(data['R'], device=device),
            't': torch.tensor(data['t'], device=device),
            'width': new_w,
            'height': new_h,
            'name': data['name']
        })
    
    # Training loop
    losses = []
    pbar = tqdm(range(num_iterations), desc="Training")
    
    for iteration in pbar:
        # Random view
        idx = np.random.randint(len(train_images))
        gt_image = train_images[idx]
        cam = train_cameras[idx]
        
        # Render
        optimizer.zero_grad()
        
        try:
            rendered, alpha = model.render(
                cam['K'], cam['R'], cam['t'],
                cam['width'], cam['height']
            )
            
            # L1 loss
            loss = torch.abs(rendered - gt_image).mean()
            
            # Add opacity regularization
            params = model.get_params()
            loss += 0.01 * params.opacities.mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            continue
        
        # Save checkpoint
        if (iteration + 1) % 1000 == 0:
            save_checkpoint(model, output_dir / f"checkpoint_{iteration+1}.pt")
    
    # Save final model
    save_checkpoint(model, output_dir / "gaussian_model.pt")
    
    # Save rendered views
    render_all_views(model, train_cameras, output_dir / "renders")
    
    return model, losses


def save_checkpoint(model: GaussianSplatModel, path: Path):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'means': model.means.data,
        'scales': model.scales.data,
        'quats': model.quats.data,
        'opacities': model.opacities.data,
        'colors': model.colors.data,
    }, path)
    print(f"  Saved checkpoint to {path}")


def render_all_views(model: GaussianSplatModel, cameras: list, output_dir: Path):
    """Render all training views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for cam in cameras:
            rendered, alpha = model.render(
                cam['K'], cam['R'], cam['t'],
                cam['width'], cam['height']
            )
            
            # Convert to image
            img = (rendered.cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(output_dir / f"{cam['name']}_render.png"), img)
    
    print(f"  Saved renders to {output_dir}")


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs" / "pass1_static"
    
    # Load corrected point cloud
    ply_path = output_dir / "pi3_pointcloud_corrected.ply"
    cameras_path = output_dir / "pi3_cameras_corrected.json"
    images_dir = base_dir / "data" / "processed" / "static_backgrounds"
    
    if not ply_path.exists():
        print(f"Error: {ply_path} not found. Run fix_pi3_orientation.py first.")
        return
    
    print("Loading point cloud...")
    points, colors, _ = load_ply(str(ply_path))
    print(f"  Loaded {len(points)} points")
    
    print("Loading training data...")
    training_data = load_training_data(cameras_path, images_dir)
    print(f"  Loaded {len(training_data)} views")
    
    if len(training_data) == 0:
        print("Error: No training images found!")
        return
    
    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, losses = train_gaussian_splatting(
        points, colors, training_data,
        output_dir / "gaussian_splatting",
        num_iterations=2000,
        device=device
    )
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
