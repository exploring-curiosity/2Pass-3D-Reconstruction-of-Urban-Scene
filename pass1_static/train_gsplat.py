#!/usr/bin/env python3
"""
Train 3D Gaussian Splatting on the Pi3 static scene.

This script uses the gsplat library with proper CUDA support.
Run in the 'gsplat' mamba environment:
    mamba activate gsplat
    python pass1_static/train_gsplat.py
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
import math

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available. Install with: pip install gsplat")


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


class GaussianModel(nn.Module):
    """3D Gaussian Splatting model."""
    
    def __init__(self, points: np.ndarray, colors: np.ndarray, device: str = "cuda"):
        super().__init__()
        
        n_points = len(points)
        print(f"Initializing {n_points} Gaussians...")
        
        # Positions
        self.means = nn.Parameter(
            torch.tensor(points, dtype=torch.float32, device=device)
        )
        
        # Scales (log space for stability)
        initial_scale = 0.05
        self.log_scales = nn.Parameter(
            torch.full((n_points, 3), math.log(initial_scale), 
                      dtype=torch.float32, device=device)
        )
        
        # Quaternions (identity rotation)
        quats = torch.zeros((n_points, 4), dtype=torch.float32, device=device)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)
        
        # Opacities (logit space)
        self.logit_opacities = nn.Parameter(
            torch.zeros(n_points, dtype=torch.float32, device=device)
        )
        
        # Colors (sigmoid space)
        colors_norm = colors.astype(np.float32) / 255.0
        # Convert to logit space
        colors_logit = np.clip(colors_norm, 0.01, 0.99)
        colors_logit = np.log(colors_logit / (1 - colors_logit))
        self.logit_colors = nn.Parameter(
            torch.tensor(colors_logit, dtype=torch.float32, device=device)
        )
        
        self.device = device
    
    @property
    def scales(self):
        return torch.exp(self.log_scales)
    
    @property
    def opacities(self):
        return torch.sigmoid(self.logit_opacities)
    
    @property
    def colors(self):
        return torch.sigmoid(self.logit_colors)
    
    @property
    def rotations(self):
        return torch.nn.functional.normalize(self.quats, dim=-1)
    
    def render(self, viewmat: torch.Tensor, K: torch.Tensor, 
               width: int, height: int) -> torch.Tensor:
        """Render from a viewpoint."""
        
        renders, alphas, meta = rasterization(
            means=self.means,
            quats=self.rotations,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            near_plane=0.1,
            far_plane=100.0,
            render_mode="RGB",
        )
        
        return renders[0], alphas[0]


def load_training_data(cameras_path: Path, images_dir: Path, scale: float = 0.25):
    """Load cameras and images for training."""
    
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    data = []
    for cam_name, cam_info in cameras.items():
        img_path = images_dir / f"{cam_name}_bg.png"
        if not img_path.exists():
            print(f"  Skipping {cam_name}: image not found")
            continue
        
        # Load and resize image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Scale intrinsics
        K = np.array(cam_info['K'], dtype=np.float32)
        K[0, :] *= scale
        K[1, :] *= scale
        
        # Get world-to-camera transform
        pose_c2w = np.array(cam_info['pose_c2w'], dtype=np.float32)
        R_c2w = pose_c2w[:3, :3]
        t_c2w = pose_c2w[:3, 3]
        
        # Invert to world-to-camera
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        
        # Build 4x4 viewmat
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R_w2c
        viewmat[:3, 3] = t_w2c
        
        data.append({
            'name': cam_name,
            'image': img,
            'K': K,
            'viewmat': viewmat,
            'width': new_w,
            'height': new_h
        })
    
    return data


def train(model: GaussianModel, train_data: list, 
          num_iterations: int = 3000, lr: float = 0.01):
    """Train the Gaussian model."""
    
    optimizer = Adam([
        {'params': [model.means], 'lr': lr * 0.1, 'name': 'means'},
        {'params': [model.log_scales], 'lr': lr * 0.5, 'name': 'scales'},
        {'params': [model.quats], 'lr': lr * 0.1, 'name': 'quats'},
        {'params': [model.logit_opacities], 'lr': lr, 'name': 'opacities'},
        {'params': [model.logit_colors], 'lr': lr, 'name': 'colors'},
    ])
    
    # Prepare tensors
    device = model.device
    images = [torch.tensor(d['image'], dtype=torch.float32, device=device) / 255.0 
              for d in train_data]
    viewmats = [torch.tensor(d['viewmat'], device=device) for d in train_data]
    Ks = [torch.tensor(d['K'], device=device) for d in train_data]
    
    losses = []
    pbar = tqdm(range(num_iterations), desc="Training")
    
    for iteration in pbar:
        # Random view
        idx = np.random.randint(len(train_data))
        gt_image = images[idx]
        viewmat = viewmats[idx]
        K = Ks[idx]
        width = train_data[idx]['width']
        height = train_data[idx]['height']
        
        optimizer.zero_grad()
        
        try:
            rendered, alpha = model.render(viewmat, K, width, height)
            
            # L1 loss
            loss = torch.abs(rendered - gt_image).mean()
            
            # Regularization
            loss += 0.01 * model.opacities.mean()  # Encourage sparsity
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            continue
    
    return losses


def save_renders(model: GaussianModel, train_data: list, output_dir: Path):
    """Save rendered images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = model.device
    model.eval()
    
    with torch.no_grad():
        for data in train_data:
            viewmat = torch.tensor(data['viewmat'], device=device)
            K = torch.tensor(data['K'], device=device)
            
            rendered, alpha = model.render(viewmat, K, data['width'], data['height'])
            
            img = (rendered.cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(output_dir / f"{data['name']}_gsplat.png"), img)
    
    print(f"Saved renders to {output_dir}")


def main():
    if not GSPLAT_AVAILABLE:
        print("Error: gsplat not available. Run in gsplat environment.")
        return
    
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
    
    # Subsample for training
    max_points = 50000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"  Subsampled to {len(points)} points")
    
    # Load training data
    cameras_path = output_dir / "pi3_cameras_corrected.json"
    images_dir = base_dir / "data" / "processed" / "static_backgrounds"
    
    print("Loading training data...")
    train_data = load_training_data(cameras_path, images_dir, scale=0.25)
    print(f"  Loaded {len(train_data)} views")
    
    if len(train_data) == 0:
        print("Error: No training images found")
        return
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = GaussianModel(points, colors, device)
    
    # Train
    print("\n=== Training Gaussian Splatting ===")
    losses = train(model, train_data, num_iterations=2000, lr=0.01)
    
    # Save model
    model_path = output_dir / "gaussian_splatting" / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'means': model.means.data,
        'log_scales': model.log_scales.data,
        'quats': model.quats.data,
        'logit_opacities': model.logit_opacities.data,
        'logit_colors': model.logit_colors.data,
    }, model_path)
    print(f"Saved model to {model_path}")
    
    # Save renders
    save_renders(model, train_data, output_dir / "gaussian_splatting" / "renders")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
