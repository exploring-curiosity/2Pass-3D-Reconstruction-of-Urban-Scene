#!/usr/bin/env python3
"""
Road Segmentation using DeepLabV3
==================================
Uses torchvision's DeepLabV3-ResNet50 pretrained on COCO for road detection.
This avoids the HuggingFace model loading issues.

COCO classes that indicate road area:
- 0: __background__
- Use area that is flat ground (no objects)
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

sys.path.insert(0, str(Path(__file__).parent.parent))

class DeepLabRoadSegmenter:
    """
    Road segmentation using DeepLabV3.
    Since COCO doesn't have explicit road class, we use:
    - Areas without objects on ground level = road
    - Areas with person, vehicle classes = dynamic
    """
    
    # COCO classes related to ground
    PERSON_CLASS = 15  # person
    VEHICLE_CLASSES = {2, 7, 14, 6}  # bicycle, car, motorbike, bus
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading DeepLabV3-ResNet50...")
        
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_resnet50(weights=weights).to(device).eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("  Loaded!")
    
    def segment(self, img_rgb: np.ndarray) -> dict:
        """
        Segment image.
        Returns masks for:
        - ground: likely road/ground areas
        - non_ground: areas with objects/buildings
        """
        h, w = img_rgb.shape[:2]
        
        # Transform
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        
        # Resize to original
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        seg = output.argmax(1)[0].cpu().numpy()
        
        # Background (class 0) is typically ground/road in outdoor scenes
        background_mask = (seg == 0)
        
        # Lower part of image is more likely to be road
        # Weight by vertical position
        y_weights = np.linspace(0.3, 1.0, h).reshape(-1, 1)
        weighted_bg = background_mask * y_weights
        
        road_mask = weighted_bg > 0.5
        
        # Non-road = any recognized class
        non_road_mask = (seg != 0)
        
        return {
            'road_mask': road_mask,
            'non_road_mask': non_road_mask,
            'full_seg': seg
        }

class CameraProjector:
    """Projects between 2D and 3D."""
    
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]
    
    def pixel_to_ground(self, u: float, v: float, z: float = 0.0):
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        ray_world = self.R_c2w @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        if abs(ray_world[2]) < 1e-6:
            return None
        t = (z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
        point = self.t_c2w + t * ray_world
        if np.linalg.norm(point[:2]) > 35:
            return None
        return point

def create_ground_masks(bg_dir: Path, cameras: dict, segmenter,
                        grid_resolution: float = 0.5, grid_size: float = 40.0):
    """
    Create 3D ground-plane masks.
    """
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    grid_dim = int(2 * grid_size / grid_resolution)
    road_votes = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    total_votes = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    
    print("\nCreating ground masks using DeepLabV3...")
    
    for cam_id in cam_ids:
        if cam_id not in cameras:
            continue
        
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists():
            print(f"  {cam_id}: No background, skipping")
            continue
        
        print(f"  {cam_id}...")
        
        bg_img = cv2.imread(str(bg_path))
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        h, w = bg_rgb.shape[:2]
        
        masks = segmenter.segment(bg_rgb)
        road_mask = masks['road_mask']
        
        print(f"    Road area: {road_mask.sum()/road_mask.size*100:.1f}%")
        
        proj = CameraProjector(cameras[cam_id])
        
        step = 8
        for y in range(0, h, step):
            for x in range(0, w, step):
                pt = proj.pixel_to_ground(x, y)
                if pt is None:
                    continue
                
                gx = int((pt[0] + grid_size) / grid_resolution)
                gy = int((pt[1] + grid_size) / grid_resolution)
                
                if 0 <= gx < grid_dim and 0 <= gy < grid_dim:
                    if road_mask[y, x]:
                        road_votes[gy, gx] += 1
                    total_votes[gy, gx] += 1
    
    # Threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        road_ratio = road_votes / (total_votes + 1e-6)
    
    road_grid = (road_ratio > 0.3) & (total_votes > 0)
    curb_grid = (road_ratio < 0.3) & (total_votes > 0)  # Not-road = curb/sidewalk
    
    # Dilate
    kernel = np.ones((3, 3), np.uint8)
    road_grid = cv2.dilate(road_grid.astype(np.uint8), kernel, iterations=2).astype(bool)
    curb_grid = cv2.dilate(curb_grid.astype(np.uint8), kernel, iterations=2).astype(bool)
    
    grid_info = {
        'resolution': grid_resolution,
        'size': grid_size,
        'dim': grid_dim,
        'origin': [-grid_size, -grid_size]
    }
    
    print(f"\n  Road: {road_grid.sum() / road_grid.size * 100:.1f}%")
    print(f"  Curb: {curb_grid.sum() / curb_grid.size * 100:.1f}%")
    
    return road_grid, curb_grid, grid_info

def save_masks(road_grid: np.ndarray, curb_grid: np.ndarray, 
               grid_info: dict, output_dir: Path):
    """Save masks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dim = grid_info['dim']
    vis = np.zeros((dim, dim, 3), dtype=np.uint8)
    vis[road_grid] = [100, 100, 100]
    vis[curb_grid] = [34, 139, 34]
    
    vis = np.flipud(vis)
    
    cv2.imwrite(str(output_dir / "ground_mask.png"), vis)
    np.save(output_dir / "road_grid.npy", road_grid)
    np.save(output_dir / "curb_grid.npy", curb_grid)
    
    with open(output_dir / "grid_info.json", 'w') as f:
        json.dump(grid_info, f, indent=2)
    
    print(f"\nSaved to {output_dir}")

def main():
    base_dir = Path(__file__).parent.parent
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    output_dir = base_dir / "outputs" / "pass1_static" / "ground_masks"
    
    segmenter = DeepLabRoadSegmenter()
    road_grid, curb_grid, grid_info = create_ground_masks(bg_dir, cameras, segmenter)
    save_masks(road_grid, curb_grid, grid_info, output_dir)

if __name__ == "__main__":
    main()
