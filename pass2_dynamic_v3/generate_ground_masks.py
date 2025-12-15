#!/usr/bin/env python3
"""
Generate Ground Masks using Cityscapes Segmentation
=====================================================
Creates numpy grid masks for road/sidewalk that can be used by the tracker.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

ROAD_CLASS = 0
SIDEWALK_CLASS = 1

class CityscapesSegmenter:
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading SegFormer (Cityscapes)...")
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    
    def segment(self, img_rgb: np.ndarray) -> dict:
        h, w = img_rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()
        return {
            'road_mask': (seg == ROAD_CLASS),
            'sidewalk_mask': (seg == SIDEWALK_CLASS)
        }

class CameraProjector:
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
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    grid_dim = int(2 * grid_size / grid_resolution)
    road_votes = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    sidewalk_votes = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    total_votes = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    
    print("\nCreating ground masks with Cityscapes...")
    
    for cam_id in cam_ids:
        if cam_id not in cameras:
            continue
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists():
            continue
        
        print(f"  {cam_id}...")
        bg_img = cv2.imread(str(bg_path))
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        h, w = bg_rgb.shape[:2]
        
        masks = segmenter.segment(bg_rgb)
        road_mask = masks['road_mask']
        sidewalk_mask = masks['sidewalk_mask']
        
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
                    total_votes[gy, gx] += 1
                    if road_mask[y, x]:
                        road_votes[gy, gx] += 1
                    if sidewalk_mask[y, x]:
                        sidewalk_votes[gy, gx] += 1
    
    # Threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        road_ratio = road_votes / (total_votes + 1e-6)
        sidewalk_ratio = sidewalk_votes / (total_votes + 1e-6)
    
    road_grid = (road_ratio > 0.3) & (total_votes > 0)
    # Curb = sidewalk OR (not road and has votes)
    curb_grid = ((sidewalk_ratio > 0.1) | ((road_ratio < 0.3) & (total_votes > 0))) & (total_votes > 0)
    
    # Dilate
    kernel = np.ones((3, 3), np.uint8)
    road_grid = cv2.dilate(road_grid.astype(np.uint8), kernel, iterations=2).astype(bool)
    curb_grid = cv2.dilate(curb_grid.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # Remove curb where road is
    curb_grid = curb_grid & ~road_grid
    
    grid_info = {
        'resolution': grid_resolution,
        'size': grid_size,
        'dim': grid_dim,
        'origin': [-grid_size, -grid_size]
    }
    
    print(f"\n  Road: {road_grid.sum() / road_grid.size * 100:.1f}%")
    print(f"  Curb: {curb_grid.sum() / curb_grid.size * 100:.1f}%")
    
    return road_grid, curb_grid, grid_info

def main():
    base_dir = Path(__file__).parent.parent
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    output_dir = base_dir / "outputs" / "pass1_static" / "ground_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    segmenter = CityscapesSegmenter()
    road_grid, curb_grid, grid_info = create_ground_masks(bg_dir, cameras, segmenter)
    
    np.save(output_dir / "road_grid.npy", road_grid)
    np.save(output_dir / "curb_grid.npy", curb_grid)
    
    with open(output_dir / "grid_info.json", 'w') as f:
        json.dump(grid_info, f, indent=2)
    
    # Visualization
    dim = grid_info['dim']
    vis = np.zeros((dim, dim, 3), dtype=np.uint8)
    vis[road_grid] = [100, 100, 100]
    vis[curb_grid] = [34, 139, 34]
    vis = np.flipud(vis)
    cv2.imwrite(str(output_dir / "ground_mask.png"), vis)
    
    print(f"\nSaved to {output_dir}")

if __name__ == "__main__":
    main()
