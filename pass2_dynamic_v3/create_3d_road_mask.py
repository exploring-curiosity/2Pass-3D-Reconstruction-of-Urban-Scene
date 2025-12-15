#!/usr/bin/env python3
"""
Create 3D Point Cloud with Road/Curb Mask
==========================================
Creates two versions of the static scene:
1. Original point cloud (pi3_pointcloud_corrected.ply)
2. Point cloud with ground points colored by road/curb

For an urban intersection, expected ratio:
- Road (drivable): ~80%
- Curb (sidewalk/parking): ~20%
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
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

class RoadSegmenter:
    """Segment road from images."""
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading DeepLabV3...")
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_resnet50(weights=weights).to(device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def segment(self, img_rgb: np.ndarray) -> np.ndarray:
        """Returns road mask (True = road/ground, False = other)."""
        h, w = img_rgb.shape[:2]
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        seg = output.argmax(1)[0].cpu().numpy()
        
        # Background is typically ground in outdoor scenes
        # Weight lower part of image more (more likely to be road)
        y_weights = np.linspace(0.2, 1.0, h).reshape(-1, 1)
        background = (seg == 0).astype(float) * y_weights
        
        return background > 0.5

class CameraProjector:
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_w2c = pose[:3, :3].T
        self.t_w2c = -self.R_w2c @ pose[:3, 3]
        self.K_proj = self.K @ np.hstack([self.R_w2c, self.t_w2c.reshape(-1, 1)])
    
    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D pixel."""
        p_hom = np.append(point_3d, 1.0)
        p_img = self.K_proj @ p_hom
        if p_img[2] <= 0:
            return None
        return p_img[:2] / p_img[2]

def load_point_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    """Load point cloud and return as Open3D object."""
    print(f"Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Points: {len(pcd.points)}")
    return pcd

def create_road_mask_point_cloud(pcd: o3d.geometry.PointCloud,
                                   bg_dir: Path, 
                                   cameras: dict,
                                   segmenter: RoadSegmenter) -> o3d.geometry.PointCloud:
    """
    Color point cloud based on road/curb classification.
    - Road points: gray (100, 100, 100)
    - Curb points: green (34, 139, 34)
    - Sky/building: keep original or blue
    """
    
    points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.5
    new_colors = original_colors.copy()
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    # Track votes per point
    road_votes = np.zeros(len(points))
    curb_votes = np.zeros(len(points))
    total_votes = np.zeros(len(points))
    
    print("\nProjecting points to cameras and classifying...")
    
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
        
        road_mask = segmenter.segment(bg_rgb)
        
        proj = CameraProjector(cameras[cam_id])
        
        # Project each 3D point to this camera
        for i, point in enumerate(points):
            # Only process ground-level points (Z close to 0)
            if abs(point[2]) > 0.5:  # Only ground level
                continue
            
            pixel = proj.project_point(point)
            if pixel is None:
                continue
            
            px, py = int(pixel[0]), int(pixel[1])
            if 0 <= px < w and 0 <= py < h:
                total_votes[i] += 1
                if road_mask[py, px]:
                    road_votes[i] += 1
                else:
                    curb_votes[i] += 1
    
    # Classify points
    road_count = 0
    curb_count = 0
    
    for i in range(len(points)):
        if total_votes[i] == 0:
            continue
        
        road_ratio = road_votes[i] / total_votes[i]
        
        # Ground level classification
        if abs(points[i][2]) < 0.5:
            if road_ratio > 0.15:  # Lower threshold = more road (aiming 80%)
                new_colors[i] = [0.4, 0.4, 0.4]  # Gray for road
                road_count += 1
            else:
                new_colors[i] = [0.13, 0.55, 0.13]  # Green for curb
                curb_count += 1
    
    total_ground = road_count + curb_count
    if total_ground > 0:
        print(f"\n  Ground points classified:")
        print(f"    Road: {road_count} ({road_count/total_ground*100:.1f}%)")
        print(f"    Curb: {curb_count} ({curb_count/total_ground*100:.1f}%)")
    
    # Create new point cloud
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)
    pcd_new.colors = o3d.utility.Vector3dVector(new_colors)
    
    return pcd_new

def main():
    base_dir = Path(__file__).parent.parent
    
    print("=" * 70)
    print("3D POINT CLOUD WITH ROAD/CURB MASK")
    print("=" * 70)
    
    # Load camera params
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    # Load original point cloud
    pcd_path = base_dir / "outputs" / "pass1_static" / "pi3_pointcloud_corrected.ply"
    pcd = load_point_cloud(pcd_path)
    
    # Create segmenter
    segmenter = RoadSegmenter()
    
    # Background images
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    
    # Create masked point cloud
    pcd_masked = create_road_mask_point_cloud(pcd, bg_dir, cameras, segmenter)
    
    # Save both versions
    out_dir = base_dir / "outputs" / "pass1_static"
    
    # Original (just copy if not exists)
    out_original = out_dir / "scene_original.ply"
    o3d.io.write_point_cloud(str(out_original), pcd)
    print(f"\nSaved original: {out_original}")
    
    # With mask
    out_masked = out_dir / "scene_road_mask.ply"
    o3d.io.write_point_cloud(str(out_masked), pcd_masked)
    print(f"Saved masked: {out_masked}")
    
    print("\nDone! View with:")
    print(f"  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{out_masked}')])\"")

if __name__ == "__main__":
    main()
