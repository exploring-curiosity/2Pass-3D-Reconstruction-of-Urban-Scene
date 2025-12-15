#!/usr/bin/env python3
"""
Cityscapes Road Segmentation (Fixed)
=====================================
Now using properly loaded SegFormer trained on Cityscapes.

Cityscapes classes:
0: road
1: sidewalk  
2: building
3: wall
4: fence
5: pole
6: traffic light
7: traffic sign
8: vegetation
9: terrain
10: sky
11: person
12: rider
13: car
14: truck
15: bus
16: train
17: motorcycle
18: bicycle
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Cityscapes classes
ROAD_CLASS = 0      # Drivable road
SIDEWALK_CLASS = 1  # Sidewalk/curb

class CityscapesSegmenter:
    """Uses SegFormer trained on Cityscapes for accurate road segmentation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading SegFormer (Cityscapes)...")
        
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
        
        print(f"  Loaded! Classes: {self.model.config.num_labels}")
    
    def segment(self, img_rgb: np.ndarray) -> dict:
        """Returns road and sidewalk masks."""
        h, w = img_rgb.shape[:2]
        
        inputs = self.processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()
        
        road_mask = (seg == ROAD_CLASS)
        sidewalk_mask = (seg == SIDEWALK_CLASS)
        
        return {
            'road_mask': road_mask,
            'sidewalk_mask': sidewalk_mask,
            'full_seg': seg
        }

def generate_2d_samples():
    """Generate 2D overlay samples for verification."""
    base_dir = Path(__file__).parent.parent
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    video_dir = base_dir / "StreetAware-sample"
    out_dir = base_dir / "outputs" / "pass2_dynamic_v3" / "cityscapes_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    segmenter = CityscapesSegmenter()
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right']
    
    print("\n=== GENERATING CITYSCAPES SEGMENTATION SAMPLES ===\n")
    
    for cam_id in cam_ids:
        print(f"{cam_id}:")
        
        # Static background
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if bg_path.exists():
            bg_img = cv2.imread(str(bg_path))
            bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            h, w = bg_rgb.shape[:2]
            
            masks = segmenter.segment(bg_rgb)
            road = masks['road_mask']
            sidewalk = masks['sidewalk_mask']
            
            print(f"  BG: Road={road.sum()/road.size*100:.1f}%, Sidewalk={sidewalk.sum()/sidewalk.size*100:.1f}%")
            
            # Create overlay
            overlay = bg_rgb.copy()
            overlay[road] = [100, 100, 100]  # Gray for road
            overlay[sidewalk] = [34, 139, 34]  # Green for sidewalk
            
            blended = cv2.addWeighted(bg_rgb, 0.5, overlay, 0.5, 0)
            cv2.imwrite(str(out_dir / f"{cam_id}_bg_cityscapes.png"), 
                       cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        # Video frame
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                masks = segmenter.segment(frame_rgb)
                road = masks['road_mask']
                sidewalk = masks['sidewalk_mask']
                
                print(f"  Frame100: Road={road.sum()/road.size*100:.1f}%, Sidewalk={sidewalk.sum()/sidewalk.size*100:.1f}%")
                
                overlay = frame_rgb.copy()
                overlay[road] = [100, 100, 100]
                overlay[sidewalk] = [34, 139, 34]
                
                blended = cv2.addWeighted(frame_rgb, 0.5, overlay, 0.5, 0)
                cv2.imwrite(str(out_dir / f"{cam_id}_frame100_cityscapes.png"),
                           cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    print(f"\nSamples saved to: {out_dir}")

class CameraProjector:
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_w2c = pose[:3, :3].T
        self.t_w2c = -self.R_w2c @ pose[:3, 3]
        self.K_proj = self.K @ np.hstack([self.R_w2c, self.t_w2c.reshape(-1, 1)])
    
    def project_point(self, point_3d: np.ndarray):
        p_hom = np.append(point_3d, 1.0)
        p_img = self.K_proj @ p_hom
        if p_img[2] <= 0:
            return None
        return p_img[:2] / p_img[2]

def create_3d_road_mask():
    """Create 3D point cloud with road/sidewalk coloring."""
    base_dir = Path(__file__).parent.parent
    
    print("\n=== CREATING 3D ROAD MASK ===\n")
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    pcd_path = base_dir / "outputs" / "pass1_static" / "pi3_pointcloud_corrected.ply"
    print(f"Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.5
    new_colors = original_colors.copy()
    
    print(f"  Points: {len(points)}")
    
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    segmenter = CityscapesSegmenter()
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    road_votes = np.zeros(len(points))
    sidewalk_votes = np.zeros(len(points))
    total_votes = np.zeros(len(points))
    
    print("\nProjecting points to cameras...")
    
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
        
        for i, point in enumerate(points):
            # Only ground-level points
            if abs(point[2]) > 0.3:
                continue
            
            pixel = proj.project_point(point)
            if pixel is None:
                continue
            
            px, py = int(pixel[0]), int(pixel[1])
            if 0 <= px < w and 0 <= py < h:
                total_votes[i] += 1
                if road_mask[py, px]:
                    road_votes[i] += 1
                elif sidewalk_mask[py, px]:
                    sidewalk_votes[i] += 1
    
    # Classify points
    road_count = 0
    sidewalk_count = 0
    
    for i in range(len(points)):
        if total_votes[i] == 0:
            continue
        
        road_ratio = road_votes[i] / total_votes[i]
        sidewalk_ratio = sidewalk_votes[i] / total_votes[i]
        
        if abs(points[i][2]) < 0.3:  # Ground level only
            if road_ratio > 0.3:
                new_colors[i] = [0.4, 0.4, 0.4]  # Gray for road
                road_count += 1
            elif sidewalk_ratio > 0.2:
                new_colors[i] = [0.13, 0.55, 0.13]  # Green for sidewalk
                sidewalk_count += 1
    
    total_ground = road_count + sidewalk_count
    if total_ground > 0:
        print(f"\n  Ground points classified:")
        print(f"    Road: {road_count} ({road_count/total_ground*100:.1f}%)")
        print(f"    Sidewalk: {sidewalk_count} ({sidewalk_count/total_ground*100:.1f}%)")
    
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)
    pcd_new.colors = o3d.utility.Vector3dVector(new_colors)
    
    out_path = base_dir / "outputs" / "pass1_static" / "scene_road_mask.ply"
    o3d.io.write_point_cloud(str(out_path), pcd_new)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--3d":
        create_3d_road_mask()
    else:
        generate_2d_samples()
        print("\nTo create 3D mask, run with --3d flag")
