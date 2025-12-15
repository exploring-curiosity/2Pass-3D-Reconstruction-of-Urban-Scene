#!/usr/bin/env python3
"""
Diagnostic: Analyze 3D projection quality
==========================================
This script visualizes how well our 3D projections align with actual objects.
It will:
1. Take sample frames
2. Show projected 3D positions back onto the image
3. Verify if boxes align with objects
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

class CameraProjector:
    """Projects between 2D and 3D."""
    
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]  # Camera center in world
        
        # Also compute w2c for reprojection
        self.R_w2c = self.R_c2w.T
        self.t_w2c = -self.R_w2c @ self.t_c2w
        
        print(f"Camera center: {self.t_c2w}")
        print(f"Camera forward direction: {self.R_c2w[:, 2]}")
        
    def pixel_to_ground(self, u: float, v: float, z: float = 0.0) -> np.ndarray:
        """Project pixel to ground plane Z=z."""
        # Ray in camera space
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        # Ray in world space
        ray_world = self.R_c2w @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        # Intersect with Z=z plane
        if abs(ray_world[2]) < 1e-6:
            return None
        
        t = (z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
            
        return self.t_c2w + t * ray_world
    
    def world_to_pixel(self, point: np.ndarray) -> np.ndarray:
        """Project 3D world point to pixel."""
        p_cam = self.R_w2c @ point + self.t_w2c
        if p_cam[2] <= 0:
            return None
        p_img = self.K @ p_cam
        return (p_img[:2] / p_cam[2]).astype(int)

def draw_box_3d(img, projector, center_3d, dims, color=(0, 255, 0)):
    """Draw 3D bounding box on image."""
    L, W, H = dims
    
    # Box corners in local space (center at origin)
    corners = np.array([
        [-L/2, -W/2, 0],
        [L/2, -W/2, 0],
        [L/2, W/2, 0],
        [-L/2, W/2, 0],
        [-L/2, -W/2, H],
        [L/2, -W/2, H],
        [L/2, W/2, H],
        [-L/2, W/2, H]
    ])
    
    # Transform to world (just translate, no rotation for now)
    corners_world = corners + center_3d
    
    # Project to image
    corners_2d = []
    for c in corners_world:
        p = projector.world_to_pixel(c)
        if p is not None:
            corners_2d.append(p)
        else:
            corners_2d.append(None)
    
    # Draw edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Verticals
    ]
    
    for i, j in edges:
        if corners_2d[i] is not None and corners_2d[j] is not None:
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, 2)
    
    # Draw center marker
    center_2d = projector.world_to_pixel(center_3d)
    if center_2d is not None:
        cv2.circle(img, tuple(center_2d), 5, (0, 0, 255), -1)

def main():
    base_dir = Path(__file__).parent.parent
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    video_dir = base_dir / "StreetAware-sample"
    out_dir = base_dir / "outputs" / "pass2_dynamic_new" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    yolo = YOLO('yolov8x.pt')
    
    VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    DIMS = {
        'car': [4.5, 1.8, 1.5],
        'truck': [7.0, 2.4, 2.8],
        'person': [0.5, 0.5, 1.7]
    }
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right']
    
    # Test frame
    test_frame = 50
    
    print("\n=== PROJECTION DIAGNOSTIC ===\n")
    
    for cam_id in cam_ids:
        print(f"\n--- {cam_id} ---")
        cap = cv2.VideoCapture(str(video_dir / f"{cam_id}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        projector = CameraProjector(cameras[cam_id])
        
        # Detect objects
        results = yolo.predict(frame, conf=0.5, verbose=False, 
                              classes=list(VALID_CLASSES.keys()))
        
        vis_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                # Draw 2D bbox
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
                # Get bottom center
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]
                
                # Project to ground
                ground_pt = projector.pixel_to_ground(cx, cy, 0.0)
                
                if ground_pt is not None:
                    print(f"  {cls_name}: bbox bottom ({cx:.0f}, {cy:.0f}) -> 3D ({ground_pt[0]:.1f}, {ground_pt[1]:.1f}, {ground_pt[2]:.1f})")
                    
                    # Draw 3D box
                    dims = DIMS.get(cls_name, [2, 1, 1.5])
                    draw_box_3d(vis_frame, projector, ground_pt, dims, (0, 255, 0))
                    
                    # Reproject ground point to verify
                    reproj = projector.world_to_pixel(ground_pt)
                    if reproj is not None:
                        cv2.circle(vis_frame, tuple(reproj), 8, (0, 255, 255), 2)
                        # Draw line from original to reprojected
                        cv2.line(vis_frame, (int(cx), int(cy)), tuple(reproj), (255, 255, 0), 1)
                else:
                    print(f"  {cls_name}: Failed to project to ground")
        
        # Save
        out_path = out_dir / f"diag_{cam_id}_frame{test_frame}.png"
        cv2.imwrite(str(out_path), vis_frame)
        print(f"  Saved: {out_path}")
    
    print(f"\nDiagnostic images saved to {out_dir}")

if __name__ == "__main__":
    main()
