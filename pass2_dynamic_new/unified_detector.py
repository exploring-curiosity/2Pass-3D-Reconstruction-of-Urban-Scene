#!/usr/bin/env python3
"""
Stage 1: Unified 8-Camera Detector
===================================
Process all 8 cameras simultaneously per frame:
1. Run YOLO detection on each camera.
2. Compute road/sidewalk validity mask from static backgrounds.
3. Project detections to 3D ground plane.
4. Filter invalid projections (off-road, out-of-bounds).
5. Output unified detection list with camera, frame, 3D position.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

# Valid YOLO classes
VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

class SegmentationModel:
    """SegFormer for road/sidewalk detection."""
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading SegFormer...")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device).eval()
        
    def get_road_mask(self, img_rgb):
        """Returns binary mask where True = valid ground (road/sidewalk)."""
        inputs = self.processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        h, w = img_rgb.shape[:2]
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()
        
        # ADE20K: 6=road, 11=sidewalk, 13=earth, 9=grass (allow some tolerance)
        mask = (seg == 6) | (seg == 11) | (seg == 13)
        return mask

class CameraProjector:
    """Projects 2D pixels to 3D ground plane."""
    
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]  # Camera center in world
        
    def to_ground(self, u, v, ground_z=0.0):
        """Project pixel (u, v) to ground plane Z=ground_z."""
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = self.R_c2w @ ray_cam
        
        if abs(ray_world[2]) < 1e-6:
            return None
        
        t = (ground_z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
            
        return self.t_c2w + t * ray_world
    
    def to_image(self, point_3d):
        """Project 3D world point to image pixel."""
        R_w2c = self.R_c2w.T
        t_w2c = -R_w2c @ self.t_c2w
        p_cam = R_w2c @ point_3d + t_w2c
        if p_cam[2] <= 0:
            return None
        p_img = self.K @ p_cam
        return p_img[:2] / p_cam[2]

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = base_dir / "StreetAware-sample"
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    
    if not cameras_path.exists():
        print("ERROR: Camera parameters not found. Run Pass 1 first.")
        sys.exit(1)
    
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    # Initialize models
    print("Loading YOLO...")
    yolo = YOLO('yolov8x.pt')
    seg_model = SegmentationModel()
    
    # Build road masks from static backgrounds
    print("\nComputing road masks from static backgrounds...")
    road_masks = {}
    projectors = {}
    
    for cam_id in cam_ids:
        if cam_id not in cameras:
            continue
        
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if bg_path.exists():
            bg_img = cv2.imread(str(bg_path))
            bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            road_masks[cam_id] = seg_model.get_road_mask(bg_rgb)
            print(f"  {cam_id}: {road_masks[cam_id].sum() / road_masks[cam_id].size * 100:.1f}% valid ground")
        else:
            print(f"  {cam_id}: No background, will use full frame")
            road_masks[cam_id] = None
            
        projectors[cam_id] = CameraProjector(cameras[cam_id])
    
    # Open all video captures
    video_caps = {}
    total_frames = 0
    fps = 30.0
    frame_size = (0, 0)
    
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            video_caps[cam_id] = cap
            total_frames = max(total_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    print(f"\nProcessing {len(video_caps)} cameras, {total_frames} frames @ {fps:.1f} FPS")
    print(f"Frame size: {frame_size}")
    
    # Process all frames
    all_detections = []  # List of detection dicts
    
    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        frame_dets = []
        
        for cam_id in cam_ids:
            if cam_id not in video_caps:
                continue
            
            cap = video_caps[cam_id]
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Run YOLO
            results = yolo.predict(frame, conf=0.25, iou=0.5, verbose=False, 
                                   classes=list(VALID_CLASSES.keys()))
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                # Bottom center of bounding box
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]
                
                # Check road mask validity
                mask = road_masks.get(cam_id)
                if mask is not None:
                    ix, iy = int(cx), int(cy)
                    h, w = mask.shape
                    ix = max(0, min(w - 1, ix))
                    iy = max(0, min(h - 1, iy))
                    if not mask[iy, ix]:
                        continue  # Off-road, skip
                
                # Project to 3D
                pos_3d = projectors[cam_id].to_ground(cx, cy)
                if pos_3d is None:
                    continue
                
                # Bounds check (reasonable intersection area: ±40m)
                if abs(pos_3d[0]) > 40 or abs(pos_3d[1]) > 40:
                    continue
                
                frame_dets.append({
                    'frame': frame_idx,
                    'camera': cam_id,
                    'class': cls_name,
                    'conf': conf,
                    'bbox': bbox.tolist(),
                    'pos_3d': pos_3d.tolist()
                })
        
        all_detections.extend(frame_dets)
    
    # Cleanup
    for cap in video_caps.values():
        cap.release()
    
    # Save
    out_path = work_dir / "detections_3d.json"
    print(f"\nSaving {len(all_detections)} detections to {out_path}")
    
    with open(out_path, 'w') as f:
        json.dump({
            'total_frames': total_frames,
            'fps': fps,
            'cameras': cam_ids,
            'detections': all_detections
        }, f)
    
    # Stats
    by_class = defaultdict(int)
    by_cam = defaultdict(int)
    for d in all_detections:
        by_class[d['class']] += 1
        by_cam[d['camera']] += 1
    
    print("\nDetections by class:")
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")
    
    print("\nDetections by camera:")
    for cam, cnt in sorted(by_cam.items()):
        print(f"  {cam}: {cnt}")

if __name__ == "__main__":
    main()
