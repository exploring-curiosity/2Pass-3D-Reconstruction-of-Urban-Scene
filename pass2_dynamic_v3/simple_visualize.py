#!/usr/bin/env python3
"""
Simple 2D BBox + Direction Visualization
=========================================
Bird's eye view with:
- 2D rectangles (not 3D cubes)
- Direction arrows for dynamic objects
- Grid overlay
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm

def main():
    base = Path(__file__).parent.parent
    out = base / "outputs" / "pass2_dynamic_v3"
    
    with open(out / "scene_4d.json") as f:
        scene = json.load(f)
    
    total_frames = scene['total_frames']
    fps = scene['fps']
    objects = scene['objects']
    frames_data = scene['frames']
    
    print(f"Scene: {len(objects)} objects, {total_frames} frames")
    
    # Video setup
    size = 800
    scale = 15  # pixels per meter
    center = size // 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out / "reconstruction_4d.mp4"), fourcc, fps, (size, size))
    
    # Object dimensions (length, width in meters)
    DIMS = {
        'car': (4.5, 1.8), 'truck': (7.0, 2.4), 'bus': (10.0, 2.5),
        'motorcycle': (2.0, 0.8), 'bicycle': (1.8, 0.5), 'person': (0.5, 0.5)
    }
    
    print(f"Rendering {total_frames} frames...")
    
    for fi in tqdm(range(total_frames)):
        # Background
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        
        # Grid
        for i in range(-20, 21, 5):
            x = center + i * scale
            cv2.line(img, (x, 0), (x, size), (60, 60, 60), 1)
            cv2.line(img, (0, x), (size, x), (60, 60, 60), 1)
        
        # Center marker
        cv2.circle(img, (center, center), 3, (100, 100, 100), -1)
        
        # Draw objects
        fk = str(fi)
        if fk in frames_data:
            for obj in frames_data[fk]:
                oid = obj['id']
                pos = obj['pos']
                rot = obj['rot']
                
                if oid not in objects:
                    continue
                
                obj_info = objects[oid]
                cls = obj_info['class']
                color = tuple(obj_info['color'][::-1])  # BGR
                is_static = obj_info['is_stationary']
                
                # Convert position to pixels
                px = int(center + pos[0] * scale)
                py = int(center - pos[1] * scale)  # Flip Y
                
                # Get yaw from quaternion
                quat = np.array(rot)
                # Simple extraction: yaw ≈ 2 * atan2(qz, qw)
                yaw = 2 * np.arctan2(quat[2], quat[3])
                
                # Get dimensions
                length, width = DIMS.get(cls, (4.0, 1.8))
                half_l = length * scale / 2
                half_w = width * scale / 2
                
                # Rectangle corners
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                corners = [
                    (-half_l, -half_w), (half_l, -half_w),
                    (half_l, half_w), (-half_l, half_w)
                ]
                
                pts = []
                for cx, cy in corners:
                    rx = cx * cos_y - cy * sin_y
                    ry = cx * sin_y + cy * cos_y
                    pts.append([int(px + rx), int(py - ry)])
                
                pts = np.array(pts, dtype=np.int32)
                
                if is_static:
                    cv2.polylines(img, [pts], True, color, 2)
                else:
                    cv2.fillPoly(img, [pts], color)
                    
                    # Direction arrow
                    arrow_len = half_l * 1.5
                    ax = int(px + arrow_len * cos_y)
                    ay = int(py - arrow_len * sin_y)
                    cv2.arrowedLine(img, (px, py), (ax, ay), (255, 255, 255), 2, tipLength=0.3)
                
                # Label
                label = oid
                cv2.putText(img, label, (px - 15, py - int(half_w) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Frame counter
        cv2.putText(img, f"Frame: {fi}/{total_frames}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Object count
        static_count = sum(1 for o in objects.values() if o['is_stationary'])
        dynamic_count = len(objects) - static_count
        cv2.putText(img, f"Static: {static_count}, Dynamic: {dynamic_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        writer.write(img)
    
    writer.release()
    print(f"\nSaved: {out / 'reconstruction_4d.mp4'}")

if __name__ == "__main__":
    main()
