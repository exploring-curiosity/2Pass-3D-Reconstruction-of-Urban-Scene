#!/usr/bin/env python3
"""
V5 Visualization
================
Visualizes scene_4d.json produced by V5 pipeline.
Handles list-based object format.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def main():
    base = Path(__file__).parent.parent
    out = base / "outputs" / "pass2_dynamic_v5"
    
    json_path = out / "scene_4d.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    with open(json_path) as f:
        objects = json.load(f)
    
    print(f"Loaded {len(objects)} objects")
    
    # Determine total frames
    max_frame = 0
    for obj in objects:
        if not obj.get('static', False):
            for kf in obj.get('keyframes', []):
                max_frame = max(max_frame, kf['frame'])
    
    total_frames = max_frame + 1
    fps = 10 # Default for V5 pipeline assumption if not stored
    
    # Video setup
    size = 900
    scale = 18  # pixels per meter
    center = size // 2
    
    out_video = out / "reconstruction_v5.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (size, size))
    
    # Colors
    STATIC_COLOR = (0, 200, 200) # Yellow-ish
    DYN_COLOR = (255, 100, 100) # Blue-ish
    
    # Pre-structure dynamic data: frame -> list of (obj, state)
    frame_map = {}
    for obj in objects:
        if not obj.get('static', False):
            for kf in obj.get('keyframes', []):
                fi = int(kf['frame'])
                if fi not in frame_map: frame_map[fi] = []
                frame_map[fi].append((obj, kf))
    
    print(f"Rendering {total_frames} frames...")
    
    for fi in tqdm(range(total_frames)):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        
        # Grid
        for i in range(-25, 26, 5):
            x = center + i * scale
            cv2.line(img, (x, 0), (x, size), (50, 50, 50), 1)
            cv2.line(img, (0, x), (size, x), (50, 50, 50), 1)
        
        cv2.circle(img, (center, center), 3, (100, 100, 100), -1)
        
        # Draw Static Objects (Always present)
        for obj in objects:
            if obj.get('static', False):
                pos = obj['position']
                yaw = obj.get('yaw', 0.0)
                
                draw_box(img, pos, yaw, STATIC_COLOR, obj['id'], center, scale)
        
        # Draw Dynamic Objects
        if fi in frame_map:
            for obj, kf in frame_map[fi]:
                pos = kf['position']
                rot = kf.get('rotation') # Quaternion [x,y,z,w]
                
                yaw = 0.0
                if rot:
                    r = R.from_quat(rot)
                    yaw = r.as_euler('zxy')[0]
                    
                draw_box(img, pos, yaw, DYN_COLOR, obj['id'], center, scale, fill=True)
        
        # Overlay Info
        cv2.putText(img, f"Frame: {fi}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        writer.write(img)
        
    writer.release()
    print(f"\nSaved: {out_video}")

def draw_box(img, pos, yaw, color, label, center, scale, fill=False):
    length = 4.5 * scale / 2
    width = 1.8 * scale / 2
    
    px = int(center + pos[0] * scale)
    py = int(center - pos[1] * scale)
    
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    corners = [(-length, -width), (length, -width), (length, width), (-length, width)]
    
    pts = []
    for cx_, cy_ in corners:
        rx = cx_ * cos_y - cy_ * sin_y
        ry = cx_ * sin_y + cy_ * cos_y
        pts.append([int(px + rx), int(py - ry)])
    pts = np.array(pts, dtype=np.int32)
    
    if fill:
        cv2.fillPoly(img, [pts], color)
        # Arrow
        ax = int(px + length * 1.5 * cos_y)
        ay = int(py - length * 1.5 * sin_y)
        cv2.arrowedLine(img, (px, py), (ax, ay), (255, 255, 255), 2, tipLength=0.3)
    else:
        cv2.polylines(img, [pts], True, color, 2)
        
    cv2.putText(img, label, (px-10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

if __name__ == "__main__":
    main()
