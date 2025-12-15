#!/usr/bin/env python3
"""
Create comprehensive static objects map with detected parking layout.
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from collections import defaultdict
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO

def main():
    base = Path(__file__).parent.parent
    out_dir = base / "outputs" / "pass2_dynamic_v4"
    
    # Load cameras
    with open(base / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    # Load masks
    mask_dir = base / "outputs/pass1_static/ground_masks"
    road_grid = np.load(mask_dir / "road_grid.npy")
    curb_grid = np.load(mask_dir / "curb_grid.npy")
    with open(mask_dir / "grid_info.json") as f:
        grid_info = json.load(f)
    
    # Detect vehicles
    yolo = YOLO('yolov8x.pt')
    bg_dir = base / "data/processed/static_backgrounds"
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    all_dets = []
    
    for cam_id in cam_ids:
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists() or cam_id not in cameras:
            continue
        
        bg = cv2.imread(str(bg_path))
        K = np.array(cameras[cam_id]['K']).reshape(3, 3)
        pose = np.array(cameras[cam_id]['pose_c2w'])
        R = pose[:3, :3]
        cam_pos = pose[:3, 3]
        
        results = yolo.predict(bg, conf=0.45, verbose=False, classes=[2, 3, 5, 7])
        
        if results[0].boxes is None:
            continue
        
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            
            ray_cam = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            ray_world = R @ ray_cam
            
            if abs(ray_world[2]) > 1e-6:
                s = -cam_pos[2] / ray_world[2]
                if s > 0:
                    pt = cam_pos + s * ray_world
                    if np.linalg.norm(pt[:2]) < 25:
                        all_dets.append({
                            'pos': pt[:2].copy(),
                            'conf': conf,
                            'aspect': bw/bh if bh > 0 else 1.0
                        })
    
    # Cluster
    positions = np.array([d['pos'] for d in all_dets])
    db = DBSCAN(eps=2.0, min_samples=1).fit(positions)
    
    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        if label >= 0:
            clusters[label].append(all_dets[i])
    
    static_objs = []
    for cluster in clusters.values():
        avg_pos = np.mean([d['pos'] for d in cluster], axis=0)
        best_conf = max(d['conf'] for d in cluster)
        static_objs.append({'pos': avg_pos, 'conf': best_conf, 'cams': len(cluster)})
    
    static_objs.sort(key=lambda x: (x['cams'], x['conf']), reverse=True)
    
    # Create visualization
    size = 900
    scale = 18
    cx, cy = size // 2, size // 2
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)
    
    # Draw road/curb
    res = grid_info['resolution']
    ox, oy = grid_info['origin']
    dim = grid_info['dim']
    
    for gy in range(dim):
        for gx in range(dim):
            wx = ox + gx * res
            wy = oy + gy * res
            sx = int(cx + wx * scale)
            sy = int(cy - wy * scale)
            if 0 <= sx < size and 0 <= sy < size:
                if road_grid[gy, gx]:
                    img[sy, sx] = (50, 50, 50)
                if curb_grid[gy, gx]:
                    img[sy, sx] = (0, 60, 0)
    
    # Draw grid
    for i in range(-20, 21, 5):
        x = cx + i * scale
        cv2.line(img, (x, 0), (x, size), (40, 40, 40), 1)
        cv2.line(img, (0, x), (size, x), (40, 40, 40), 1)
        cv2.putText(img, str(i), (x+2, cy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    
    # Draw axes
    cv2.line(img, (cx, 0), (cx, size), (70, 70, 70), 1)
    cv2.line(img, (0, cy), (size, cy), (70, 70, 70), 1)
    
    # Draw parked cars
    for i, obj in enumerate(static_objs[:15]):
        px = int(cx + obj['pos'][0] * scale)
        py = int(cy - obj['pos'][1] * scale)
        
        # Determine orientation based on position
        x, y = obj['pos']
        if y < -3:  # Bottom row - horizontal
            yaw = 0
        elif x > 5:  # Right side - could be either
            yaw = 0
        else:
            yaw = 0
        
        # Draw rectangle
        l, w = 4.5 * scale / 2, 1.8 * scale / 2
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        corners = [(-l, -w), (l, -w), (l, w), (-l, w)]
        pts = []
        for cx_, cy_ in corners:
            rx = cx_ * cos_y - cy_ * sin_y
            ry = cx_ * sin_y + cy_ * cos_y
            pts.append([int(px + rx), int(py - ry)])
        pts = np.array(pts, dtype=np.int32)
        
        # Color by camera count
        if obj['cams'] >= 3:
            color = (0, 255, 255)  # Yellow - high confidence
        elif obj['cams'] >= 2:
            color = (0, 200, 200)  
        else:
            color = (0, 150, 150)  # Dim yellow
        
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, f"S{i+1}", (px-10, py-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Legend
    cv2.putText(img, "Static Parked Cars Detection", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Found: {len(static_objs)} clusters, showing top 15", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(img, "Yellow = multi-cam detection (high conf)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Save
    out_path = out_dir / "static_layout.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")
    
    # Also print positions
    print("\nStatic car positions:")
    for i, obj in enumerate(static_objs[:15]):
        print(f"  S{i+1}: ({obj['pos'][0]:6.1f}, {obj['pos'][1]:6.1f}), cams={obj['cams']}, conf={obj['conf']:.2f}")

if __name__ == "__main__":
    main()
