#!/usr/bin/env python3
"""
Create visualization with point cloud background and properly rotated/placed cars.
All cars rotated 90 degrees from previous orientation.
"""

import open3d as o3d
import numpy as np
import cv2
import json
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO

def main():
    base = Path(__file__).parent.parent
    out_dir = base / "outputs" / "pass2_dynamic_v4"
    
    # Load point cloud
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(base / 'outputs/pass1_static/pi3_pointcloud_corrected.ply'))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Create top-down view
    size = 900
    scale = 18
    cx, cy = size // 2, size // 2
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Draw point cloud (every 3rd point for speed)
    print("Drawing point cloud background...")
    for i in range(0, len(points), 3):
        pt = points[i]
        col = colors[i]
        px = int(cx + pt[0] * scale)
        py = int(cy - pt[1] * scale)
        
        if 0 <= px < size and 0 <= py < size:
            bgr = (int(col[2] * 255), int(col[1] * 255), int(col[0] * 255))
            img[py, px] = bgr
    
    # Load cameras and detect static objects
    print("\nLoading cameras...")
    with open(base / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    print("Loading YOLO...")
    yolo = YOLO('yolov8x.pt')
    bg_dir = base / "data/processed/static_backgrounds"
    
    # Detect vehicles
    all_dets = []
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    print("Detecting vehicles...")
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
            
            bottom_center_x = (bbox[0] + bbox[2]) / 2
            bottom_center_y = bbox[3]
            
            ray_cam = np.linalg.inv(K) @ np.array([bottom_center_x, bottom_center_y, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            ray_world = R @ ray_cam
            
            if abs(ray_world[2]) > 1e-6:
                s = -cam_pos[2] / ray_world[2]
                if s > 0:
                    pt = cam_pos + s * ray_world
                    if np.linalg.norm(pt[:2]) < 25:
                        all_dets.append({
                            'pos': pt[:2].copy(),
                            'conf': conf
                        })
    
    # Cluster detections
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
        static_objs.append({
            'pos': avg_pos, 
            'conf': best_conf, 
            'cams': len(cluster)
        })
    
    static_objs.sort(key=lambda x: (x['cams'], x['conf']), reverse=True)
    
    print(f"\nDetected {len(static_objs)} static objects")
    
    # Draw cars with 90° rotation (all cars perpendicular to before)
    # Cars should be oriented to parking spots along road edges
    print("Drawing cars (90° rotated from before)...")
    
    for i, obj in enumerate(static_objs[:15]):
        x, y = obj['pos']
        px = int(cx + x * scale)
        py = int(cy - y * scale)
        
        # All cars rotated 90° - now vertical cars become horizontal and vice versa
        # Based on position, determine if car should be horizontal or vertical
        # Looking at user's image: cars park perpendicular to road edges
        
        # Rotate all 90° from before means:
        # Before: top/bottom=0°, left/right=90°
        # After: top/bottom=90°, left/right=0°
        
        # Orientation: Top/Bottom = Horizontal (0), Left/Right = Vertical (90)
        if y < -5 or y > 12:  # Top/bottom edges
            yaw = 0.0
        else:  # Left/right edges
            yaw = np.pi / 2
        
        # Car dimensions
        length = 4.5 * scale / 2
        width = 1.8 * scale / 2
        
        # Draw rotated rectangle
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        corners = [(-length, -width), (length, -width), (length, width), (-length, width)]
        pts = []
        for cx_, cy_ in corners:
            rx = cx_ * cos_y - cy_ * sin_y
            ry = cx_ * sin_y + cy_ * cos_y
            pts.append([int(px + rx), int(py - ry)])
        pts = np.array(pts, dtype=np.int32)
        
        # Color by confidence
        if obj['cams'] >= 2:
            color = (0, 255, 255)  # Yellow - high confidence
        else:
            color = (0, 200, 200)  # Dimmer yellow
        
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, f"S{i+1}", (px-12, py-int(width)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        print(f"  S{i+1}: ({x:6.1f}, {y:6.1f}), yaw={np.degrees(yaw):.0f}°")
    
    # Save
    out_path = out_dir / "cars_on_pointcloud.png"
    cv2.imwrite(str(out_path), img)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
