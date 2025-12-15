#!/usr/bin/env python3
"""
Static Layout with L-Shape Filtering
=====================================
Only keeps parked cars that form the expected L-shaped pattern:
- Bottom edge: Y < -2
- Right edge: X > 4 and -2 <= Y <= 10
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

def matches_l_shape(x, y):
    """Check if position matches expected L-shape parking pattern."""
    # Bottom edge
    if y < -2:
        return True, "bottom"
    # Right edge  
    if x > 4 and -2 <= y <= 10:
        return True, "right"
    return False, None

def main():
    base = Path(__file__).parent.parent
    out_dir = base / "outputs" / "pass2_dynamic_v4"
    
    # Load cameras
    with open(base / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    # Detect vehicles
    print("Loading YOLO...")
    yolo = YOLO('yolov8x.pt')
    bg_dir = base / "data/processed/static_backgrounds"
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    all_dets = []
    
    print("Detecting static vehicles from backgrounds...")
    for cam_id in cam_ids:
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists() or cam_id not in cameras:
            continue
        
        print(f"  {cam_id}...")
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
    
    print(f"\nTotal raw detections: {len(all_dets)}")
    
    # Cluster
    positions = np.array([d['pos'] for d in all_dets])
    db = DBSCAN(eps=2.0, min_samples=1).fit(positions)
    
    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        if label >= 0:
            clusters[label].append(all_dets[i])
    
    print(f"Clusters: {len(clusters)}")
    
    # Process clusters and filter for L-shape
    static_objs = []
    for cluster in clusters.values():
        avg_pos = np.mean([d['pos'] for d in cluster], axis=0)
        best_conf = max(d['conf'] for d in cluster)
        
        # Check if matches L-shape
        matches, edge = matches_l_shape(avg_pos[0], avg_pos[1])
        if matches:
            static_objs.append({
                'pos': avg_pos, 
                'conf': best_conf, 
                'cams': len(cluster),
                'edge': edge
            })
    
    static_objs.sort(key=lambda x: (x['cams'], x['conf']), reverse=True)
    
    print(f"\nL-shape cars: {len(static_objs)}")
    for i, obj in enumerate(static_objs):
        print(f"  S{i+1}: ({obj['pos'][0]:6.1f}, {obj['pos'][1]:6.1f}), edge={obj['edge']}, cams={obj['cams']}")
    
    # Create visualization
    print("\nGenerating visualization...")
    size = 900
    scale = 18
    cx, cy = size // 2, size // 2
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)
    
    # Draw L-shape road outline
    # Bottom edge line (Y = -2)
    y_line = int(cy - (-2) * scale)
    cv2.line(img, (0, y_line), (size, y_line), (50, 80, 50), 2)
    cv2.putText(img, "Bottom Edge (Y<-2)", (10, y_line + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 80, 50), 1)
    
    # Right edge line (X = 4)
    x_line = int(cx + 4 * scale)
    cv2.line(img, (x_line, 0), (x_line, y_line), (50, 50, 80), 2)
    cv2.putText(img, "Right Edge (X>4)", (x_line + 5, cy - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 80), 1)
    
    # Draw grid
    for i in range(-20, 21, 5):
        x = cx + i * scale
        cv2.line(img, (x, 0), (x, size), (35, 35, 35), 1)
        cv2.line(img, (0, x), (size, x), (35, 35, 35), 1)
        cv2.putText(img, str(i), (x+2, cy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (60, 60, 60), 1)
    
    # Draw axes
    cv2.line(img, (cx, 0), (cx, size), (60, 60, 60), 1)
    cv2.line(img, (0, cy), (size, cy), (60, 60, 60), 1)
    cv2.putText(img, "0,0", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    # Draw parked cars
    for i, obj in enumerate(static_objs[:15]):
        px = int(cx + obj['pos'][0] * scale)
        py = int(cy - obj['pos'][1] * scale)
        
        # Orientation based on edge
        if obj['edge'] == 'bottom':
            yaw = 0  # Horizontal
            color = (0, 165, 255)  # Orange for bottom
        else:
            yaw = 0  # Also horizontal for right edge (parallel parking)
            color = (255, 165, 0)  # Blue for right
        
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
        
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, f"S{i+1}", (px-10, py-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Count by edge
    bottom_count = sum(1 for o in static_objs[:15] if o['edge'] == 'bottom')
    right_count = sum(1 for o in static_objs[:15] if o['edge'] == 'right')
    
    # Legend
    cv2.putText(img, "Static Cars - L-Shape Filtered", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Bottom edge (orange): {bottom_count} cars", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    cv2.putText(img, f"Right edge (blue): {right_count} cars", (10, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    cv2.putText(img, f"Total: {len(static_objs[:15])} cars", (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Save
    out_path = out_dir / "static_l_shape.png"
    cv2.imwrite(str(out_path), img)
    print(f"\nSaved: {out_path}")
    
    # Also save as JSON for pipeline
    static_data = []
    for i, obj in enumerate(static_objs[:15]):
        static_data.append({
            'pos': [float(obj['pos'][0]), float(obj['pos'][1])],
            'edge': obj['edge'],
            'yaw': 0.0,  # Horizontal
            'conf': float(obj['conf'])
        })
    
    json_path = out_dir / "static_objects.json"
    with open(json_path, 'w') as f:
        json.dump(static_data, f, indent=2)
    print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()
