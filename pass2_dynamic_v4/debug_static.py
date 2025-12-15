#!/usr/bin/env python3
"""
Debug Static Objects with Road Overlay
========================================
Creates a debug visualization showing:
1. Road/curb areas from segmentation
2. Detected static objects  
3. For self-verification against expected layout
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

def create_debug_image(cameras: dict, static_objects: list, road_grid, curb_grid, grid_info, save_path: Path):
    """Create debug image with roads and static objects."""
    
    size = 900
    scale = 15  # pixels per meter
    center = size // 2
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    
    # Draw road/curb from grids
    if road_grid is not None:
        resolution = grid_info['resolution']
        origin = grid_info['origin']
        dim = grid_info['dim']
        
        for gy in range(dim):
            for gx in range(dim):
                wx = origin[0] + gx * resolution
                wy = origin[1] + gy * resolution
                
                sx = int(center + wx * scale)
                sy = int(center - wy * scale)
                
                if 0 <= sx < size and 0 <= sy < size:
                    if road_grid[gy, gx]:
                        img[sy, sx] = (70, 70, 70)  # Gray for road
                    if curb_grid[gy, gx]:
                        img[sy, sx] = (0, 80, 0)  # Dark green for curb
    
    # Draw grid lines and labels
    for i in range(-25, 26, 5):
        x = center + i * scale
        cv2.line(img, (x, 0), (x, size), (50, 50, 50), 1)
        cv2.line(img, (0, x), (size, x), (50, 50, 50), 1)
        
        cv2.putText(img, f"{i}", (x + 2, center + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
        cv2.putText(img, f"{-i}", (center + 5, x + 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    
    # Draw axes
    cv2.line(img, (center, 0), (center, size), (100, 100, 100), 1)
    cv2.line(img, (0, center), (size, center), (100, 100, 100), 1)
    
    # Draw origin marker
    cv2.circle(img, (center, center), 5, (255, 255, 255), -1)
    cv2.putText(img, "0,0", (center + 8, center - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw zone boundaries for reference
    # Bottom zone (Y < -5)
    y_bottom = int(center - (-5) * scale)
    cv2.line(img, (0, y_bottom), (size, y_bottom), (100, 50, 50), 1)
    cv2.putText(img, "Y=-5 (bottom zone below)", (10, y_bottom - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 50, 50), 1)
    
    # Right zone (X > 5)
    x_right = int(center + 5 * scale)
    cv2.line(img, (x_right, 0), (x_right, size), (50, 50, 100), 1)
    
    # Draw static objects with color coding by zone
    for i, obj in enumerate(static_objects[:15]):
        pos = obj['pos']
        yaw = obj.get('yaw', 0.0)
        
        px = int(center + pos[0] * scale)
        py = int(center - pos[1] * scale)
        
        # Determine zone and color
        x, y = pos
        if y < -5:  # Bottom zone
            color = (0, 100, 255)  # Orange
            zone = "B"
        elif x > 5 and y > -5:  # Right zone
            color = (255, 100, 0)  # Blue
            zone = "R"
        elif y > 15:  # Top zone
            color = (100, 255, 100)  # Light green
            zone = "T"
        else:
            color = (200, 200, 200)  # Gray - other
            zone = "?"
        
        # Car rectangle
        length = 4.5 * scale / 2
        width = 1.8 * scale / 2
        
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        corners = [
            (-length, -width), (length, -width),
            (length, width), (-length, width)
        ]
        
        pts = []
        for cx, cy in corners:
            rx = cx * cos_y - cy * sin_y
            ry = cx * sin_y + cy * cos_y
            pts.append([int(px + rx), int(py - ry)])
        
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], True, color, 2)
        
        # Label
        label = f"S{i+1}({zone})"
        cv2.putText(img, label, (px - 20, py - int(width) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    # Legend
    cv2.putText(img, "Road (gray) | Curb (green) | Static Cars (by zone)", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "B=Bottom(orange) R=Right(blue) T=Top(green)", (10, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    # Count by zone
    bottom = sum(1 for o in static_objects[:15] if o['pos'][1] < -5)
    right = sum(1 for o in static_objects[:15] if o['pos'][0] > 5 and o['pos'][1] > -5)
    top = sum(1 for o in static_objects[:15] if o['pos'][1] > 15)
    cv2.putText(img, f"Counts: Bottom={bottom}, Right={right}, Top={top}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imwrite(str(save_path), img)
    print(f"  Debug image saved: {save_path}")

def detect_static_from_backgrounds(bg_dir: Path, cameras: dict, yolo: YOLO):
    """Detect parked vehicles from backgrounds with improved clustering."""
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    all_dets = []
    
    for cam_id in cam_ids:
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists() or cam_id not in cameras:
            continue
        
        print(f"  {cam_id}...")
        
        bg_img = cv2.imread(str(bg_path))
        h, w = bg_img.shape[:2]
        
        K = np.array(cameras[cam_id]['K']).reshape(3, 3)
        pose = np.array(cameras[cam_id]['pose_c2w'])
        R_c2w = pose[:3, :3]
        cam_pos = pose[:3, 3]
        
        results = yolo.predict(bg_img, conf=0.4, verbose=False, classes=[2, 3, 5, 7])
        
        if results[0].boxes is None:
            continue
        
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu())
            cls_name = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}[cls_id]
            conf = float(box.conf[0].cpu())
            
            if conf < 0.45:
                continue
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]
            
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            aspect = bw / bh if bh > 0 else 1.0
            
            # Project to 3D
            ray_cam = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            ray_world = R_c2w @ ray_cam
            
            if abs(ray_world[2]) < 1e-6:
                continue
            
            s = -cam_pos[2] / ray_world[2]
            if s < 0:
                continue
            
            pos = cam_pos + s * ray_world
            
            if np.linalg.norm(pos[:2]) > 30:
                continue
            
            all_dets.append({
                'pos': pos[:2].copy(),
                'cls': cls_name,
                'conf': conf,
                'camera': cam_id,
                'aspect': aspect
            })
    
    print(f"  Total raw detections: {len(all_dets)}")
    
    if len(all_dets) < 2:
        return []
    
    # DBSCAN clustering
    positions = np.array([d['pos'] for d in all_dets])
    db = DBSCAN(eps=2.5, min_samples=1).fit(positions)
    labels = db.labels_
    
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            clusters[label].append(all_dets[i])
    
    print(f"  Clusters: {len(clusters)}")
    
    static_objects = []
    
    for cluster_id, cluster in clusters.items():
        avg_pos = np.mean([d['pos'] for d in cluster], axis=0)
        best_det = max(cluster, key=lambda x: x['conf'])
        num_cams = len(set(d['camera'] for d in cluster))
        avg_aspect = np.mean([d['aspect'] for d in cluster])
        
        # Determine yaw based on position
        x, y = avg_pos
        
        # Looking at the user's sketch:
        # - Bottom boundary has horizontal row of cars
        # - There's an L-shaped road with cars along edges
        if y < -4:  # Bottom area - horizontal
            yaw = 0.0
        elif x > 8 and y > 0:  # Right side - could be vertical
            yaw = np.pi / 2
        else:
            # Use aspect ratio
            if avg_aspect > 1.5:
                yaw = 0.0
            else:
                yaw = np.pi / 2
        
        static_objects.append({
            'pos': avg_pos,
            'cls': best_det['cls'],
            'conf': best_det['conf'],
            'num_cameras': num_cams,
            'yaw': yaw
        })
    
    static_objects.sort(key=lambda x: (x['num_cameras'], x['conf']), reverse=True)
    
    return static_objects

def main():
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "outputs" / "pass2_dynamic_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STATIC OBJECT DEBUG - With Road Overlay")
    print("=" * 60)
    
    # Load cameras
    with open(base_dir / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    # Load ground masks
    mask_dir = base_dir / "outputs/pass1_static/ground_masks"
    road_grid = curb_grid = grid_info = None
    
    if (mask_dir / "road_grid.npy").exists():
        road_grid = np.load(mask_dir / "road_grid.npy")
        curb_grid = np.load(mask_dir / "curb_grid.npy")
        with open(mask_dir / "grid_info.json") as f:
            grid_info = json.load(f)
        print(f"\nLoaded ground masks: {grid_info['dim']}x{grid_info['dim']}")
    
    # Detect static objects
    print("\nDetecting static objects...")
    yolo = YOLO('yolov8x.pt')
    bg_dir = base_dir / "data/processed/static_backgrounds"
    
    static_objects = detect_static_from_backgrounds(bg_dir, cameras, yolo)
    
    print(f"\nStatic objects found: {len(static_objects)}")
    for i, obj in enumerate(static_objects[:15]):
        print(f"  S{i+1}: pos=({obj['pos'][0]:6.1f}, {obj['pos'][1]:6.1f}), yaw={np.degrees(obj['yaw']):4.0f}°, cams={obj['num_cameras']}")
    
    # Create debug image
    debug_path = out_dir / "static_debug.png"
    create_debug_image(cameras, static_objects, road_grid, curb_grid, grid_info, debug_path)
    
    print("\nDone. Check static_debug.png")

if __name__ == "__main__":
    main()
