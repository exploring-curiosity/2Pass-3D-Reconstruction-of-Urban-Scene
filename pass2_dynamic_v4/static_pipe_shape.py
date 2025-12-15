#!/usr/bin/env python3
"""
Static Layout with Pipe-Shape =||= Pattern
==========================================
Places parked cars along all road edges forming a pipe shape:
- Bottom edge: horizontal road at Y < -10
- Top edge: horizontal road at Y > 15
- Left edge: vertical road at X < -10
- Right edge: vertical road at X > 15
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

def matches_pipe_shape(x, y):
    """Check if position matches expected pipe-shape parking pattern.
    
    Pipe shape =||= has roads on 4 sides:
    - Bottom edge: Y < -10, X between -5 and 15
    - Top edge: Y > 15, X between -8 and 12
    - Left edge: X < -8, Y between -10 and 10
    - Right edge: X > 12, Y between -12 and 12
    """
    # Bottom edge (horizontal road at bottom)
    if y < -8 and -6 < x < 16:
        return True, "bottom"
    # Top edge (horizontal road at top)
    if y > 12 and -10 < x < 14:
        return True, "top"
    # Left edge (vertical road at left)
    if x < -8 and -10 < y < 10:
        return True, "left"
    # Right edge (vertical road at right)
    if x > 10 and -12 < y < 14:
        return True, "right"
    
    return False, None

def get_orientation_for_edge(edge):
    """Get car orientation based on which edge it's parked on."""
    if edge in ("bottom", "top"):
        return 0.0  # Horizontal - car pointing along X axis
    else:  # left, right
        return np.pi / 2  # Vertical - car pointing along Y axis

def main():
    base = Path(__file__).parent.parent
    out_dir = base / "outputs" / "pass2_dynamic_v4"
    
    # Load cameras
    with open(base / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    # Load road layout for visualization
    print("Loading road layout from point cloud...")
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(base / 'outputs/pass1_static/pi3_pointcloud_corrected.ply'))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    ground_mask = np.abs(points[:, 2]) < 0.3
    ground_colors = colors[ground_mask]
    ground_pts = points[ground_mask]
    brightness = ground_colors.mean(axis=1)
    road_pts = ground_pts[brightness < 0.35]
    curb_pts = ground_pts[brightness > 0.5]
    
    # Detect vehicles
    print("\nLoading YOLO...")
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
    
    # Filter for pipe-shape
    static_objs = []
    for cluster in clusters.values():
        avg_pos = np.mean([d['pos'] for d in cluster], axis=0)
        best_conf = max(d['conf'] for d in cluster)
        
        matches, edge = matches_pipe_shape(avg_pos[0], avg_pos[1])
        if matches:
            yaw = get_orientation_for_edge(edge)
            static_objs.append({
                'pos': avg_pos, 
                'conf': best_conf, 
                'cams': len(cluster),
                'edge': edge,
                'yaw': yaw
            })
    
    static_objs.sort(key=lambda x: (x['cams'], x['conf']), reverse=True)
    
    # Count by edge
    edge_counts = defaultdict(int)
    for obj in static_objs:
        edge_counts[obj['edge']] += 1
    
    print(f"\nPipe-shape cars: {len(static_objs)}")
    print(f"  Bottom: {edge_counts['bottom']}, Top: {edge_counts['top']}")
    print(f"  Left: {edge_counts['left']}, Right: {edge_counts['right']}")
    
    for i, obj in enumerate(static_objs[:20]):
        print(f"  S{i+1}: ({obj['pos'][0]:6.1f}, {obj['pos'][1]:6.1f}), edge={obj['edge']}, cams={obj['cams']}")
    
    # Create visualization with road overlay
    print("\nGenerating visualization...")
    size = 900
    scale = 18
    cx_img, cy_img = size // 2, size // 2
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)
    
    # Draw road points
    for pt in road_pts[::5]:
        px = int(cx_img + pt[0] * scale)
        py = int(cy_img - pt[1] * scale)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (50, 50, 50)
    
    # Draw curb points
    for pt in curb_pts[::10]:
        px = int(cx_img + pt[0] * scale)
        py = int(cy_img - pt[1] * scale)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (0, 50, 0)
    
    # Draw grid
    for i in range(-25, 26, 5):
        x = cx_img + i * scale
        cv2.line(img, (x, 0), (x, size), (35, 35, 35), 1)
        cv2.line(img, (0, x), (size, x), (35, 35, 35), 1)
    
    # Draw axes
    cv2.line(img, (cx_img, 0), (cx_img, size), (60, 60, 60), 1)
    cv2.line(img, (0, cy_img), (size, cy_img), (60, 60, 60), 1)
    
    # Color scheme for edges
    edge_colors = {
        'bottom': (0, 165, 255),  # Orange
        'top': (0, 255, 255),     # Yellow
        'left': (255, 100, 100),  # Light blue
        'right': (255, 165, 0),   # Blue
    }
    
    # Draw parked cars
    for i, obj in enumerate(static_objs[:15]):
        px = int(cx_img + obj['pos'][0] * scale)
        py = int(cy_img - obj['pos'][1] * scale)
        
        yaw = obj['yaw']
        color = edge_colors.get(obj['edge'], (200, 200, 200))
        
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
        cv2.putText(img, f"S{i+1}", (px-10, py-25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    # Legend
    cv2.putText(img, "Pipe-Shape Parking =||=", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Bottom:{edge_counts['bottom']} Top:{edge_counts['top']} Left:{edge_counts['left']} Right:{edge_counts['right']}", 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(img, f"Total: {len(static_objs)}", (10, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    # Save
    out_path = out_dir / "static_pipe_shape.png"
    cv2.imwrite(str(out_path), img)
    print(f"\nSaved: {out_path}")
    
    # Save as JSON
    static_data = []
    for i, obj in enumerate(static_objs[:15]):
        static_data.append({
            'pos': [float(obj['pos'][0]), float(obj['pos'][1])],
            'edge': obj['edge'],
            'yaw': float(obj['yaw']),
            'conf': float(obj['conf'])
        })
    
    json_path = out_dir / "static_objects_pipe.json"
    with open(json_path, 'w') as f:
        json.dump(static_data, f, indent=2)
    print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()
