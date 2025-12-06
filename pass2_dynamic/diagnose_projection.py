#!/usr/bin/env python3
"""
Diagnose 3D projection accuracy.

This script:
1. Takes a single frame where an object is visible in multiple cameras
2. Projects the object's bottom-center from each camera to 3D
3. Visualizes how much the projections disagree
4. Identifies sources of error
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


class CameraProjector:
    """Projects 2D image points to 3D ground plane.
    
    IMPORTANT: camera_params contains pose_c2w (camera-to-world transform).
    """
    
    def __init__(self, camera_params: dict, image_size: Tuple[int, int]):
        self.K = np.array(camera_params['K']).reshape(3, 3)
        
        # pose_c2w: camera-to-world transform
        pose_c2w = np.array(camera_params['pose_c2w'])
        self.R_c2w = pose_c2w[:3, :3]  # Camera-to-world rotation
        self.cam_pos = pose_c2w[:3, 3]  # Camera position in world
        
        self.image_size = image_size
        
    def project_to_ground(self, pixel: np.ndarray, ground_z: float = 0.0) -> Optional[np.ndarray]:
        """Project a 2D pixel to the ground plane (z=ground_z)."""
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        
        # Ray direction in camera coordinates (normalized)
        ray_cam = np.linalg.inv(self.K) @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform ray to world coordinates
        ray_world = self.R_c2w @ ray_cam
        
        # Find intersection with ground plane z = ground_z
        if abs(ray_world[2]) < 1e-6:
            return None
            
        s = (ground_z - self.cam_pos[2]) / ray_world[2]
        
        if s < 0:  # Behind camera
            return None
            
        point_3d = self.cam_pos + s * ray_world
        return point_3d


def diagnose_projection(
    video_dir: Path,
    camera_params_path: Path,
    frame_idx: int = 100,  # Frame to analyze
    ground_z: float = 0.0
):
    """Diagnose projection accuracy."""
    
    print("=" * 60)
    print("3D Projection Diagnosis")
    print("=" * 60)
    
    # Load camera params
    with open(camera_params_path) as f:
        camera_params = json.load(f)
    
    # Load YOLO
    print("\nLoading YOLO model...")
    model = YOLO('yolov8x.pt')
    
    # Get video paths
    cam_order = ['s1-left', 's1-right', 's2-left', 's2-right', 
                 's3-left', 's3-right', 's4-left', 's4-right']
    
    video_paths = {}
    for cam_id in cam_order:
        path = video_dir / f"{cam_id}.mp4"
        if path.exists():
            video_paths[cam_id] = path
    
    # Get video info
    first_cap = cv2.VideoCapture(str(list(video_paths.values())[0]))
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()
    
    # Create projectors
    projectors = {
        cam_id: CameraProjector(camera_params[cam_id], (width, height))
        for cam_id in video_paths.keys()
    }
    
    # Read frame from each camera and detect objects
    print(f"\nAnalyzing frame {frame_idx}...")
    
    detections_by_cam = {}
    frames = {}
    
    for cam_id, video_path in video_paths.items():
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        frames[cam_id] = frame
        
        # Detect objects
        results = model.predict(frame, conf=0.5, verbose=False, classes=[2, 5, 7])  # car, bus, truck
        
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()
            
            for i in range(len(xyxys)):
                bbox = xyxys[i]
                cls_name = model.names[cls_ids[i]]
                conf = confs[i]
                
                # Bottom center
                bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                
                # Project to 3D
                projector = projectors[cam_id]
                pos_3d = projector.project_to_ground(bottom_center, ground_z)
                
                if pos_3d is not None:
                    detections.append({
                        'bbox': bbox,
                        'class': cls_name,
                        'conf': conf,
                        'bottom_center': bottom_center,
                        'pos_3d': pos_3d
                    })
        
        detections_by_cam[cam_id] = detections
        print(f"  {cam_id}: {len(detections)} vehicles detected")
    
    # Print all 3D positions
    print("\n" + "=" * 60)
    print("3D Positions from each camera:")
    print("=" * 60)
    
    all_positions = []
    for cam_id in cam_order:
        if cam_id not in detections_by_cam:
            continue
        print(f"\n{cam_id}:")
        for det in detections_by_cam[cam_id]:
            pos = det['pos_3d']
            print(f"  {det['class']}: 3D=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]  conf={det['conf']:.2f}")
            all_positions.append((cam_id, det['class'], pos))
    
    # Analyze clustering - which detections might be the same object?
    print("\n" + "=" * 60)
    print("Potential same-object clusters (within 5m):")
    print("=" * 60)
    
    # Simple clustering by distance
    used = set()
    clusters = []
    
    for i, (cam1, cls1, pos1) in enumerate(all_positions):
        if i in used:
            continue
        
        cluster = [(cam1, cls1, pos1)]
        used.add(i)
        
        for j, (cam2, cls2, pos2) in enumerate(all_positions):
            if j in used or j <= i:
                continue
            if cam1 == cam2:  # Same camera, different object
                continue
            
            dist = np.linalg.norm(pos1[:2] - pos2[:2])
            if dist < 5.0:  # Within 5 meters
                cluster.append((cam2, cls2, pos2))
                used.add(j)
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        positions = []
        for cam_id, cls_name, pos in cluster:
            print(f"  {cam_id}: {cls_name} at [{pos[0]:.2f}, {pos[1]:.2f}]")
            positions.append(pos[:2])
        
        positions = np.array(positions)
        spread = np.std(positions, axis=0)
        print(f"  → Position spread: std_x={spread[0]:.2f}m, std_y={spread[1]:.2f}m")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)
    
    # Create a top-down view of all projected points
    fig_size = 800
    margin = 50
    scale = 15  # pixels per meter
    
    # Find bounds
    all_xy = [pos[:2] for _, _, pos in all_positions]
    if all_xy:
        all_xy = np.array(all_xy)
        center = np.mean(all_xy, axis=0)
    else:
        center = np.array([0, 0])
    
    # Create image
    img = np.ones((fig_size, fig_size, 3), dtype=np.uint8) * 255
    
    # Draw grid
    for i in range(-20, 21, 5):
        x = int(fig_size/2 + i * scale)
        cv2.line(img, (x, 0), (x, fig_size), (220, 220, 220), 1)
        cv2.line(img, (0, x), (fig_size, x), (220, 220, 220), 1)
    
    # Draw axes
    cv2.line(img, (fig_size//2, 0), (fig_size//2, fig_size), (200, 200, 200), 2)
    cv2.line(img, (0, fig_size//2), (fig_size, fig_size//2), (200, 200, 200), 2)
    
    # Colors for each camera
    cam_colors = {
        's1-left': (255, 0, 0),    # Blue
        's1-right': (255, 100, 0), # Light blue
        's2-left': (0, 255, 0),    # Green
        's2-right': (0, 255, 100), # Light green
        's3-left': (0, 0, 255),    # Red
        's3-right': (100, 0, 255), # Pink
        's4-left': (0, 255, 255),  # Yellow
        's4-right': (100, 255, 255), # Light yellow
    }
    
    # Draw camera positions
    for cam_id, params in camera_params.items():
        R = np.array(params['R']).reshape(3, 3)
        t = np.array(params['t']).reshape(3, 1)
        C = -R.T @ t
        
        px = int(fig_size/2 + (C[0,0] - center[0]) * scale)
        py = int(fig_size/2 - (C[1,0] - center[1]) * scale)
        
        color = cam_colors.get(cam_id, (128, 128, 128))
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.putText(img, cam_id, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw projected points
    for cam_id, cls_name, pos in all_positions:
        px = int(fig_size/2 + (pos[0] - center[0]) * scale)
        py = int(fig_size/2 - (pos[1] - center[1]) * scale)
        
        color = cam_colors.get(cam_id, (128, 128, 128))
        cv2.circle(img, (px, py), 5, color, -1)
    
    # Draw cluster ellipses
    for cluster in clusters:
        positions = np.array([pos[:2] for _, _, pos in cluster])
        if len(positions) >= 2:
            mean_pos = np.mean(positions, axis=0)
            px = int(fig_size/2 + (mean_pos[0] - center[0]) * scale)
            py = int(fig_size/2 - (mean_pos[1] - center[1]) * scale)
            
            # Draw circle around cluster
            max_dist = np.max(np.linalg.norm(positions - mean_pos, axis=1))
            radius = int(max_dist * scale) + 5
            cv2.circle(img, (px, py), radius, (0, 0, 0), 2)
    
    # Add legend
    y = 30
    for cam_id, color in cam_colors.items():
        cv2.circle(img, (20, y), 5, color, -1)
        cv2.putText(img, cam_id, (30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 20
    
    # Add title
    cv2.putText(img, f"Frame {frame_idx} - 3D Projections (1 grid = 5m)", 
                (fig_size//2 - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save
    output_path = Path(__file__).parent.parent / "outputs" / "pass2_dynamic" / "projection_diagnosis.png"
    cv2.imwrite(str(output_path), img)
    print(f"\nSaved: {output_path}")
    
    # Also save annotated camera frames
    grid_h, grid_w = 360, 480
    grid = np.zeros((grid_h * 4, grid_w * 2, 3), dtype=np.uint8)
    
    for i, cam_id in enumerate(cam_order):
        if cam_id not in frames:
            continue
        
        frame = frames[cam_id].copy()
        
        # Draw detections
        for det in detections_by_cam.get(cam_id, []):
            bbox = det['bbox'].astype(int)
            pos = det['pos_3d']
            color = cam_colors.get(cam_id, (128, 128, 128))
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Show 3D position
            label = f"[{pos[0]:.1f}, {pos[1]:.1f}]"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add camera label
        cv2.putText(frame, cam_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Resize and place in grid
        frame_small = cv2.resize(frame, (grid_w, grid_h))
        row = i // 2
        col = i % 2
        grid[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w] = frame_small
    
    grid_path = Path(__file__).parent.parent / "outputs" / "pass2_dynamic" / "projection_diagnosis_frames.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"Saved: {grid_path}")
    
    return detections_by_cam, clusters


def main():
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "StreetAware-sample"
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    
    # Try multiple frames
    for frame_idx in [50, 100, 200, 300]:
        print(f"\n\n{'#' * 60}")
        print(f"FRAME {frame_idx}")
        print(f"{'#' * 60}")
        diagnose_projection(video_dir, cameras_path, frame_idx)


if __name__ == "__main__":
    main()
