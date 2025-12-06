#!/usr/bin/env python3
"""
Cross-Camera Object Tracking with Visual Verification

This script:
1. Runs per-camera detection and tracking with locked classes
2. Projects detections to 3D ground plane
3. Associates tracks across cameras using 3D position overlap
4. Generates a combined 8-camera grid video showing GLOBAL track IDs
   so you can visually verify cross-camera association

The output video shows all 8 cameras simultaneously with:
- Same color = same global track ID across cameras
- Global ID displayed on each detection
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import colorsys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        sat = 0.9
        val = 0.9
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))  # BGR
    return colors


class CameraProjector:
    """Projects 2D image points to 3D ground plane.
    
    IMPORTANT: camera_params contains pose_c2w (camera-to-world transform).
    R is the rotation matrix that transforms camera frame to world frame.
    t is the camera position in world coordinates.
    """
    
    def __init__(self, camera_params: dict, image_size: Tuple[int, int]):
        self.K = np.array(camera_params['K']).reshape(3, 3)
        
        # pose_c2w: camera-to-world transform
        # R transforms vectors from camera frame to world frame
        # t is camera position in world frame
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
        
        # Transform ray to world coordinates using camera-to-world rotation
        ray_world = self.R_c2w @ ray_cam
        
        # Find intersection with ground plane z = ground_z
        # Line: P = cam_pos + s * ray_world
        # Ground: P[2] = ground_z
        # Solve: cam_pos[2] + s * ray_world[2] = ground_z
        
        if abs(ray_world[2]) < 1e-6:
            return None  # Ray parallel to ground
            
        s = (ground_z - self.cam_pos[2]) / ray_world[2]
        
        if s < 0:  # Intersection behind camera
            return None
            
        point_3d = self.cam_pos + s * ray_world
        return point_3d  # Return full 3D point


def run_cross_camera_tracking(
    video_dir: Path,
    output_dir: Path,
    camera_params_path: Path,
    ground_z: float = 0.0,
    association_threshold: float = 3.0  # meters
):
    """
    Run cross-camera tracking and generate verification video.
    """
    print("=" * 60)
    print("Cross-Camera Object Tracking")
    print("=" * 60)
    
    # Load camera params
    with open(camera_params_path) as f:
        camera_params = json.load(f)
    
    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO('yolov8x.pt')
    
    # Get video paths in consistent order
    cam_order = ['s1-left', 's1-right', 's2-left', 's2-right', 
                 's3-left', 's3-right', 's4-left', 's4-right']
    video_paths = {}
    for cam_id in cam_order:
        path = video_dir / f"{cam_id}.mp4"
        if path.exists():
            video_paths[cam_id] = path
    
    # Get video info
    first_cap = cv2.VideoCapture(str(list(video_paths.values())[0]))
    total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()
    
    print(f"Processing {len(video_paths)} cameras, {total_frames} frames @ {fps:.1f} fps")
    
    # Create projectors
    projectors = {
        cam_id: CameraProjector(camera_params[cam_id], (width, height))
        for cam_id in video_paths.keys()
        if cam_id in camera_params
    }
    
    # ============================================
    # PHASE 1: Per-camera detection and tracking
    # ============================================
    print("\n" + "=" * 60)
    print("PHASE 1: Per-camera tracking with locked classes")
    print("=" * 60)
    
    # Structure: {cam_id: {local_track_id: {'class': str, 'frames': {frame_idx: (bbox, pos_3d, conf)}}}}
    per_camera_tracks: Dict[str, Dict[int, Dict]] = {}
    
    for cam_id, video_path in video_paths.items():
        if cam_id not in projectors:
            continue
            
        print(f"\n  [{cam_id}]")
        projector = projectors[cam_id]
        
        # Pass 1A: Collect detections
        print(f"    Collecting detections...")
        all_detections = {}
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="      ", leave=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(
                frame, conf=0.3, iou=0.5, verbose=False,
                classes=[0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
            )
            
            frame_dets = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.int().cpu().numpy()
                
                for i in range(len(xyxys)):
                    bbox = xyxys[i].copy()
                    cls_name = model.names[cls_ids[i]]
                    conf = confs[i]
                    
                    # Project to 3D
                    bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                    pos_3d = projector.project_to_ground(bottom_center, ground_z)
                    
                    if pos_3d is not None and abs(pos_3d[0]) < 50 and abs(pos_3d[1]) < 50:
                        frame_dets.append((bbox, cls_name, conf, pos_3d))
            
            all_detections[frame_idx] = frame_dets
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Pass 1B: Build tracks with IoU matching
        print(f"    Building tracks...")
        tracks: Dict[int, Dict[int, Tuple]] = {}
        next_track_id = 1
        active_tracks: Dict[int, Tuple[np.ndarray, int]] = {}
        track_velocities: Dict[int, np.ndarray] = {}
        
        for frame_idx in range(len(all_detections)):
            dets = all_detections[frame_idx]
            if not dets:
                continue
            
            matched_dets = set()
            
            if active_tracks:
                track_ids = list(active_tracks.keys())
                costs = np.zeros((len(track_ids), len(dets)))
                
                for ti, track_id in enumerate(track_ids):
                    last_bbox, last_frame = active_tracks[track_id]
                    frames_gap = frame_idx - last_frame
                    
                    velocity = track_velocities.get(track_id)
                    if velocity is not None:
                        predicted = last_bbox.copy()
                        predicted[0] += velocity[0] * frames_gap
                        predicted[1] += velocity[1] * frames_gap
                        predicted[2] += velocity[0] * frames_gap
                        predicted[3] += velocity[1] * frames_gap
                    else:
                        predicted = last_bbox
                    
                    for di, (det_bbox, _, _, _) in enumerate(dets):
                        iou = compute_iou(predicted, det_bbox)
                        costs[ti, di] = 1 - iou
                
                while True:
                    if costs.size == 0:
                        break
                    min_idx = np.unravel_index(np.argmin(costs), costs.shape)
                    min_cost = costs[min_idx]
                    
                    if min_cost > 0.7:
                        break
                    
                    ti, di = min_idx
                    track_id = track_ids[ti]
                    det_bbox, det_cls, det_conf, det_pos = dets[di]
                    
                    tracks[track_id][frame_idx] = (det_bbox, det_cls, det_conf, det_pos)
                    
                    last_bbox, last_frame = active_tracks[track_id]
                    if frame_idx > last_frame:
                        dx = (det_bbox[0] + det_bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2
                        dy = (det_bbox[1] + det_bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2
                        dt = frame_idx - last_frame
                        new_vel = np.array([dx / dt, dy / dt])
                        if track_id in track_velocities:
                            track_velocities[track_id] = 0.7 * track_velocities[track_id] + 0.3 * new_vel
                        else:
                            track_velocities[track_id] = new_vel
                    
                    active_tracks[track_id] = (det_bbox, frame_idx)
                    matched_dets.add(di)
                    costs[ti, :] = np.inf
                    costs[:, di] = np.inf
            
            for di, (det_bbox, det_cls, det_conf, det_pos) in enumerate(dets):
                if di not in matched_dets:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {frame_idx: (det_bbox, det_cls, det_conf, det_pos)}
                    active_tracks[track_id] = (det_bbox, frame_idx)
            
            stale = [tid for tid, (_, last_f) in active_tracks.items() if frame_idx - last_f > 30]
            for tid in stale:
                del active_tracks[tid]
                if tid in track_velocities:
                    del track_velocities[tid]
        
        # Pass 1C: Lock class at peak confidence
        print(f"    Locking classes...")
        per_camera_tracks[cam_id] = {}
        
        for track_id, track_frames in tracks.items():
            if len(track_frames) < 5:
                continue
            
            class_scores = defaultdict(float)
            for f, (bbox, cls_name, conf, pos_3d) in track_frames.items():
                class_scores[cls_name] += conf
            
            locked_class = max(class_scores.keys(), key=lambda c: class_scores[c])
            
            if 'bicycle' in class_scores and 'person' in class_scores:
                if class_scores['bicycle'] > class_scores['person'] * 0.3:
                    locked_class = 'bicycle'
            
            per_camera_tracks[cam_id][track_id] = {
                'class': locked_class,
                'frames': {f: (bbox, pos_3d, conf) for f, (bbox, _, conf, pos_3d) in track_frames.items()}
            }
        
        print(f"    → {len(per_camera_tracks[cam_id])} tracks")
    
    # ============================================
    # PHASE 2: Cross-camera association
    # Use PAIRED cameras (left-right) which view the same area
    # ============================================
    print("\n" + "=" * 60)
    print("PHASE 2: Cross-camera association")
    print("=" * 60)
    
    # Camera pairs that view the same area
    camera_pairs = [
        ('s1-left', 's1-right'),
        ('s2-left', 's2-right'),
        ('s3-left', 's3-right'),
        ('s4-left', 's4-right'),
    ]
    
    # Build global tracks
    local_to_global: Dict[Tuple[str, int], int] = {}
    global_tracks: Dict[int, Dict] = {}
    next_global_id = 1
    
    # First, associate within camera pairs (left-right see same objects)
    # Use bbox position in overlapping field of view, not 3D projection
    print("\n  Step 1: Associate within camera pairs...")
    
    for left_cam, right_cam in camera_pairs:
        if left_cam not in per_camera_tracks or right_cam not in per_camera_tracks:
            continue
        
        print(f"\n    Pairing {left_cam} <-> {right_cam}")
        
        left_tracks = per_camera_tracks[left_cam]
        right_tracks = per_camera_tracks[right_cam]
        
        # For each left track, find best matching right track
        matched_right = set()
        
        for left_id, left_data in left_tracks.items():
            if len(left_data['frames']) < 10:
                continue
            
            left_frames = set(left_data['frames'].keys())
            left_class = left_data['class']
            
            best_match = None
            best_score = -1
            
            for right_id, right_data in right_tracks.items():
                if right_id in matched_right:
                    continue
                if right_data['class'] != left_class:
                    continue
                if len(right_data['frames']) < 10:
                    continue
                
                right_frames = set(right_data['frames'].keys())
                
                # Check temporal overlap
                common = left_frames & right_frames
                overlap = len(common)
                if overlap < 10:
                    continue
                
                # Check bbox vertical position similarity (Y coordinate)
                # Objects at same real-world position should have similar Y in both views
                y_diffs = []
                size_ratios = []
                for f in list(common)[:50]:  # Sample frames
                    left_bbox = left_data['frames'][f][0]
                    right_bbox = right_data['frames'][f][0]
                    
                    # Normalized Y position (0-1)
                    left_y = (left_bbox[1] + left_bbox[3]) / 2 / height
                    right_y = (right_bbox[1] + right_bbox[3]) / 2 / height
                    y_diffs.append(abs(left_y - right_y))
                    
                    # Size ratio
                    left_h = left_bbox[3] - left_bbox[1]
                    right_h = right_bbox[3] - right_bbox[1]
                    if left_h > 0 and right_h > 0:
                        size_ratios.append(min(left_h, right_h) / max(left_h, right_h))
                
                median_y_diff = np.median(y_diffs)
                median_size_ratio = np.median(size_ratios) if size_ratios else 0
                
                # Good match: similar Y position and similar size
                # Y diff < 0.15 (15% of image height) and size ratio > 0.5
                if median_y_diff < 0.15 and median_size_ratio > 0.5:
                    score = overlap * median_size_ratio / (1 + median_y_diff)
                    if score > best_score:
                        best_match = right_id
                        best_score = score
            
            if best_match is not None:
                # Create global track with both
                global_id = next_global_id
                next_global_id += 1
                
                local_to_global[(left_cam, left_id)] = global_id
                local_to_global[(right_cam, best_match)] = global_id
                matched_right.add(best_match)
                
                global_tracks[global_id] = {
                    'class': left_class,
                    'cameras': {left_cam: left_id, right_cam: best_match}
                }
                print(f"      {left_cam}:{left_id} <-> {right_cam}:{best_match} = G{global_id} (score={best_score:.1f})")
            else:
                # Left track alone
                global_id = next_global_id
                next_global_id += 1
                local_to_global[(left_cam, left_id)] = global_id
                global_tracks[global_id] = {
                    'class': left_class,
                    'cameras': {left_cam: left_id}
                }
        
        # Add unmatched right tracks
        for right_id, right_data in right_tracks.items():
            if right_id in matched_right:
                continue
            if len(right_data['frames']) < 10:
                continue
            if (right_cam, right_id) in local_to_global:
                continue
            
            global_id = next_global_id
            next_global_id += 1
            local_to_global[(right_cam, right_id)] = global_id
            global_tracks[global_id] = {
                'class': right_data['class'],
                'cameras': {right_cam: right_id}
            }
    
    # Step 2: Try to associate across different camera positions (s1, s2, s3, s4)
    # This is harder because objects move between camera views
    print("\n  Step 2: Associate across camera positions...")
    
    # Group global tracks by class
    tracks_by_class = defaultdict(list)
    for gid, gdata in global_tracks.items():
        tracks_by_class[gdata['class']].append(gid)
    
    # For each class, try to merge tracks that have sequential temporal coverage
    merged = set()
    merge_map = {}
    
    for cls_name, gids in tracks_by_class.items():
        if len(gids) < 2:
            continue
        
        # Get frame ranges for each track
        track_ranges = {}
        for gid in gids:
            all_frames = set()
            for cam_id, local_id in global_tracks[gid]['cameras'].items():
                all_frames.update(per_camera_tracks[cam_id][local_id]['frames'].keys())
            if all_frames:
                track_ranges[gid] = (min(all_frames), max(all_frames))
        
        # Try to merge tracks with overlapping or adjacent frame ranges
        for i, gid1 in enumerate(gids):
            if gid1 in merged or gid1 not in track_ranges:
                continue
            
            r1 = track_ranges[gid1]
            
            for gid2 in gids[i+1:]:
                if gid2 in merged or gid2 not in track_ranges:
                    continue
                
                r2 = track_ranges[gid2]
                
                # Check for temporal overlap or adjacency
                overlap_start = max(r1[0], r2[0])
                overlap_end = min(r1[1], r2[1])
                
                if overlap_end >= overlap_start - 30:  # Allow 30 frame gap
                    # Check 3D position similarity in overlapping region
                    if overlap_end >= overlap_start:
                        # Get positions from both tracks
                        pos1 = {}
                        for cam_id, local_id in global_tracks[gid1]['cameras'].items():
                            for f, (_, pos, _) in per_camera_tracks[cam_id][local_id]['frames'].items():
                                if overlap_start <= f <= overlap_end:
                                    if f not in pos1:
                                        pos1[f] = []
                                    pos1[f].append(pos)
                        
                        pos2 = {}
                        for cam_id, local_id in global_tracks[gid2]['cameras'].items():
                            for f, (_, pos, _) in per_camera_tracks[cam_id][local_id]['frames'].items():
                                if overlap_start <= f <= overlap_end:
                                    if f not in pos2:
                                        pos2[f] = []
                                    pos2[f].append(pos)
                        
                        common = set(pos1.keys()) & set(pos2.keys())
                        if len(common) >= 5:
                            dists = []
                            for f in common:
                                p1 = np.mean(pos1[f], axis=0)
                                p2 = np.mean(pos2[f], axis=0)
                                dists.append(np.linalg.norm(p1 - p2))
                            
                            if np.median(dists) < association_threshold:
                                # Merge gid2 into gid1
                                merged.add(gid2)
                                merge_map[gid2] = gid1
                                global_tracks[gid1]['cameras'].update(global_tracks[gid2]['cameras'])
                                print(f"    Merged G{gid2} into G{gid1} ({cls_name}, dist={np.median(dists):.1f}m)")
    
    # Remove merged tracks and update mappings
    for old_gid in merged:
        del global_tracks[old_gid]
    
    for key, gid in list(local_to_global.items()):
        if gid in merge_map:
            local_to_global[key] = merge_map[gid]
    
    print(f"\n  Total: {len(global_tracks)} global tracks")
    
    # Print summary
    print("\n  Global Track Summary:")
    for gid, gdata in sorted(global_tracks.items()):
        cams = list(gdata['cameras'].keys())
        print(f"    G{gid}: {gdata['class']}, cameras: {cams}")
    
    # ============================================
    # PHASE 3: Generate verification video
    # ============================================
    print("\n" + "=" * 60)
    print("PHASE 3: Generating verification video")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate colors for global tracks
    colors = generate_distinct_colors(len(global_tracks) + 1)
    global_colors = {gid: colors[i] for i, gid in enumerate(global_tracks.keys())}
    
    # Open all video captures
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    # Output video (4x2 grid)
    grid_w, grid_h = 640, 360  # Size per camera
    output_w, output_h = grid_w * 2, grid_h * 4
    
    output_path = output_dir / "cross_camera_verification.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    
    print(f"\n  Rendering {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames), desc="  Rendering"):
        # Read frames from all cameras
        frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = cv2.resize(frame, (grid_w, grid_h))
            else:
                frames[cam_id] = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Draw detections with global IDs
        for cam_id in cam_order:
            if cam_id not in frames or cam_id not in per_camera_tracks:
                continue
            
            frame = frames[cam_id]
            
            for local_id, track_data in per_camera_tracks[cam_id].items():
                if frame_idx not in track_data['frames']:
                    continue
                
                bbox, pos_3d, conf = track_data['frames'][frame_idx]
                
                # Get global ID
                key = (cam_id, local_id)
                if key not in local_to_global:
                    continue
                
                global_id = local_to_global[key]
                color = global_colors.get(global_id, (128, 128, 128))
                
                # Scale bbox to grid size
                scale_x = grid_w / width
                scale_y = grid_h / height
                bbox_scaled = [
                    int(bbox[0] * scale_x), int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x), int(bbox[3] * scale_y)
                ]
                
                # Draw bbox
                cv2.rectangle(frame, (bbox_scaled[0], bbox_scaled[1]), 
                              (bbox_scaled[2], bbox_scaled[3]), color, 2)
                
                # Draw global ID (large, prominent)
                label = f"G{global_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                cv2.rectangle(frame,
                              (bbox_scaled[0], bbox_scaled[1] - label_size[1] - 8),
                              (bbox_scaled[0] + label_size[0] + 4, bbox_scaled[1]),
                              color, -1)
                cv2.putText(frame, label,
                            (bbox_scaled[0] + 2, bbox_scaled[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add camera label
            cv2.putText(frame, cam_id, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Arrange in 4x2 grid
        grid = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        for i, cam_id in enumerate(cam_order):
            if cam_id in frames:
                row = i // 2
                col = i % 2
                y1, y2 = row * grid_h, (row + 1) * grid_h
                x1, x2 = col * grid_w, (col + 1) * grid_w
                grid[y1:y2, x1:x2] = frames[cam_id]
        
        # Add frame counter
        cv2.putText(grid, f"Frame: {frame_idx}/{total_frames}", (10, output_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(grid)
    
    # Cleanup
    for cap in caps.values():
        cap.release()
    out.release()
    
    print(f"\n  Saved: {output_path}")
    
    # Save track data
    track_data_path = output_dir / "cross_camera_tracks.json"
    output_data = {
        'global_tracks': {
            str(gid): {
                'class': gdata['class'],
                'cameras': gdata['cameras']
            }
            for gid, gdata in global_tracks.items()
        },
        'local_to_global': {
            f"{cam_id}:{local_id}": global_id
            for (cam_id, local_id), global_id in local_to_global.items()
        }
    }
    with open(track_data_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved: {track_data_path}")
    
    return global_tracks, local_to_global


def main():
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "StreetAware-sample"
    output_dir = base_dir / "outputs" / "pass2_dynamic"
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    
    run_cross_camera_tracking(video_dir, output_dir, cameras_path)


if __name__ == "__main__":
    main()
