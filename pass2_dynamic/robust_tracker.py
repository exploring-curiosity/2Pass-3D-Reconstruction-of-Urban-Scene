#!/usr/bin/env python3
"""
Robust Multi-Camera 3D Tracker

Strategy:
1. Per-camera tracking with IoU-based association (handles detection gaps)
2. Multi-view triangulation when object visible in 2+ cameras
3. Kalman filter for position smoothing and prediction
4. Track interpolation to fill detection gaps
5. Post-processing to merge fragmented tracks

Key improvements over previous approach:
- Track objects per-camera first, then associate across cameras
- Use IoU tracking to maintain identity even when detection confidence varies
- Triangulate 3D position from multiple views when possible
- Smooth trajectories with Kalman filter
- Merge tracks that are clearly the same object
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import colorsys
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Only these 4 classes
VALID_CLASSES = {'car', 'truck', 'bicycle', 'person'}
CLASS_IDS = [0, 1, 2, 7]  # person=0, bicycle=1, car=2, truck=7


@dataclass
class KalmanTracker:
    """Kalman filter for 2D position tracking with velocity."""
    
    # State: [x, y, vx, vy]
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 100)
    
    # Process noise
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.5, 0.5, 1.0, 1.0]))
    # Measurement noise
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))
    
    def predict(self, dt: float = 1.0):
        """Predict next state."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q * dt
        return self.state[:2].copy()
    
    def update(self, measurement: np.ndarray):
        """Update with measurement [x, y]."""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Innovation
        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance
        return self.state[:2].copy()
    
    @property
    def position(self):
        return self.state[:2].copy()
    
    @property
    def velocity(self):
        return self.state[2:4].copy()


@dataclass
class PerCameraTrack:
    """Track within a single camera view."""
    track_id: int
    cam_id: str
    class_name: str
    class_confidence: float
    
    # Frame data: {frame_idx: (bbox, confidence)}
    detections: Dict[int, Tuple[np.ndarray, float]] = field(default_factory=dict)
    
    # Class voting
    class_votes: Dict[str, float] = field(default_factory=dict)
    
    last_seen: int = 0
    age: int = 0  # Total frames since creation
    
    def add_detection(self, frame_idx: int, bbox: np.ndarray, cls_name: str, conf: float):
        self.detections[frame_idx] = (bbox, conf)
        self.last_seen = frame_idx
        
        # Update class votes
        if cls_name not in self.class_votes:
            self.class_votes[cls_name] = 0
        self.class_votes[cls_name] += conf
        
        # Update locked class
        best_class = max(self.class_votes.keys(), key=lambda c: self.class_votes[c])
        self.class_name = best_class
        self.class_confidence = self.class_votes[best_class]
    
    def get_bbox(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx in self.detections:
            return self.detections[frame_idx][0]
        return None
    
    @property
    def frame_range(self):
        if not self.detections:
            return (0, 0)
        frames = list(self.detections.keys())
        return (min(frames), max(frames))


@dataclass
class GlobalTrack:
    """Track in 3D world coordinates, potentially seen by multiple cameras."""
    track_id: int
    class_name: str
    
    # Kalman filter for position
    kalman: KalmanTracker = field(default_factory=KalmanTracker)
    
    # Per-camera track associations: {cam_id: per_camera_track_id}
    camera_tracks: Dict[str, int] = field(default_factory=dict)
    
    # Frame data: {frame_idx: {'position': [x,y,z], 'cameras': {cam_id: bbox}}}
    frames: Dict[int, Dict] = field(default_factory=dict)
    
    last_seen: int = 0
    frames_since_seen: int = 0
    
    def get_position(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx in self.frames:
            return np.array(self.frames[frame_idx]['position'])
        return None


class CameraProjector:
    """Projects 2D image points to 3D ground plane and vice versa."""
    
    def __init__(self, camera_params: dict):
        self.K = np.array(camera_params['K']).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        
        pose_c2w = np.array(camera_params['pose_c2w'])
        self.R_c2w = pose_c2w[:3, :3]
        self.cam_pos = pose_c2w[:3, 3]
        
        # World to camera
        self.R_w2c = self.R_c2w.T
        self.t_w2c = -self.R_w2c @ self.cam_pos
        
    def project_to_ground(self, pixel: np.ndarray, ground_z: float = 0.0) -> Optional[np.ndarray]:
        """Project 2D pixel to 3D ground plane."""
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        ray_cam = self.K_inv @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = self.R_c2w @ ray_cam
        
        if abs(ray_world[2]) < 1e-6:
            return None
        s = (ground_z - self.cam_pos[2]) / ray_world[2]
        if s < 0:
            return None
        return self.cam_pos + s * ray_world
    
    def project_to_image(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D point to 2D image."""
        # Transform to camera coordinates
        p_cam = self.R_w2c @ point_3d + self.t_w2c
        if p_cam[2] <= 0:
            return None
        # Project
        p_img = self.K @ p_cam
        return p_img[:2] / p_img[2]
    
    def get_ray(self, pixel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ray origin and direction for a pixel."""
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        ray_cam = self.K_inv @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = self.R_c2w @ ray_cam
        return self.cam_pos.copy(), ray_world


def triangulate_point(projectors: Dict[str, CameraProjector], 
                      observations: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Triangulate 3D point from multiple 2D observations.
    Uses least-squares solution to find point closest to all rays.
    """
    if len(observations) < 2:
        return None
    
    # Build system of equations
    # For each ray: point = origin + t * direction
    # Minimize sum of squared distances from point to all rays
    
    A = []
    b = []
    
    for cam_id, pixel in observations.items():
        if cam_id not in projectors:
            continue
        origin, direction = projectors[cam_id].get_ray(pixel)
        
        # Distance from point P to ray: ||(P - origin) - ((P - origin) · direction) * direction||
        # This can be written as: (I - d*d^T) * (P - origin) = 0
        # Rearranging: (I - d*d^T) * P = (I - d*d^T) * origin
        
        d = direction.reshape(3, 1)
        M = np.eye(3) - d @ d.T
        A.append(M)
        b.append(M @ origin)
    
    if len(A) < 2:
        return None
    
    A = np.vstack(A)
    b = np.hstack(b)
    
    # Solve least squares
    try:
        point, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if solution is reasonable
        if abs(point[0]) > 50 or abs(point[1]) > 50 or point[2] < -1 or point[2] > 5:
            return None
        
        return point
    except:
        return None


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def run_robust_tracking(
    video_dir: Path,
    output_dir: Path,
    camera_params_path: Path,
    iou_threshold: float = 0.3,
    max_age: int = 30,  # Max frames to keep track without detection
    min_track_length: int = 15,
):
    print("=" * 70)
    print("Robust Multi-Camera 3D Tracker")
    print("=" * 70)
    
    with open(camera_params_path) as f:
        camera_params = json.load(f)
    
    print("\nLoading YOLO model...")
    model = YOLO('yolov8x.pt')
    
    cam_order = ['s1-left', 's1-right', 's2-left', 's2-right', 
                 's3-left', 's3-right', 's4-left', 's4-right']
    
    video_paths = {cam_id: video_dir / f"{cam_id}.mp4" 
                   for cam_id in cam_order if (video_dir / f"{cam_id}.mp4").exists()}
    
    first_cap = cv2.VideoCapture(str(list(video_paths.values())[0]))
    total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()
    
    print(f"Processing {len(video_paths)} cameras, {total_frames} frames @ {fps:.1f} fps")
    
    projectors = {cam_id: CameraProjector(camera_params[cam_id]) for cam_id in video_paths}
    
    # ========================================
    # PHASE 1: Per-camera IoU tracking
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: Per-camera IoU tracking")
    print("=" * 70)
    
    # Per-camera tracks: {cam_id: {track_id: PerCameraTrack}}
    per_camera_tracks: Dict[str, Dict[int, PerCameraTrack]] = {cam_id: {} for cam_id in video_paths}
    next_track_id = {cam_id: 1 for cam_id in video_paths}
    
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    for frame_idx in tqdm(range(total_frames), desc="Per-camera tracking"):
        for cam_id in cam_order:
            if cam_id not in caps:
                continue
            
            ret, frame = caps[cam_id].read()
            if not ret:
                continue
            
            # Detect
            results = model.predict(frame, conf=0.3, iou=0.5, verbose=False, classes=CLASS_IDS)
            
            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())
                    cls_name = model.names[cls_id]
                    if cls_name in VALID_CLASSES:
                        detections.append((bbox, cls_name, conf))
            
            # Get active tracks for this camera
            active_tracks = {
                tid: t for tid, t in per_camera_tracks[cam_id].items()
                if frame_idx - t.last_seen < max_age
            }
            
            # Match detections to tracks using IoU
            if active_tracks and detections:
                track_ids = list(active_tracks.keys())
                cost_matrix = np.zeros((len(detections), len(track_ids)))
                
                for di, (bbox, cls_name, conf) in enumerate(detections):
                    for ti, tid in enumerate(track_ids):
                        track = active_tracks[tid]
                        # Get last known bbox
                        last_bbox = track.get_bbox(track.last_seen)
                        if last_bbox is not None:
                            iou_score = iou(bbox, last_bbox)
                            # Penalize class mismatch
                            if track.class_name != cls_name:
                                if not (track.class_name in ['car', 'truck'] and cls_name in ['car', 'truck']):
                                    iou_score *= 0.5
                            cost_matrix[di, ti] = 1 - iou_score
                        else:
                            cost_matrix[di, ti] = 1
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matched_dets = set()
                matched_tracks = set()
                
                for di, ti in zip(row_ind, col_ind):
                    if cost_matrix[di, ti] < 1 - iou_threshold:
                        bbox, cls_name, conf = detections[di]
                        tid = track_ids[ti]
                        per_camera_tracks[cam_id][tid].add_detection(frame_idx, bbox, cls_name, conf)
                        matched_dets.add(di)
                        matched_tracks.add(ti)
            else:
                matched_dets = set()
            
            # Create new tracks for unmatched detections
            for di, (bbox, cls_name, conf) in enumerate(detections):
                if di not in matched_dets:
                    tid = next_track_id[cam_id]
                    next_track_id[cam_id] += 1
                    track = PerCameraTrack(
                        track_id=tid,
                        cam_id=cam_id,
                        class_name=cls_name,
                        class_confidence=conf
                    )
                    track.add_detection(frame_idx, bbox, cls_name, conf)
                    per_camera_tracks[cam_id][tid] = track
            
            # Age all tracks
            for track in per_camera_tracks[cam_id].values():
                track.age += 1
    
    for cap in caps.values():
        cap.release()
    
    # Filter short tracks
    for cam_id in per_camera_tracks:
        per_camera_tracks[cam_id] = {
            tid: t for tid, t in per_camera_tracks[cam_id].items()
            if len(t.detections) >= min_track_length
        }
    
    total_per_cam = sum(len(tracks) for tracks in per_camera_tracks.values())
    print(f"\nPer-camera tracks: {total_per_cam}")
    for cam_id in cam_order:
        if cam_id in per_camera_tracks:
            print(f"  {cam_id}: {len(per_camera_tracks[cam_id])} tracks")
    
    # ========================================
    # PHASE 2: Cross-camera association with 3D
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: Cross-camera 3D association")
    print("=" * 70)
    
    global_tracks: Dict[int, GlobalTrack] = {}
    next_global_id = 1
    
    # For each frame, find which per-camera tracks are active
    # Project their positions to 3D and associate
    
    for frame_idx in tqdm(range(total_frames), desc="3D association"):
        # Collect active per-camera tracks for this frame
        frame_observations = []  # List of (cam_id, track_id, bbox, class_name, pos_3d)
        
        for cam_id in cam_order:
            if cam_id not in per_camera_tracks:
                continue
            for tid, track in per_camera_tracks[cam_id].items():
                if frame_idx in track.detections:
                    bbox, conf = track.detections[frame_idx]
                    # Project bottom center to 3D
                    bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                    pos_3d = projectors[cam_id].project_to_ground(bottom_center)
                    if pos_3d is not None and abs(pos_3d[0]) < 40 and abs(pos_3d[1]) < 40:
                        frame_observations.append((cam_id, tid, bbox, track.class_name, pos_3d))
        
        if not frame_observations:
            # Predict existing tracks
            for gt in global_tracks.values():
                if gt.frames_since_seen < max_age:
                    gt.kalman.predict()
                    gt.frames_since_seen += 1
            continue
        
        # Try to triangulate when same object visible in multiple cameras
        # Group observations by 3D proximity
        used = set()
        observation_groups = []
        
        for i, (cam_id1, tid1, bbox1, cls1, pos1) in enumerate(frame_observations):
            if i in used:
                continue
            
            group = [(cam_id1, tid1, bbox1, cls1, pos1)]
            used.add(i)
            
            for j, (cam_id2, tid2, bbox2, cls2, pos2) in enumerate(frame_observations):
                if j in used or cam_id1 == cam_id2:
                    continue
                
                # Check if same class category
                if cls1 != cls2:
                    if not (cls1 in ['car', 'truck'] and cls2 in ['car', 'truck']):
                        continue
                
                # Check 3D proximity - be more generous for same-frame association
                dist = np.linalg.norm(pos1[:2] - pos2[:2])
                if dist < 4.0:  # Within 4 meters (projection noise)
                    group.append((cam_id2, tid2, bbox2, cls2, pos2))
                    used.add(j)
            
            observation_groups.append(group)
        
        # For each group, compute best 3D position
        group_positions = []
        for group in observation_groups:
            if len(group) >= 2:
                # Triangulate from multiple views
                obs_dict = {cam_id: np.array([(bbox[0]+bbox[2])/2, bbox[3]]) 
                            for cam_id, _, bbox, _, _ in group}
                pos_3d = triangulate_point(projectors, obs_dict)
                if pos_3d is None:
                    # Fallback to average
                    pos_3d = np.mean([p[:3] for _, _, _, _, p in group], axis=0)
            else:
                pos_3d = group[0][4][:3]
            
            # Determine class (majority vote)
            class_votes = defaultdict(int)
            for _, _, _, cls, _ in group:
                class_votes[cls] += 1
            best_class = max(class_votes.keys(), key=lambda c: class_votes[c])
            
            group_positions.append((group, pos_3d, best_class))
        
        # Match groups to existing global tracks
        active_global = {
            gid: gt for gid, gt in global_tracks.items()
            if gt.frames_since_seen < max_age
        }
        
        if active_global and group_positions:
            global_ids = list(active_global.keys())
            cost_matrix = np.zeros((len(group_positions), len(global_ids)))
            
            for gi, (group, pos_3d, cls) in enumerate(group_positions):
                for ti, gid in enumerate(global_ids):
                    gt = active_global[gid]
                    pred_pos = gt.kalman.position
                    dist = np.linalg.norm(pos_3d[:2] - pred_pos)
                    
                    # Class penalty
                    if gt.class_name != cls:
                        if not (gt.class_name in ['car', 'truck'] and cls in ['car', 'truck']):
                            dist += 10
                    
                    cost_matrix[gi, ti] = dist
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_groups = set()
            matched_globals = set()
            
            for gi, ti in zip(row_ind, col_ind):
                if cost_matrix[gi, ti] < 5.0:  # Max 5m distance
                    group, pos_3d, cls = group_positions[gi]
                    gid = global_ids[ti]
                    gt = global_tracks[gid]
                    
                    # Update Kalman
                    gt.kalman.update(pos_3d[:2])
                    gt.last_seen = frame_idx
                    gt.frames_since_seen = 0
                    
                    # Store frame data
                    gt.frames[frame_idx] = {
                        'position': gt.kalman.position.tolist() + [0.0],
                        'cameras': {cam_id: bbox.tolist() for cam_id, _, bbox, _, _ in group}
                    }
                    
                    # Update camera track associations
                    for cam_id, tid, _, _, _ in group:
                        gt.camera_tracks[cam_id] = tid
                    
                    matched_groups.add(gi)
                    matched_globals.add(ti)
        else:
            matched_groups = set()
        
        # Create new global tracks for unmatched groups
        for gi, (group, pos_3d, cls) in enumerate(group_positions):
            if gi not in matched_groups:
                gt = GlobalTrack(
                    track_id=next_global_id,
                    class_name=cls
                )
                next_global_id += 1
                
                gt.kalman.state[:2] = pos_3d[:2]
                gt.last_seen = frame_idx
                gt.frames_since_seen = 0
                
                gt.frames[frame_idx] = {
                    'position': pos_3d[:2].tolist() + [0.0],
                    'cameras': {cam_id: bbox.tolist() for cam_id, _, bbox, _, _ in group}
                }
                
                for cam_id, tid, _, _, _ in group:
                    gt.camera_tracks[cam_id] = tid
                
                global_tracks[gt.track_id] = gt
        
        # Predict unmatched global tracks
        for gid in active_global:
            if gid not in matched_globals:
                global_tracks[gid].kalman.predict()
                global_tracks[gid].frames_since_seen += 1
    
    # Filter short tracks
    global_tracks = {
        gid: gt for gid, gt in global_tracks.items()
        if len(gt.frames) >= min_track_length
    }
    
    print(f"\nGlobal tracks before merging: {len(global_tracks)}")
    
    # ========================================
    # PHASE 3: Merge fragmented tracks
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 3: Merging fragmented tracks")
    print("=" * 70)
    
    # Find tracks that are likely the same object
    # (similar class, one ends shortly before another starts, positions align)
    
    track_list = list(global_tracks.values())
    parent = list(range(len(track_list)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, t1 in enumerate(track_list):
        frames1 = sorted(t1.frames.keys())
        if not frames1:
            continue
        end1 = max(frames1)
        pos1_end = np.array(t1.frames[end1]['position'][:2])
        
        for j, t2 in enumerate(track_list):
            if i >= j:
                continue
            
            # Check class compatibility
            if t1.class_name != t2.class_name:
                if not (t1.class_name in ['car', 'truck'] and t2.class_name in ['car', 'truck']):
                    continue
            
            frames2 = sorted(t2.frames.keys())
            if not frames2:
                continue
            start2 = min(frames2)
            
            # Check temporal gap
            gap = start2 - end1
            if 1 <= gap <= 15:  # Gap of 1-15 frames only
                pos2_start = np.array(t2.frames[start2]['position'][:2])
                dist = np.linalg.norm(pos1_end - pos2_start)
                
                # Strict distance: 1.5m base + 0.3m per frame (max ~6m for 15 frame gap)
                max_dist = 1.5 + 0.3 * gap
                
                if dist < max_dist:
                    print(f"  Merging T{t1.track_id} (ends f{end1}) with T{t2.track_id} (starts f{start2}), gap={gap}f, dist={dist:.1f}m")
                    union(i, j)
    
    # Group merged tracks
    merged_groups = defaultdict(list)
    for i, t in enumerate(track_list):
        merged_groups[find(i)].append(t)
    
    # Create merged tracks
    final_tracks: Dict[int, GlobalTrack] = {}
    next_final_id = 1
    
    for group in merged_groups.values():
        # Sort by first frame
        group.sort(key=lambda t: min(t.frames.keys()) if t.frames else 0)
        
        # Merge all frames
        merged = GlobalTrack(
            track_id=next_final_id,
            class_name=group[0].class_name
        )
        next_final_id += 1
        
        for t in group:
            for f, data in t.frames.items():
                if f not in merged.frames:
                    merged.frames[f] = data
                else:
                    # Merge camera data
                    merged.frames[f]['cameras'].update(data['cameras'])
            merged.camera_tracks.update(t.camera_tracks)
        
        # Re-smooth positions with Kalman
        sorted_frames = sorted(merged.frames.keys())
        if sorted_frames:
            merged.kalman.state[:2] = np.array(merged.frames[sorted_frames[0]]['position'][:2])
            
            for f in sorted_frames:
                pos = np.array(merged.frames[f]['position'][:2])
                merged.kalman.update(pos)
                merged.frames[f]['position'] = merged.kalman.position.tolist() + [0.0]
        
        final_tracks[merged.track_id] = merged
    
    print(f"\nTracks after temporal merging: {len(final_tracks)}")
    
    # ========================================
    # Additional: Merge stationary objects with same position
    # ========================================
    print("\n  Merging stationary objects with same position...")
    
    # Compute average position and variance for each track
    track_stats = {}
    for tid, gt in final_tracks.items():
        if gt.frames:
            positions = np.array([gt.frames[f]['position'][:2] for f in gt.frames])
            avg_pos = np.mean(positions, axis=0)
            variance = np.var(positions, axis=0).sum()
            track_stats[tid] = (avg_pos, variance)
    
    # Find tracks with very similar average positions (same stationary object)
    # Only merge if BOTH tracks are stationary (low variance)
    parent2 = {tid: tid for tid in final_tracks}
    
    def find2(x):
        if parent2[x] != x:
            parent2[x] = find2(parent2[x])
        return parent2[x]
    
    def union2(x, y):
        px, py = find2(x), find2(y)
        if px != py:
            parent2[px] = py
    
    tids = list(final_tracks.keys())
    for i, tid1 in enumerate(tids):
        for tid2 in tids[i+1:]:
            gt1, gt2 = final_tracks[tid1], final_tracks[tid2]
            
            # Must be same class
            if gt1.class_name != gt2.class_name:
                if not (gt1.class_name in ['car', 'truck'] and gt2.class_name in ['car', 'truck']):
                    continue
            
            if tid1 not in track_stats or tid2 not in track_stats:
                continue
            
            avg1, var1 = track_stats[tid1]
            avg2, var2 = track_stats[tid2]
            
            # Only merge if BOTH are stationary (variance < 3m^2)
            if var1 > 3.0 or var2 > 3.0:
                continue
            
            # Check average position - strict 1.5m threshold
            dist = np.linalg.norm(avg1 - avg2)
            if dist < 1.5:
                print(f"    Merging stationary T{tid1} with T{tid2}, avg_dist={dist:.1f}m")
                union2(tid1, tid2)
    
    # Group and merge
    merged_groups2 = defaultdict(list)
    for tid in tids:
        merged_groups2[find2(tid)].append(final_tracks[tid])
    
    final_tracks2 = {}
    next_id = 1
    for group in merged_groups2.values():
        merged = GlobalTrack(track_id=next_id, class_name=group[0].class_name)
        next_id += 1
        
        for gt in group:
            for f, data in gt.frames.items():
                if f not in merged.frames:
                    merged.frames[f] = data
                else:
                    merged.frames[f]['cameras'].update(data['cameras'])
            merged.camera_tracks.update(gt.camera_tracks)
        
        final_tracks2[merged.track_id] = merged
    
    final_tracks = final_tracks2
    print(f"\nFinal tracks after all merging: {len(final_tracks)}")
    
    # ========================================
    # PHASE 4: Smooth trajectories and interpolate gaps
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 4: Smoothing and interpolating")
    print("=" * 70)
    
    for gt in final_tracks.values():
        frames = sorted(gt.frames.keys())
        if len(frames) < 5:
            continue
        
        # First, apply median filter to remove outliers
        positions = np.array([gt.frames[f]['position'][:2] for f in frames])
        
        # Detect and fix outliers using local median
        window = 5
        for i in range(len(frames)):
            start = max(0, i - window // 2)
            end = min(len(frames), i + window // 2 + 1)
            
            local_median = np.median(positions[start:end], axis=0)
            dist_to_median = np.linalg.norm(positions[i] - local_median)
            
            # If point is far from local median, replace with median
            if dist_to_median > 2.0:  # More than 2m from local median
                positions[i] = local_median
        
        # Apply moving average smoothing
        smoothed = np.copy(positions)
        for i in range(len(frames)):
            start = max(0, i - 2)
            end = min(len(frames), i + 3)
            smoothed[i] = np.mean(positions[start:end], axis=0)
        
        # Update positions
        for i, f in enumerate(frames):
            gt.frames[f]['position'] = smoothed[i].tolist() + [0.0]
        
        # Interpolate gaps
        interpolated = 0
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            gap = f2 - f1
            
            if 2 <= gap <= 10:
                pos1 = smoothed[i]
                pos2 = smoothed[i + 1]
                
                for f in range(f1 + 1, f2):
                    t = (f - f1) / gap
                    pos = pos1 * (1 - t) + pos2 * t
                    gt.frames[f] = {
                        'position': pos.tolist() + [0.0],
                        'cameras': {},
                        'interpolated': True
                    }
                    interpolated += 1
    
    # ========================================
    # Summary and output
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    class_counts = defaultdict(int)
    for gt in final_tracks.values():
        class_counts[gt.class_name] += 1
    
    print(f"\nTotal tracks: {len(final_tracks)}")
    print("\nBy class:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")
    
    # Analyze movement
    stationary = 0
    moving = 0
    for gt in final_tracks.values():
        frames = sorted(gt.frames.keys())
        if len(frames) < 2:
            continue
        positions = [np.array(gt.frames[f]['position'][:2]) for f in frames]
        total_movement = sum(np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions)))
        if total_movement < 3:
            stationary += 1
        else:
            moving += 1
    
    print(f"\nStationary (<3m movement): {stationary}")
    print(f"Moving (>=3m movement): {moving}")
    
    # ========================================
    # Generate output files
    # ========================================
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories
    trajectory_data = {
        'fps': fps,
        'max_frame': total_frames - 1,
        'num_tracks': len(final_tracks),
        'trajectories': []
    }
    
    for gt in final_tracks.values():
        frames = sorted(gt.frames.keys())
        if len(frames) < 3:
            continue
        
        positions = [np.array(gt.frames[f]['position'][:2]) for f in frames]
        total_movement = sum(np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions)))
        
        traj_frames = []
        for f in frames:
            pos = gt.frames[f]['position']
            # Compute heading
            idx = frames.index(f)
            if idx < len(frames) - 1:
                next_pos = gt.frames[frames[idx + 1]]['position']
                delta = np.array(next_pos[:2]) - np.array(pos[:2])
                if np.linalg.norm(delta) > 0.1:
                    heading = float(np.arctan2(delta[1], delta[0]))
                else:
                    heading = 0
            else:
                heading = 0
            
            traj_frames.append({
                'frame_idx': f,
                'position_3d': [pos[0], pos[1], pos[2]],
                'heading': heading
            })
        
        trajectory_data['trajectories'].append({
            'track_id': gt.track_id,
            'class_name': gt.class_name,
            'total_movement_m': float(total_movement),
            'frames': traj_frames
        })
    
    traj_path = output_dir / "multicam_trajectories.json"
    with open(traj_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"\nSaved: {traj_path}")
    
    # ========================================
    # Generate verification video
    # ========================================
    print("\n" + "=" * 70)
    print("Generating verification video")
    print("=" * 70)
    
    colors = generate_colors(len(final_tracks) + 10)
    track_colors = {gt.track_id: colors[i % len(colors)] for i, gt in enumerate(final_tracks.values())}
    
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    grid_w, grid_h = 640, 360
    output_w, output_h = grid_w * 2, grid_h * 4
    
    output_path = output_dir / "cross_camera_verification.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = cv2.resize(frame, (grid_w, grid_h))
            else:
                frames[cam_id] = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Draw tracks
        for gt in final_tracks.values():
            if frame_idx not in gt.frames:
                continue
            
            color = track_colors[gt.track_id]
            frame_data = gt.frames[frame_idx]
            
            for cam_id, bbox in frame_data.get('cameras', {}).items():
                if cam_id not in frames:
                    continue
                
                frame = frames[cam_id]
                bbox = np.array(bbox)
                
                scale_x, scale_y = grid_w / width, grid_h / height
                bbox_s = [int(bbox[0]*scale_x), int(bbox[1]*scale_y),
                          int(bbox[2]*scale_x), int(bbox[3]*scale_y)]
                
                # Draw interpolated frames differently
                thickness = 1 if frame_data.get('interpolated') else 2
                cv2.rectangle(frame, (bbox_s[0], bbox_s[1]), (bbox_s[2], bbox_s[3]), color, thickness)
                
                label = f"T{gt.track_id} {gt.class_name}"
                cv2.putText(frame, label, (bbox_s[0], bbox_s[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Camera labels
        for cam_id in cam_order:
            if cam_id in frames:
                cv2.putText(frames[cam_id], cam_id, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Arrange grid
        grid = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        for i, cam_id in enumerate(cam_order):
            if cam_id in frames:
                row, col = i // 2, i % 2
                grid[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w] = frames[cam_id]
        
        cv2.putText(grid, f"Frame {frame_idx}/{total_frames}", (10, output_h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        out.write(grid)
    
    for cap in caps.values():
        cap.release()
    out.release()
    
    print(f"Saved: {output_path}")
    
    return final_tracks


def main():
    base_dir = Path(__file__).parent.parent
    run_robust_tracking(
        video_dir=base_dir / "StreetAware-sample",
        output_dir=base_dir / "outputs" / "pass2_dynamic",
        camera_params_path=base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json",
        iou_threshold=0.3,
        max_age=20,  # Reduced from 30
        min_track_length=20  # Increased from 15
    )


if __name__ == "__main__":
    main()
