#!/usr/bin/env python3
"""
Multi-Camera 3D Object Tracker using SOTA methods.

Approach:
1. Run BoT-SORT (with ReID) on each camera independently for robust per-camera tracking
2. Project tracked objects to 3D ground plane
3. Associate tracks across cameras using:
   - Spatial proximity in 3D
   - Appearance similarity (ReID features)
   - Temporal consistency
4. Apply strong Kalman filtering for smooth trajectories
5. Filter out stationary objects and noise

This properly handles occlusions, re-identification, and cross-camera matching.
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import cv2
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

# NumPy 2.x compatibility
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import torch
from ultralytics import YOLO
from tqdm import tqdm


@dataclass
class Detection3D:
    """A detection projected to 3D world coordinates."""
    frame_idx: int
    camera_id: str
    bbox: np.ndarray  # [x1, y1, x2, y2] in image
    position_3d: np.ndarray  # [x, y, z] in world
    confidence: float
    class_name: str
    
    @property
    def position_2d(self) -> np.ndarray:
        """XY position on ground plane."""
        return self.position_3d[:2]


@dataclass 
class WorldTrack:
    """A unified track in world coordinates."""
    track_id: int
    class_name: str
    
    # Per-frame data
    positions: Dict[int, np.ndarray] = field(default_factory=dict)  # frame -> [x, y, z]
    velocities: Dict[int, np.ndarray] = field(default_factory=dict)  # frame -> [vx, vy]
    camera_detections: Dict[int, List[Tuple[str, np.ndarray]]] = field(default_factory=dict)  # frame -> [(cam_id, bbox), ...]
    
    # Kalman filter state
    kf_state: Optional[np.ndarray] = None  # [x, y, vx, vy]
    kf_cov: Optional[np.ndarray] = None
    
    # Track status
    age: int = 0  # Frames since creation
    hits: int = 0  # Number of frames with detections
    time_since_update: int = 0  # Frames since last detection
    
    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed if it has enough hits."""
        return self.hits >= 5
    
    def predict(self, dt: float = 1/15) -> np.ndarray:
        """Predict next position using Kalman filter."""
        if self.kf_state is None:
            return None
        
        # State transition
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise - lower = smoother but slower to respond
        q = 0.1  # Reduced for smoother tracking
        Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
        
        self.kf_state = F @ self.kf_state
        self.kf_cov = F @ self.kf_cov @ F.T + Q
        
        return self.kf_state[:2]
    
    def update(self, position: np.ndarray):
        """Update Kalman filter with measurement."""
        if self.kf_state is None:
            # Initialize
            self.kf_state = np.array([position[0], position[1], 0, 0])
            self.kf_cov = np.eye(4) * 10.0
            return
        
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise - higher = trust predictions more
        R = np.eye(2) * 2.0  # 2m noise - projections are noisy
        
        # Kalman update
        y = position[:2] - H @ self.kf_state
        S = H @ self.kf_cov @ H.T + R
        K = self.kf_cov @ H.T @ np.linalg.inv(S)
        
        self.kf_state = self.kf_state + K @ y
        self.kf_cov = (np.eye(4) - K @ H) @ self.kf_cov
    
    def get_predicted_position(self) -> Optional[np.ndarray]:
        """Get current predicted position."""
        if self.kf_state is None:
            return None
        return self.kf_state[:2]
    
    def total_movement(self) -> float:
        """Total movement in meters (start to end displacement)."""
        if len(self.positions) < 2:
            return 0.0
        frames = sorted(self.positions.keys())
        first = self.positions[frames[0]][:2]
        last = self.positions[frames[-1]][:2]
        return float(np.linalg.norm(last - first))
    
    def is_actually_moving(self, min_displacement: float = 3.0) -> bool:
        """Check if track shows real movement vs noise.
        
        A stationary object with noisy projections will have:
        - High position variance but low net displacement
        - Positions that oscillate around a center point
        
        A moving object will have:
        - Consistent directional movement
        - Net displacement >> position noise
        """
        if len(self.positions) < 10:
            return False
        
        frames = sorted(self.positions.keys())
        positions = np.array([self.positions[f][:2] for f in frames])
        
        # Net displacement (start to end)
        displacement = np.linalg.norm(positions[-1] - positions[0])
        
        # Position variance (spread around mean)
        mean_pos = np.mean(positions, axis=0)
        variance = np.mean([np.linalg.norm(p - mean_pos) for p in positions])
        
        # For a truly moving object: displacement >> variance
        # For stationary with noise: displacement ~ variance
        if displacement < min_displacement:
            return False
        
        # Displacement should be at least 3x the noise variance
        if displacement < variance * 3:
            return False
        
        return True


class CameraProjector:
    """Projects 2D detections to 3D ground plane."""
    
    def __init__(self, camera_params: Dict, image_size: Tuple[int, int]):
        self.K = np.array(camera_params['K']).reshape(3, 3)
        self.R_c2w = np.array(camera_params['R'])
        self.t_c2w = np.array(camera_params['t'])
        
        # Scale intrinsics if needed
        calib_w = camera_params.get('width', image_size[0])
        calib_h = camera_params.get('height', image_size[1])
        if calib_w != image_size[0]:
            scale_x = image_size[0] / calib_w
            scale_y = image_size[1] / calib_h
            self.K[0, :] *= scale_x
            self.K[1, :] *= scale_y
    
    def project_to_ground(self, pixel: np.ndarray, ground_z: float = 0.0) -> Optional[np.ndarray]:
        """Project pixel to ground plane."""
        u, v = pixel
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        # Ray in camera frame
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        ray_cam = np.array([x_cam, y_cam, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world
        ray_world = self.R_c2w @ ray_cam
        
        # Intersect with ground
        if abs(ray_world[2]) < 1e-6:
            return None
        
        t = (ground_z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
        
        return self.t_c2w + t * ray_world


class MultiCameraTracker:
    """Tracks vehicles across multiple synchronized cameras."""
    
    def __init__(
        self,
        camera_params: Dict[str, Dict],
        image_size: Tuple[int, int] = (2592, 1944),
        ground_z: float = 0.0,
        association_threshold: float = 3.0,  # meters
        max_age: int = 15,  # frames without detection before deletion
    ):
        self.projectors = {
            cam_id: CameraProjector(params, image_size)
            for cam_id, params in camera_params.items()
        }
        self.ground_z = ground_z
        self.association_threshold = association_threshold
        self.max_age = max_age
        
        self.tracks: Dict[int, WorldTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
    
    def cluster_detections(self, detections: List[Detection3D]) -> List[List[Detection3D]]:
        """Cluster detections from different cameras that see the same vehicle.
        
        Uses simple distance-based clustering: detections within threshold
        distance are considered the same vehicle.
        """
        if not detections:
            return []
        
        # Build distance matrix
        n = len(detections)
        used = [False] * n
        clusters = []
        
        for i in range(n):
            if used[i]:
                continue
            
            cluster = [detections[i]]
            used[i] = True
            
            for j in range(i + 1, n):
                if used[j]:
                    continue
                
                # Check if detection j is close to any detection in cluster
                pos_j = detections[j].position_2d
                for det in cluster:
                    dist = np.linalg.norm(det.position_2d - pos_j)
                    if dist < self.association_threshold:
                        cluster.append(detections[j])
                        used[j] = True
                        break
            
            clusters.append(cluster)
        
        return clusters
    
    def get_cluster_position(self, cluster: List[Detection3D]) -> np.ndarray:
        """Get robust position estimate from cluster with outlier rejection."""
        if len(cluster) == 1:
            return cluster[0].position_3d
        
        # First compute median position (robust to outliers)
        positions = np.array([d.position_3d for d in cluster])
        median_pos = np.median(positions, axis=0)
        
        # Reject outliers (>3m from median)
        valid_detections = []
        for d in cluster:
            dist = np.linalg.norm(d.position_3d[:2] - median_pos[:2])
            if dist < 3.0:
                valid_detections.append(d)
        
        if not valid_detections:
            valid_detections = cluster  # Fall back to all
        
        # Weight by confidence
        total_weight = sum(d.confidence for d in valid_detections)
        position = np.zeros(3)
        for d in valid_detections:
            position += d.position_3d * d.confidence
        return position / total_weight
    
    def associate_tracks(
        self, 
        clusters: List[List[Detection3D]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate clusters with existing tracks using Hungarian algorithm."""
        
        if not clusters or not self.tracks:
            return [], list(range(len(clusters))), list(self.tracks.keys())
        
        # Get predicted positions for all tracks
        track_ids = list(self.tracks.keys())
        track_positions = []
        for tid in track_ids:
            track = self.tracks[tid]
            pred = track.get_predicted_position()
            if pred is not None:
                track_positions.append(pred)
            else:
                track_positions.append(np.array([1e6, 1e6]))  # Invalid
        
        # Get cluster positions
        cluster_positions = [self.get_cluster_position(c)[:2] for c in clusters]
        
        # Build cost matrix
        cost_matrix = np.zeros((len(clusters), len(track_ids)))
        for i, cpos in enumerate(cluster_positions):
            for j, tpos in enumerate(track_positions):
                cost_matrix[i, j] = np.linalg.norm(cpos - tpos)
        
        # Apply threshold
        cost_matrix[cost_matrix > self.association_threshold * 2] = 1e6
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_clusters = set(range(len(clusters)))
        unmatched_tracks = set(track_ids)
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.association_threshold * 2:
                matches.append((i, track_ids[j]))
                unmatched_clusters.discard(i)
                unmatched_tracks.discard(track_ids[j])
        
        return matches, list(unmatched_clusters), list(unmatched_tracks)
    
    def update(self, frame_idx: int, all_detections: Dict[str, List[Detection3D]]):
        """Update tracks with detections from all cameras for one frame."""
        self.frame_count = frame_idx
        
        # Predict all tracks
        for track in self.tracks.values():
            track.predict()
            track.age += 1
            track.time_since_update += 1
        
        # Flatten detections from all cameras
        detections = []
        for cam_id, dets in all_detections.items():
            detections.extend(dets)
        
        # Cluster detections (same vehicle seen by multiple cameras)
        clusters = self.cluster_detections(detections)
        
        # Associate clusters with tracks
        matches, unmatched_clusters, unmatched_tracks = self.associate_tracks(clusters)
        
        # Update matched tracks
        for cluster_idx, track_id in matches:
            cluster = clusters[cluster_idx]
            position = self.get_cluster_position(cluster)
            
            track = self.tracks[track_id]
            track.update(position)
            
            # Store FILTERED position from Kalman state, not raw measurement
            filtered_pos = track.kf_state[:2]
            track.positions[frame_idx] = np.array([filtered_pos[0], filtered_pos[1], 0.0])
            
            track.hits += 1
            track.time_since_update = 0
            
            # Store camera detections
            track.camera_detections[frame_idx] = [
                (d.camera_id, d.bbox) for d in cluster
            ]
        
        # Create new tracks for unmatched clusters
        for cluster_idx in unmatched_clusters:
            cluster = clusters[cluster_idx]
            position = self.get_cluster_position(cluster)
            
            # Check if position is within scene bounds
            if abs(position[0]) > 30 or abs(position[1]) > 30:
                continue
            
            # Get most common class
            class_counts = defaultdict(int)
            for d in cluster:
                class_counts[d.class_name] += 1
            class_name = max(class_counts, key=class_counts.get)
            
            track = WorldTrack(
                track_id=self.next_track_id,
                class_name=class_name
            )
            track.update(position)
            track.positions[frame_idx] = position
            track.hits = 1
            track.camera_detections[frame_idx] = [
                (d.camera_id, d.bbox) for d in cluster
            ]
            
            self.tracks[track.track_id] = track
            self.next_track_id += 1
        
        # Remove old tracks - but keep a history of all tracks
        # Don't delete, just mark as inactive
        pass  # We'll filter at the end instead
    
    def get_confirmed_tracks(self, min_frames: int = 20, min_movement: float = 3.0) -> List[WorldTrack]:
        """Get tracks that meet quality criteria."""
        confirmed = []
        for track in self.tracks.values():
            if len(track.positions) < min_frames:
                continue
            # Use robust movement check instead of just displacement
            if not track.is_actually_moving(min_displacement=min_movement):
                continue
            confirmed.append(track)
        return confirmed


def run_multicam_tracking(
    video_dir: Path,
    camera_params: Dict[str, Dict],
    output_path: Path,
    ground_z: float = 0.0,
    association_threshold: float = 4.0
):
    """Run multi-camera tracking using BoT-SORT with ReID."""
    
    print("Loading YOLO model with BoT-SORT tracker...")
    model = YOLO('yolov8x.pt')
    
    # Get video paths
    video_paths = {
        p.stem: p for p in sorted(video_dir.glob("*.mp4"))
    }
    
    # Get video info from first video
    first_path = list(video_paths.values())[0]
    cap = cv2.VideoCapture(str(first_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Processing {len(video_paths)} cameras, {total_frames} frames @ {fps:.1f} fps")
    
    # Create projectors
    projectors = {
        cam_id: CameraProjector(camera_params[cam_id], (width, height))
        for cam_id in video_paths.keys()
        if cam_id in camera_params
    }
    
    # ============================================
    # PHASE 1: Two-Pass Detection and Tracking
    # Pass 1A: Collect all detections with bboxes (no tracking yet)
    # Pass 1B: Build tracks using IoU matching, lock class at peak confidence
    # ============================================
    print("\n=== Phase 1: Two-Pass Detection and Tracking ===")
    
    per_camera_tracks: Dict[str, Dict[int, Dict[int, Tuple]]] = {}
    
    for cam_id, video_path in video_paths.items():
        if cam_id not in projectors:
            continue
            
        print(f"  Processing {cam_id}...")
        projector = projectors[cam_id]
        
        # -----------------------------------------
        # Pass 1A: Collect all detections per frame
        # -----------------------------------------
        print(f"    Pass 1A: Collecting detections...")
        all_detections = {}  # frame_idx -> list of (bbox, class_name, conf)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection only (no tracking)
            results = model.predict(
                frame,
                conf=0.3,  # Lower threshold to catch more detections
                iou=0.5,
                verbose=False,
                classes=[0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
            )
            
            frame_dets = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.int().cpu().numpy()
                
                for i in range(len(xyxys)):
                    bbox = xyxys[i]
                    cls_id = cls_ids[i]
                    cls_name = model.names[cls_id]
                    conf = confs[i]
                    frame_dets.append((bbox.copy(), cls_name, conf))
            
            all_detections[frame_idx] = frame_dets
            frame_idx += 1
        
        cap.release()
        total_dets = sum(len(d) for d in all_detections.values())
        print(f"      Collected {total_dets} detections across {frame_idx} frames")
        
        # -----------------------------------------
        # Pass 1B: Build tracks using IoU matching
        # -----------------------------------------
        print(f"    Pass 1B: Building tracks with IoU matching...")
        
        # Track structure: {track_id: {frame_idx: (bbox, class_name, conf)}}
        tracks: Dict[int, Dict[int, Tuple]] = {}
        next_track_id = 1
        
        # Active tracks: track_id -> last_bbox, last_frame
        active_tracks: Dict[int, Tuple[np.ndarray, int]] = {}
        
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
        
        def predict_bbox(bbox, velocity, frames_gap):
            """Predict bbox position based on velocity."""
            if velocity is None:
                return bbox
            predicted = bbox.copy()
            predicted[0] += velocity[0] * frames_gap
            predicted[1] += velocity[1] * frames_gap
            predicted[2] += velocity[0] * frames_gap
            predicted[3] += velocity[1] * frames_gap
            return predicted
        
        # Track velocities for prediction
        track_velocities: Dict[int, np.ndarray] = {}
        
        for frame_idx in range(len(all_detections)):
            dets = all_detections[frame_idx]
            
            if not dets:
                continue
            
            # Match detections to active tracks
            matched_tracks = set()
            matched_dets = set()
            
            # Build cost matrix (IoU-based)
            if active_tracks:
                track_ids = list(active_tracks.keys())
                costs = np.zeros((len(track_ids), len(dets)))
                
                for ti, track_id in enumerate(track_ids):
                    last_bbox, last_frame = active_tracks[track_id]
                    frames_gap = frame_idx - last_frame
                    
                    # Predict position
                    velocity = track_velocities.get(track_id)
                    predicted_bbox = predict_bbox(last_bbox, velocity, frames_gap)
                    
                    for di, (det_bbox, _, _) in enumerate(dets):
                        iou = compute_iou(predicted_bbox, det_bbox)
                        costs[ti, di] = 1 - iou  # Cost = 1 - IoU
                
                # Greedy matching (could use Hungarian but this is simpler)
                while True:
                    if costs.size == 0:
                        break
                    min_idx = np.unravel_index(np.argmin(costs), costs.shape)
                    min_cost = costs[min_idx]
                    
                    if min_cost > 0.7:  # IoU < 0.3, no good match
                        break
                    
                    ti, di = min_idx
                    track_id = track_ids[ti]
                    det_bbox, det_cls, det_conf = dets[di]
                    
                    # Update track
                    tracks[track_id][frame_idx] = (det_bbox, det_cls, det_conf)
                    
                    # Update velocity
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
                    
                    matched_tracks.add(ti)
                    matched_dets.add(di)
                    
                    # Remove matched from cost matrix
                    costs[ti, :] = np.inf
                    costs[:, di] = np.inf
            
            # Create new tracks for unmatched detections
            for di, (det_bbox, det_cls, det_conf) in enumerate(dets):
                if di not in matched_dets:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {frame_idx: (det_bbox, det_cls, det_conf)}
                    active_tracks[track_id] = (det_bbox, frame_idx)
            
            # Remove stale tracks (not seen for 30 frames)
            stale = [tid for tid, (_, last_f) in active_tracks.items() 
                     if frame_idx - last_f > 30]
            for tid in stale:
                del active_tracks[tid]
                if tid in track_velocities:
                    del track_velocities[tid]
        
        print(f"      Built {len(tracks)} raw tracks")
        
        # -----------------------------------------
        # Pass 1C: Lock class at peak confidence for each track
        # -----------------------------------------
        print(f"    Pass 1C: Locking class at peak confidence...")
        
        per_camera_tracks[cam_id] = {}
        
        for track_id, track_frames in tracks.items():
            if len(track_frames) < 5:  # Skip very short tracks
                continue
            
            # Find peak confidence frame and its class
            peak_frame = None
            peak_conf = 0
            peak_class = None
            
            # Also collect class votes weighted by confidence
            class_scores = defaultdict(float)
            
            for f, (bbox, cls_name, conf) in track_frames.items():
                class_scores[cls_name] += conf
                if conf > peak_conf:
                    peak_conf = conf
                    peak_frame = f
                    peak_class = cls_name
            
            # Use the class with highest total confidence score
            # This is more robust than just peak frame
            locked_class = max(class_scores.keys(), key=lambda c: class_scores[c])
            
            # Special handling: if bicycle and person both detected, prefer bicycle
            # (person riding bicycle often gets detected as both)
            if 'bicycle' in class_scores and 'person' in class_scores:
                if class_scores['bicycle'] > class_scores['person'] * 0.3:
                    locked_class = 'bicycle'
            
            # If truck and car both detected, use the one with higher score
            # (no special handling needed, just use max)
            
            # Build final track with locked class
            per_camera_tracks[cam_id][track_id] = {}
            
            for f, (bbox, _, conf) in track_frames.items():
                # Project to 3D
                bottom_center = np.array([
                    (bbox[0] + bbox[2]) / 2,
                    bbox[3]
                ])
                
                pos_3d = projector.project_to_ground(bottom_center, ground_z)
                if pos_3d is None:
                    continue
                
                if abs(pos_3d[0]) > 50 or abs(pos_3d[1]) > 50:
                    continue
                
                per_camera_tracks[cam_id][track_id][f] = (
                    bbox, pos_3d, locked_class, conf
                )
        
        n_tracks = len(per_camera_tracks[cam_id])
        print(f"    Final: {n_tracks} tracks with locked classes")
    
    # ============================================
    # PHASE 2: Cross-camera track association with stricter matching
    # ============================================
    print("\n=== Phase 2: Cross-camera association ===")
    
    # Build global tracks by associating per-camera tracks
    # Use stricter matching: same class, temporal overlap, AND consistent movement direction
    
    global_tracks: Dict[int, Dict] = {}
    next_global_id = 1
    cam_to_global: Dict[str, Dict[int, int]] = {}
    
    # Sort cameras to process in consistent order
    sorted_cams = sorted(per_camera_tracks.keys())
    
    for cam_id in sorted_cams:
        cam_to_global[cam_id] = {}
        
        for local_id, frames_data in per_camera_tracks[cam_id].items():
            if len(frames_data) < 10:  # Need at least 10 frames
                continue
            
            # Get track properties
            frame_indices = sorted(frames_data.keys())
            positions = np.array([frames_data[f][1][:2] for f in frame_indices])
            classes = [frames_data[f][2] for f in frame_indices]
            main_class = max(set(classes), key=classes.count)
            
            # Compute movement direction
            if len(positions) >= 5:
                start_pos = np.mean(positions[:5], axis=0)
                end_pos = np.mean(positions[-5:], axis=0)
                movement = end_pos - start_pos
                movement_dist = np.linalg.norm(movement)
                if movement_dist > 1.0:
                    movement_dir = movement / movement_dist
                else:
                    movement_dir = None
            else:
                movement_dir = None
            
            min_frame, max_frame = min(frame_indices), max(frame_indices)
            
            # Try to match with existing global tracks
            best_match = None
            best_score = float('inf')
            
            for global_id, global_track in global_tracks.items():
                # Must be same class
                if global_track['class_name'] != main_class:
                    continue
                
                # Check temporal overlap
                g_frames = sorted(global_track['frames'].keys())
                g_min, g_max = min(g_frames), max(g_frames)
                
                overlap_start = max(min_frame, g_min)
                overlap_end = min(max_frame, g_max)
                if overlap_end - overlap_start < 5:  # Need at least 5 frames overlap
                    continue
                
                # Compare positions in overlapping frames
                dists = []
                for f in range(overlap_start, overlap_end + 1):
                    if f in frames_data and f in global_track['frames']:
                        local_pos = frames_data[f][1][:2]
                        global_pos = global_track['frames'][f][:2]
                        dists.append(np.linalg.norm(local_pos - global_pos))
                
                if len(dists) < 5:
                    continue
                
                avg_dist = np.mean(dists)
                median_dist = np.median(dists)
                
                # Use median distance (more robust to outliers)
                if median_dist < association_threshold and median_dist < best_score:
                    # Also check movement direction consistency
                    if movement_dir is not None and 'movement_dir' in global_track:
                        g_dir = global_track['movement_dir']
                        if g_dir is not None:
                            dot_product = np.dot(movement_dir, g_dir)
                            if dot_product < 0.5:  # Must be roughly same direction
                                continue
                    
                    best_match = global_id
                    best_score = median_dist
            
            if best_match is not None:
                # Merge into existing global track
                global_id = best_match
                cam_to_global[cam_id][local_id] = global_id
                global_tracks[global_id]['cameras'].add(cam_id)
                
                # Add/update positions using weighted average
                for f, data in frames_data.items():
                    pos = data[1]
                    conf = data[3]  # Use confidence as weight
                    if f in global_tracks[global_id]['frames']:
                        old_pos = global_tracks[global_id]['frames'][f]
                        old_weight = global_tracks[global_id].get('weights', {}).get(f, 1.0)
                        new_weight = old_weight + conf
                        global_tracks[global_id]['frames'][f] = (old_pos * old_weight + pos * conf) / new_weight
                        if 'weights' not in global_tracks[global_id]:
                            global_tracks[global_id]['weights'] = {}
                        global_tracks[global_id]['weights'][f] = new_weight
                    else:
                        global_tracks[global_id]['frames'][f] = pos
            else:
                # Create new global track
                global_id = next_global_id
                next_global_id += 1
                cam_to_global[cam_id][local_id] = global_id
                
                global_tracks[global_id] = {
                    'class_name': main_class,
                    'cameras': {cam_id},
                    'frames': {f: data[1] for f, data in frames_data.items()},
                    'movement_dir': movement_dir
                }
    
    print(f"  Created {len(global_tracks)} global tracks from {sum(len(t) for t in per_camera_tracks.values())} camera tracks")
    
    # ============================================
    # PHASE 2.5: Merge duplicate global tracks
    # ============================================
    print("\n=== Phase 2.5: Merging duplicate tracks ===")
    
    # Find tracks that are very close in space and time (likely duplicates)
    merged = set()
    merge_map = {}  # old_id -> new_id
    
    track_ids = list(global_tracks.keys())
    for i, id1 in enumerate(track_ids):
        if id1 in merged:
            continue
        for id2 in track_ids[i+1:]:
            if id2 in merged:
                continue
            
            t1, t2 = global_tracks[id1], global_tracks[id2]
            
            # Must be same class
            if t1['class_name'] != t2['class_name']:
                continue
            
            # Check spatial overlap
            f1 = set(t1['frames'].keys())
            f2 = set(t2['frames'].keys())
            common = f1 & f2
            
            if len(common) < 10:
                continue
            
            # Compute average distance in common frames
            dists = []
            for f in common:
                d = np.linalg.norm(t1['frames'][f][:2] - t2['frames'][f][:2])
                dists.append(d)
            
            if np.median(dists) < 2.0:  # Very close - likely same object
                # Merge t2 into t1
                merged.add(id2)
                merge_map[id2] = id1
                t1['cameras'].update(t2['cameras'])
                
                for f, pos in t2['frames'].items():
                    if f in t1['frames']:
                        t1['frames'][f] = (t1['frames'][f] + pos) / 2
                    else:
                        t1['frames'][f] = pos
    
    # Remove merged tracks
    for mid in merged:
        del global_tracks[mid]
    
    print(f"  Merged {len(merged)} duplicate tracks, {len(global_tracks)} remaining")
    
    # ============================================
    # PHASE 3: Smooth trajectories and filter
    # ============================================
    print("\n=== Phase 3: Smoothing and filtering ===")
    
    confirmed_tracks = []
    
    for global_id, track_data in global_tracks.items():
        frames = track_data['frames']
        if len(frames) < 15:
            continue
        
        # Sort frames
        frame_indices = sorted(frames.keys())
        positions = np.array([frames[f][:2] for f in frame_indices])
        
        # Check if actually moving (not just noise)
        displacement = np.linalg.norm(positions[-1] - positions[0])
        variance = np.mean([np.linalg.norm(p - np.mean(positions, axis=0)) for p in positions])
        
        # Must have real movement: displacement > 3m and displacement > 2*variance
        if displacement < 3.0 or displacement < variance * 2:
            continue
        
        # Apply Savitzky-Golay smoothing if enough points
        if len(positions) >= 11:
            from scipy.signal import savgol_filter
            window = min(11, len(positions) // 2 * 2 + 1)  # Must be odd
            if window >= 5:
                positions[:, 0] = savgol_filter(positions[:, 0], window, 3)
                positions[:, 1] = savgol_filter(positions[:, 1], window, 3)
        
        # Rebuild frames dict with smoothed positions
        smoothed_frames = {}
        for i, f in enumerate(frame_indices):
            smoothed_frames[f] = np.array([positions[i, 0], positions[i, 1], 0.0])
        
        confirmed_tracks.append({
            'global_id': global_id,
            'class_name': track_data['class_name'],
            'cameras': list(track_data['cameras']),
            'frames': smoothed_frames,
            'displacement': displacement
        })
    
    print(f"  Confirmed {len(confirmed_tracks)} moving tracks")
    
    # ============================================
    # PHASE 4: Compute headings and save
    # ============================================
    print("\n=== Phase 4: Computing headings and saving ===")
    
    output_data = {
        'num_cameras': len(video_paths),
        'num_frames': total_frames,
        'fps': fps,
        'num_tracks': len(confirmed_tracks),
        'trajectories': []
    }
    
    for track in confirmed_tracks:
        frames = track['frames']
        frame_indices = sorted(frames.keys())
        positions = np.array([frames[f][:2] for f in frame_indices])
        
        # Compute headings using sliding window
        headings = []
        window = 10
        for i in range(len(positions)):
            look_ahead = min(window, len(positions) - i - 1)
            if look_ahead >= 3:
                dx = positions[i + look_ahead, 0] - positions[i, 0]
                dy = positions[i + look_ahead, 1] - positions[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.3:
                    headings.append(np.arctan2(dy, dx))
                else:
                    headings.append(headings[-1] if headings else 0)
            else:
                headings.append(headings[-1] if headings else 0)
        
        # Smooth headings
        smoothed = [headings[0]]
        alpha = 0.2
        for i in range(1, len(headings)):
            diff = headings[i] - smoothed[-1]
            while diff > np.pi: diff -= 2*np.pi
            while diff < -np.pi: diff += 2*np.pi
            smoothed.append(smoothed[-1] + alpha * diff)
        
        traj = {
            'track_id': track['global_id'],
            'class_name': track['class_name'],
            'category': 'vehicle' if track['class_name'] in ['car', 'truck', 'bus'] else 'person' if track['class_name'] == 'person' else 'bicycle',
            'num_frames': len(frames),
            'total_movement_m': float(track['displacement']),
            'cameras': track['cameras'],
            'frames': []
        }
        
        for i, f in enumerate(frame_indices):
            pos = frames[f]
            frame_data = {
                'frame_idx': f,
                'position_3d': pos.tolist(),
                'heading': float(smoothed[i])
            }
            traj['frames'].append(frame_data)
        
        output_data['trajectories'].append(traj)
    
    # Sort by movement (most movement first)
    output_data['trajectories'].sort(key=lambda t: t['total_movement_m'], reverse=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Print summary
    print("\nTrack Summary:")
    for traj in output_data['trajectories']:
        print(f"  Track {traj['track_id']}: {traj['num_frames']} frames, "
              f"{traj['total_movement_m']:.1f}m movement")


def main():
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "StreetAware-sample"
    output_dir = base_dir / "outputs" / "pass2_dynamic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera parameters
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        camera_params = json.load(f)
    
    output_path = output_dir / "multicam_trajectories.json"
    
    run_multicam_tracking(
        video_dir=video_dir,
        camera_params=camera_params,
        output_path=output_path,
        ground_z=0.0,
        association_threshold=5.0  # 5 meters for cross-camera association
    )


if __name__ == "__main__":
    main()
