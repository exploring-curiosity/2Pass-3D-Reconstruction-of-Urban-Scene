#!/usr/bin/env python3
"""
Unified 3D Cross-Camera Tracker

This implements a robust multi-camera tracking system:
1. Project all detections from all 8 cameras to a common 3D ground plane
2. For each frame, cluster detections by 3D position to find same objects
3. Track objects over time using Kalman filter with velocity prediction
4. Handle occlusion by predicting position when object is temporarily lost
5. Lock class label based on highest confidence detection

Key design decisions:
- All cameras share a common world coordinate system (from calibration)
- Objects are tracked in 3D world space, not per-camera 2D
- Kalman filter predicts position during occlusion (up to N frames)
- Class is locked after seeing object with high confidence
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


@dataclass
class Detection3D:
    """A single detection projected to 3D."""
    cam_id: str
    frame_idx: int
    bbox: np.ndarray  # [x1, y1, x2, y2] in image
    pos_3d: np.ndarray  # [x, y, z] in world
    cls_name: str
    confidence: float


@dataclass 
class GlobalTrack:
    """A track in 3D world coordinates, potentially seen by multiple cameras."""
    track_id: int
    cls_name: str
    cls_confidence: float  # Confidence of locked class
    
    # Kalman filter state: [x, y, vx, vy]
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 10)
    
    # Track history
    frames_seen: Dict[int, Dict[str, Tuple]] = field(default_factory=dict)
    # frames_seen[frame_idx][cam_id] = (bbox, pos_3d, conf)
    
    last_seen_frame: int = -1
    frames_since_seen: int = 0
    total_detections: int = 0
    
    # Class voting - use regular dict, initialize in update
    class_votes: Dict[str, float] = field(default_factory=dict)
    
    def predict(self, dt: float = 1.0):
        """Predict next state using constant velocity model."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Q = np.eye(4) * 0.5  # Process noise
        Q[0, 0] = Q[1, 1] = 0.1  # Position noise
        Q[2, 2] = Q[3, 3] = 0.5  # Velocity noise
        
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        self.frames_since_seen += 1
    
    def update(self, pos_3d: np.ndarray):
        """Update state with new observation."""
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * 1.0  # Measurement noise (1 meter std)
        
        z = pos_3d[:2]  # Observation (x, y)
        y = z - H @ self.state  # Innovation
        S = H @ self.covariance @ H.T + R  # Innovation covariance
        K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance
        self.frames_since_seen = 0
    
    @property
    def position(self) -> np.ndarray:
        """Current estimated position."""
        return self.state[:2]
    
    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity."""
        return self.state[2:4]


class CameraProjector:
    """Projects 2D image points to 3D ground plane."""
    
    def __init__(self, camera_params: dict):
        self.K = np.array(camera_params['K']).reshape(3, 3)
        pose_c2w = np.array(camera_params['pose_c2w'])
        self.R_c2w = pose_c2w[:3, :3]
        self.cam_pos = pose_c2w[:3, 3]
        
    def project_to_ground(self, pixel: np.ndarray, ground_z: float = 0.0) -> Optional[np.ndarray]:
        """Project a 2D pixel to the ground plane."""
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        ray_cam = np.linalg.inv(self.K) @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = self.R_c2w @ ray_cam
        
        if abs(ray_world[2]) < 1e-6:
            return None
            
        s = (ground_z - self.cam_pos[2]) / ray_world[2]
        
        if s < 0:
            return None
            
        point_3d = self.cam_pos + s * ray_world
        return point_3d


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good distribution
        sat, val = 0.85, 0.9
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def run_unified_3d_tracking(
    video_dir: Path,
    output_dir: Path,
    camera_params_path: Path,
    ground_z: float = 0.0,
    association_threshold: float = 3.0,  # meters - max distance to associate
    max_frames_lost: int = 30,  # frames to keep predicting during occlusion
    min_track_length: int = 5,  # minimum detections to confirm track
):
    """
    Run unified 3D tracking across all cameras.
    """
    print("=" * 70)
    print("Unified 3D Cross-Camera Tracker")
    print("=" * 70)
    
    # Load camera params
    with open(camera_params_path) as f:
        camera_params = json.load(f)
    
    # Load YOLO
    print("\nLoading YOLO model...")
    model = YOLO('yolov8x.pt')
    
    # Camera order
    cam_order = ['s1-left', 's1-right', 's2-left', 's2-right', 
                 's3-left', 's3-right', 's4-left', 's4-right']
    
    # Get video paths
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
    projectors = {cam_id: CameraProjector(camera_params[cam_id]) for cam_id in video_paths}
    
    # ========================================
    # PHASE 1: Collect all detections
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: Collecting detections from all cameras")
    print("=" * 70)
    
    # all_detections[frame_idx] = list of Detection3D
    all_detections: Dict[int, List[Detection3D]] = defaultdict(list)
    
    # Open all video captures
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        for cam_id in cam_order:
            if cam_id not in caps:
                continue
                
            cap = caps[cam_id]
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Run detection
            results = model.predict(
                frame, conf=0.3, iou=0.5, verbose=False,
                classes=[0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
            )
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
                
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()
            
            projector = projectors[cam_id]
            
            for i in range(len(xyxys)):
                bbox = xyxys[i]
                cls_name = model.names[cls_ids[i]]
                conf = confs[i]
                
                # Project bottom center to 3D
                bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                pos_3d = projector.project_to_ground(bottom_center, ground_z)
                
                # Filter out invalid projections
                if pos_3d is None:
                    continue
                if abs(pos_3d[0]) > 50 or abs(pos_3d[1]) > 50:
                    continue
                    
                det = Detection3D(
                    cam_id=cam_id,
                    frame_idx=frame_idx,
                    bbox=bbox,
                    pos_3d=pos_3d,
                    cls_name=cls_name,
                    confidence=conf
                )
                all_detections[frame_idx].append(det)
    
    # Close captures
    for cap in caps.values():
        cap.release()
    
    total_dets = sum(len(dets) for dets in all_detections.values())
    print(f"\nTotal detections: {total_dets}")
    
    # ========================================
    # PHASE 2: Track in 3D with Kalman filter
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: 3D Tracking with Kalman Filter")
    print("=" * 70)
    
    global_tracks: Dict[int, GlobalTrack] = {}
    next_track_id = 1
    
    for frame_idx in tqdm(range(total_frames), desc="Tracking"):
        detections = all_detections[frame_idx]
        
        # Predict all active tracks
        for track in global_tracks.values():
            if track.frames_since_seen < max_frames_lost:
                track.predict(dt=1.0)
        
        if not detections:
            continue
        
        # Get active tracks (not lost for too long)
        active_tracks = {
            tid: t for tid, t in global_tracks.items() 
            if t.frames_since_seen < max_frames_lost
        }
        
        # Build cost matrix for Hungarian algorithm
        # Rows: detections, Cols: tracks
        if active_tracks:
            track_ids = list(active_tracks.keys())
            cost_matrix = np.zeros((len(detections), len(track_ids)))
            
            for di, det in enumerate(detections):
                for ti, tid in enumerate(track_ids):
                    track = active_tracks[tid]
                    
                    # Distance cost
                    dist = np.linalg.norm(det.pos_3d[:2] - track.position)
                    
                    # Class mismatch penalty
                    if track.cls_name != det.cls_name:
                        # Allow some class confusion (truck/car, person/bicycle)
                        if not ((track.cls_name in ['car', 'truck'] and det.cls_name in ['car', 'truck']) or
                                (track.cls_name in ['person', 'bicycle'] and det.cls_name in ['person', 'bicycle'])):
                            dist += 10  # Heavy penalty for incompatible classes
                    
                    cost_matrix[di, ti] = dist
            
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_dets = set()
            matched_tracks = set()
            
            for di, ti in zip(row_ind, col_ind):
                if cost_matrix[di, ti] < association_threshold + 5:  # Allow some slack
                    det = detections[di]
                    tid = track_ids[ti]
                    track = active_tracks[tid]
                    
                    # Update track
                    track.update(det.pos_3d)
                    track.last_seen_frame = frame_idx
                    track.total_detections += 1
                    
                    # Store detection info
                    if frame_idx not in track.frames_seen:
                        track.frames_seen[frame_idx] = {}
                    track.frames_seen[frame_idx][det.cam_id] = (det.bbox, det.pos_3d, det.confidence)
                    
                    # Update class votes
                    if det.cls_name not in track.class_votes:
                        track.class_votes[det.cls_name] = 0.0
                    track.class_votes[det.cls_name] += det.confidence
                    
                    # Update locked class if this is higher confidence
                    best_class = max(track.class_votes.keys(), key=lambda c: track.class_votes[c])
                    track.cls_name = best_class
                    track.cls_confidence = track.class_votes[best_class]
                    
                    matched_dets.add(di)
                    matched_tracks.add(ti)
        else:
            matched_dets = set()
        
        # Create new tracks for unmatched detections
        # But first, cluster nearby unmatched detections (same object from multiple cameras)
        unmatched = [det for di, det in enumerate(detections) if di not in matched_dets]
        
        if unmatched:
            # Cluster unmatched detections by position
            clusters = []
            used = set()
            
            for i, det1 in enumerate(unmatched):
                if i in used:
                    continue
                    
                cluster = [det1]
                used.add(i)
                
                for j, det2 in enumerate(unmatched):
                    if j in used:
                        continue
                    if det1.cam_id == det2.cam_id:
                        continue  # Same camera = different objects
                        
                    dist = np.linalg.norm(det1.pos_3d[:2] - det2.pos_3d[:2])
                    if dist < association_threshold:
                        cluster.append(det2)
                        used.add(j)
                
                clusters.append(cluster)
            
            # Create track for each cluster
            for cluster in clusters:
                # Compute centroid
                positions = np.array([d.pos_3d for d in cluster])
                centroid = np.mean(positions, axis=0)
                
                # Best class from cluster
                class_votes = defaultdict(float)
                for det in cluster:
                    class_votes[det.cls_name] += det.confidence
                best_class = max(class_votes.keys(), key=lambda c: class_votes[c])
                
                # Create new track
                track = GlobalTrack(
                    track_id=next_track_id,
                    cls_name=best_class,
                    cls_confidence=class_votes[best_class]
                )
                track.state[:2] = centroid[:2]
                track.last_seen_frame = frame_idx
                track.total_detections = len(cluster)
                track.class_votes = dict(class_votes)
                
                # Store detections
                track.frames_seen[frame_idx] = {}
                for det in cluster:
                    track.frames_seen[frame_idx][det.cam_id] = (det.bbox, det.pos_3d, det.confidence)
                
                global_tracks[next_track_id] = track
                next_track_id += 1
    
    # Filter tracks by minimum length
    confirmed_tracks = {
        tid: t for tid, t in global_tracks.items()
        if t.total_detections >= min_track_length
    }
    
    print(f"\nTotal tracks created: {len(global_tracks)}")
    print(f"Confirmed tracks (>= {min_track_length} detections): {len(confirmed_tracks)}")
    
    # Analyze tracks
    cam_counts = defaultdict(int)
    for track in confirmed_tracks.values():
        cams = set()
        for frame_data in track.frames_seen.values():
            cams.update(frame_data.keys())
        cam_counts[len(cams)] += 1
    
    print("\nTracks by camera coverage:")
    for n_cams in sorted(cam_counts.keys()):
        print(f"  {n_cams} camera(s): {cam_counts[n_cams]} tracks")
    
    # ========================================
    # PHASE 3: Generate verification video
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 3: Generating verification video")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate colors
    colors = generate_colors(len(confirmed_tracks) + 10)
    track_colors = {tid: colors[i % len(colors)] for i, tid in enumerate(confirmed_tracks.keys())}
    
    # Reopen video captures
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    # Output video (4x2 grid)
    grid_w, grid_h = 640, 360
    output_w, output_h = grid_w * 2, grid_h * 4
    
    output_path = output_dir / "cross_camera_verification.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        # Read frames
        frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = cv2.resize(frame, (grid_w, grid_h))
            else:
                frames[cam_id] = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Draw detections with global track IDs
        for tid, track in confirmed_tracks.items():
            if frame_idx not in track.frames_seen:
                continue
                
            color = track_colors[tid]
            
            for cam_id, (bbox, pos_3d, conf) in track.frames_seen[frame_idx].items():
                if cam_id not in frames:
                    continue
                    
                frame = frames[cam_id]
                
                # Scale bbox
                scale_x = grid_w / width
                scale_y = grid_h / height
                bbox_scaled = [
                    int(bbox[0] * scale_x), int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x), int(bbox[3] * scale_y)
                ]
                
                # Draw bbox
                cv2.rectangle(frame, (bbox_scaled[0], bbox_scaled[1]),
                              (bbox_scaled[2], bbox_scaled[3]), color, 2)
                
                # Draw label
                label = f"G{tid} {track.cls_name}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame,
                              (bbox_scaled[0], bbox_scaled[1] - label_size[1] - 6),
                              (bbox_scaled[0] + label_size[0] + 4, bbox_scaled[1]),
                              color, -1)
                cv2.putText(frame, label,
                            (bbox_scaled[0] + 2, bbox_scaled[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw 3D position
                pos_label = f"[{pos_3d[0]:.1f},{pos_3d[1]:.1f}]"
                cv2.putText(frame, pos_label,
                            (bbox_scaled[0], bbox_scaled[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add camera labels
        for cam_id in cam_order:
            if cam_id in frames:
                cv2.putText(frames[cam_id], cam_id, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(grid)
    
    # Cleanup
    for cap in caps.values():
        cap.release()
    out.release()
    
    print(f"\nSaved: {output_path}")
    
    # Save track data
    track_data = {
        'total_tracks': len(confirmed_tracks),
        'tracks': {}
    }
    
    for tid, track in confirmed_tracks.items():
        cams_seen = set()
        for frame_data in track.frames_seen.values():
            cams_seen.update(frame_data.keys())
        
        track_data['tracks'][str(tid)] = {
            'class': track.cls_name,
            'total_detections': track.total_detections,
            'cameras': list(cams_seen),
            'frame_range': [min(track.frames_seen.keys()), max(track.frames_seen.keys())]
        }
    
    track_path = output_dir / "unified_tracks.json"
    with open(track_path, 'w') as f:
        json.dump(track_data, f, indent=2)
    print(f"Saved: {track_path}")
    
    # Print summary of multi-camera tracks
    print("\n" + "=" * 70)
    print("Multi-camera tracks (3+ cameras):")
    print("=" * 70)
    
    for tid, track in sorted(confirmed_tracks.items(), key=lambda x: -len(set().union(*[set(f.keys()) for f in x[1].frames_seen.values()]))):
        cams = set()
        for frame_data in track.frames_seen.values():
            cams.update(frame_data.keys())
        if len(cams) >= 3:
            print(f"  G{tid}: {track.cls_name}, {track.total_detections} dets, cameras: {sorted(cams)}")
    
    return confirmed_tracks


def main():
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "StreetAware-sample"
    output_dir = base_dir / "outputs" / "pass2_dynamic"
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    
    run_unified_3d_tracking(
        video_dir, 
        output_dir, 
        cameras_path,
        association_threshold=3.0,  # 3 meters
        max_frames_lost=30,  # Keep predicting for 30 frames during occlusion
        min_track_length=5
    )


if __name__ == "__main__":
    main()
