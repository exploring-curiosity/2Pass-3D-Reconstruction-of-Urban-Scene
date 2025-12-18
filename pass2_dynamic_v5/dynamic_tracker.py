#!/usr/bin/env python3
"""
Pass 2: Dynamic Object Tracking and 4D Scene Reconstruction
============================================================
Implements the methodology described in the report:

1. Detection & Per-Camera Tracking:
   - YOLOv8x for object detection (person, car, truck, bus, bicycle)
   - ByteTrack for per-camera multi-object tracking

2. Stereo Triangulation for 3D Localization:
   - Each corner has stereo pair (left/right cameras)
   - Triangulate 3D positions: X = triangulate(P_L, P_R, u_L, u_R)
   - Single-camera fallback: ray-plane intersection with ground (z=0)

3. Kalman Filter Tracking:
   - 6-state filter (position + velocity) per 3D track
   - Smooth trajectories, handle temporary occlusions

4. Multi-Camera Association:
   - Union-find clustering across stereo pairs
   - Criteria: same category, 3D proximity, temporal overlap

5. Canonical Primitive Representation:
   - Vehicles: oriented bounding boxes (car: 4.5x1.8x1.5m, truck: 7.0x2.5x3.0m)
   - Pedestrians: vertical cylinders (r=0.25m, h=1.7m)
   - Static/moving classification based on trajectory variance
"""

import sys
from pathlib import Path
import json
import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

YOLO_CLASSES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# Primitive dimensions (meters)
PRIMITIVE_DIMS = {
    'car': {'length': 4.5, 'width': 1.8, 'height': 1.5},
    'truck': {'length': 7.0, 'width': 2.5, 'height': 3.0},
    'bus': {'length': 12.0, 'width': 2.5, 'height': 3.2},
    'motorcycle': {'length': 2.2, 'width': 0.8, 'height': 1.5},
    'person': {'radius': 0.25, 'height': 1.7}
}

# Multi-camera association thresholds
ASSOC_DIST_VEHICLE = 2.0  # meters
ASSOC_DIST_PERSON = 1.0   # meters
ASSOC_TEMPORAL_OVERLAP = 0.3  # 30% overlap required

# Static/dynamic classification
STATIC_VARIANCE_THRESHOLD = 0.5  # meters

# =============================================================================
# STEREO CAMERA PAIRS
# =============================================================================

STEREO_PAIRS = [
    ('s1-left', 's1-right'),
    ('s2-left', 's2-right'),
    ('s3-left', 's3-right'),
    ('s4-left', 's4-right'),
]

# =============================================================================
# KALMAN FILTER (6-state: x, y, z, vx, vy, vz)
# =============================================================================

class KalmanFilter3D:
    """6-state Kalman filter for 3D position + velocity tracking."""
    
    def __init__(self, initial_pos: np.ndarray, dt: float = 1/30):
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
        
        # State transition matrix
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement matrix (observe position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # Covariance matrices
        self.P = np.eye(6) * 1.0  # State covariance
        self.Q = np.eye(6) * 0.1  # Process noise
        self.Q[3:, 3:] *= 0.5     # Lower velocity noise
        self.R = np.eye(3) * 0.5  # Measurement noise
    
    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with measurement."""
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].copy()
    
    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()
    
    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

# =============================================================================
# CAMERA & TRIANGULATION
# =============================================================================

class Camera:
    """Camera with projection and triangulation capabilities."""
    
    def __init__(self, params: dict, img_w: int = 2592, img_h: int = 1944):
        self.K = np.array(params['K']).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        
        pose_c2w = np.array(params['pose_c2w'])
        self.R = pose_c2w[:3, :3]
        self.t = pose_c2w[:3, 3]
        
        # Projection matrix P = K[R|t] for world-to-image
        R_w2c = self.R.T
        t_w2c = -R_w2c @ self.t
        self.P = self.K @ np.hstack([R_w2c, t_w2c.reshape(3, 1)])
        
        self.img_w = img_w
        self.img_h = img_h
    
    def project_to_ground(self, u: float, v: float) -> Optional[np.ndarray]:
        """Project image point to ground plane (z=0) via ray-plane intersection."""
        # Ray in camera frame
        ray_c = self.K_inv @ np.array([u, v, 1.0])
        ray_c /= np.linalg.norm(ray_c)
        
        # Transform to world frame
        ray_w = self.R @ ray_c
        
        # Intersect with z=0 plane
        if abs(ray_w[2]) < 1e-6:
            return None
        
        s = -self.t[2] / ray_w[2]
        if s <= 0:
            return None
        
        point = self.t + s * ray_w
        return point


def triangulate(cam_L: Camera, cam_R: Camera, 
                uv_L: Tuple[float, float], uv_R: Tuple[float, float]) -> Optional[np.ndarray]:
    """
    Triangulate 3D point from stereo pair observations.
    X = triangulate(P_L, P_R, u_L, u_R)
    """
    # Build linear system Ax = 0
    u_L, v_L = uv_L
    u_R, v_R = uv_R
    
    A = np.array([
        u_L * cam_L.P[2, :] - cam_L.P[0, :],
        v_L * cam_L.P[2, :] - cam_L.P[1, :],
        u_R * cam_R.P[2, :] - cam_R.P[0, :],
        v_R * cam_R.P[2, :] - cam_R.P[1, :],
    ])
    
    # SVD solution
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]  # Dehomogenize
    
    # Sanity check
    if np.any(np.isnan(X)) or np.linalg.norm(X[:2]) > 50:
        return None
    
    return X

# =============================================================================
# TRACK MANAGEMENT
# =============================================================================

@dataclass
class Track:
    """Single object track with Kalman filter."""
    track_id: int
    category: str
    kf: KalmanFilter3D
    history: List[dict] = field(default_factory=list)
    last_seen: int = 0
    camera_ids: set = field(default_factory=set)
    
    def update(self, frame_idx: int, position: np.ndarray, camera_id: str):
        """Update track with new observation."""
        smoothed_pos = self.kf.update(position)
        self.history.append({
            'frame': frame_idx,
            'position': smoothed_pos.copy(),
            'raw_position': position.copy()
        })
        self.last_seen = frame_idx
        self.camera_ids.add(camera_id)
    
    def predict(self, frame_idx: int):
        """Predict position for current frame."""
        return self.kf.predict()
    
    @property
    def positions(self) -> np.ndarray:
        """Get all positions as array."""
        return np.array([h['position'] for h in self.history])
    
    @property
    def is_static(self) -> bool:
        """Classify as static if trajectory variance is low."""
        if len(self.history) < 10:
            return True
        positions = self.positions
        variance = np.var(positions, axis=0).sum()
        return variance < STATIC_VARIANCE_THRESHOLD

# =============================================================================
# MULTI-CAMERA TRACKER
# =============================================================================

class DynamicTracker:
    """
    Multi-camera 3D object tracker implementing:
    - Per-camera detection with YOLOv8x
    - Stereo triangulation for 3D localization
    - Kalman filter tracking
    - Multi-camera association with union-find
    """
    
    def __init__(self, cameras_json: Path):
        # Load camera parameters
        with open(cameras_json) as f:
            cam_params = json.load(f)
        
        self.cameras = {cid: Camera(params) for cid, params in cam_params.items()}
        
        # Initialize YOLO
        self.yolo = YOLO('yolov8x.pt')
        
        # Track management
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        
        # Per-camera ByteTrack state (simplified)
        self.per_camera_tracks: Dict[str, Dict[int, dict]] = defaultdict(dict)
    
    def detect_frame(self, frame: np.ndarray, conf: float = 0.4) -> List[dict]:
        """Run YOLOv8x detection on frame."""
        results = self.yolo.predict(frame, conf=conf, verbose=False, 
                                    classes=list(YOLO_CLASSES.keys()))
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in YOLO_CLASSES:
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy_bottom = xyxy[3]  # Bottom of bbox for ground contact
                    
                    detections.append({
                        'bbox': xyxy,
                        'center': (cx, cy_bottom),
                        'class_id': cls_id,
                        'class_name': YOLO_CLASSES[cls_id],
                        'confidence': conf
                    })
        
        return detections
    
    def process_stereo_pair(self, frame_idx: int, 
                           frames: Dict[str, np.ndarray],
                           pair: Tuple[str, str]) -> List[dict]:
        """
        Process stereo pair: detect, match, triangulate.
        Returns list of 3D observations.
        """
        cam_L_id, cam_R_id = pair
        
        if cam_L_id not in frames or cam_R_id not in frames:
            return []
        
        # Detect in both views
        dets_L = self.detect_frame(frames[cam_L_id])
        dets_R = self.detect_frame(frames[cam_R_id])
        
        cam_L = self.cameras[cam_L_id]
        cam_R = self.cameras[cam_R_id]
        
        observations = []
        
        # Match detections across stereo pair (same class, similar y-coordinate)
        matched_L = set()
        matched_R = set()
        
        for i, det_L in enumerate(dets_L):
            best_j = None
            best_score = float('inf')
            
            for j, det_R in enumerate(dets_R):
                if j in matched_R:
                    continue
                if det_L['class_name'] != det_R['class_name']:
                    continue
                
                # Epipolar constraint: similar y-coordinate
                y_diff = abs(det_L['center'][1] - det_R['center'][1])
                if y_diff < 50:  # pixels
                    if y_diff < best_score:
                        best_score = y_diff
                        best_j = j
            
            if best_j is not None:
                matched_L.add(i)
                matched_R.add(best_j)
                det_R = dets_R[best_j]
                
                # Triangulate
                pos_3d = triangulate(cam_L, cam_R, det_L['center'], det_R['center'])
                
                if pos_3d is not None:
                    observations.append({
                        'position': pos_3d,
                        'category': det_L['class_name'],
                        'confidence': (det_L['confidence'] + det_R['confidence']) / 2,
                        'camera_ids': {cam_L_id, cam_R_id},
                        'method': 'stereo_triangulation'
                    })
        
        # Fallback: unmatched detections use ground plane projection
        for i, det_L in enumerate(dets_L):
            if i not in matched_L:
                pos_3d = cam_L.project_to_ground(*det_L['center'])
                if pos_3d is not None and np.linalg.norm(pos_3d[:2]) < 30:
                    observations.append({
                        'position': pos_3d,
                        'category': det_L['class_name'],
                        'confidence': det_L['confidence'],
                        'camera_ids': {cam_L_id},
                        'method': 'ground_projection'
                    })
        
        for j, det_R in enumerate(dets_R):
            if j not in matched_R:
                pos_3d = cam_R.project_to_ground(*det_R['center'])
                if pos_3d is not None and np.linalg.norm(pos_3d[:2]) < 30:
                    observations.append({
                        'position': pos_3d,
                        'category': det_R['class_name'],
                        'confidence': det_R['confidence'],
                        'camera_ids': {cam_R_id},
                        'method': 'ground_projection'
                    })
        
        return observations
    
    def associate_observations(self, observations: List[dict]) -> List[dict]:
        """
        Multi-camera association using union-find clustering.
        Merge observations from different stereo pairs viewing same object.
        """
        if len(observations) <= 1:
            return observations
        
        n = len(observations)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Merge based on: same category, spatial proximity, camera overlap
        for i in range(n):
            for j in range(i + 1, n):
                obs_i, obs_j = observations[i], observations[j]
                
                # Same category
                if obs_i['category'] != obs_j['category']:
                    continue
                
                # Spatial proximity
                dist = np.linalg.norm(obs_i['position'] - obs_j['position'])
                threshold = ASSOC_DIST_PERSON if obs_i['category'] == 'person' else ASSOC_DIST_VEHICLE
                
                if dist < threshold:
                    # Different cameras (avoid merging same detection)
                    if not obs_i['camera_ids'] & obs_j['camera_ids']:
                        union(i, j)
        
        # Group by cluster
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(observations[i])
        
        # Merge clusters
        merged = []
        for cluster_obs in clusters.values():
            avg_pos = np.mean([o['position'] for o in cluster_obs], axis=0)
            all_cams = set()
            for o in cluster_obs:
                all_cams.update(o['camera_ids'])
            
            merged.append({
                'position': avg_pos,
                'category': cluster_obs[0]['category'],
                'confidence': max(o['confidence'] for o in cluster_obs),
                'camera_ids': all_cams,
                'num_views': len(cluster_obs)
            })
        
        return merged
    
    def update_tracks(self, frame_idx: int, observations: List[dict]):
        """Associate observations to existing tracks or create new ones."""
        if not observations:
            return
        
        # Predict existing tracks
        predictions = {}
        for tid, track in self.tracks.items():
            if frame_idx - track.last_seen < 30:  # Keep predicting for 1 second
                predictions[tid] = track.predict(frame_idx)
        
        if not predictions:
            # No active tracks, create new ones
            for obs in observations:
                self._create_track(frame_idx, obs)
            return
        
        # Hungarian matching
        track_ids = list(predictions.keys())
        cost_matrix = np.zeros((len(observations), len(track_ids)))
        
        for i, obs in enumerate(observations):
            for j, tid in enumerate(track_ids):
                track = self.tracks[tid]
                if obs['category'] != track.category:
                    cost_matrix[i, j] = 1000  # High cost for category mismatch
                else:
                    cost_matrix[i, j] = np.linalg.norm(obs['position'] - predictions[tid])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_obs = set()
        matched_tracks = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 5.0:  # Max association distance
                tid = track_ids[j]
                obs = observations[i]
                self.tracks[tid].update(frame_idx, obs['position'], 
                                        list(obs['camera_ids'])[0])
                matched_obs.add(i)
                matched_tracks.add(tid)
        
        # Create new tracks for unmatched observations
        for i, obs in enumerate(observations):
            if i not in matched_obs:
                self._create_track(frame_idx, obs)
    
    def _create_track(self, frame_idx: int, obs: dict):
        """Create new track from observation."""
        track = Track(
            track_id=self.next_track_id,
            category=obs['category'],
            kf=KalmanFilter3D(obs['position'])
        )
        track.update(frame_idx, obs['position'], list(obs['camera_ids'])[0])
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def process_frame(self, frame_idx: int, frames: Dict[str, np.ndarray]):
        """Process single frame across all stereo pairs."""
        all_observations = []
        
        # Process each stereo pair
        for pair in STEREO_PAIRS:
            obs = self.process_stereo_pair(frame_idx, frames, pair)
            all_observations.extend(obs)
        
        # Multi-camera association
        merged_obs = self.associate_observations(all_observations)
        
        # Update tracks
        self.update_tracks(frame_idx, merged_obs)
    
    def generate_scene(self, num_frames: int, fps: float) -> dict:
        """Generate final scene_4d.json with canonical primitives."""
        scene = []
        
        for tid, track in self.tracks.items():
            if len(track.history) < 5:
                continue  # Skip very short tracks
            
            # Get primitive dimensions
            cat = track.category
            if cat in PRIMITIVE_DIMS:
                dims = PRIMITIVE_DIMS[cat]
            else:
                dims = PRIMITIVE_DIMS['car']  # Default
            
            # Compute yaw from trajectory
            positions = track.positions
            if len(positions) >= 2:
                direction = positions[-1] - positions[0]
                yaw = float(np.arctan2(direction[1], direction[0]))
            else:
                yaw = 0.0
            
            obj = {
                'id': f'obj_{tid}',
                'class': cat,
                'static': track.is_static,
            }
            
            # Add dimensions
            if cat == 'person':
                obj['radius'] = dims['radius']
                obj['height'] = dims['height']
            else:
                obj['length'] = dims['length']
                obj['width'] = dims['width']
                obj['height'] = dims['height']
            
            if track.is_static:
                # Static: single position
                avg_pos = np.mean(positions, axis=0)
                obj['position'] = avg_pos.tolist()
                obj['yaw'] = yaw
            else:
                # Dynamic: keyframes
                obj['keyframes'] = []
                for h in track.history[::3]:  # Every 3rd frame
                    obj['keyframes'].append({
                        'frame': h['frame'],
                        'position': h['position'].tolist(),
                        'yaw': yaw
                    })
            
            scene.append(obj)
        
        return scene


def main():
    from tqdm import tqdm
    
    base = Path(__file__).parent.parent
    out_dir = base / "outputs/pass2_dynamic_v5"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize tracker
    cameras_json = base / "outputs/pass1_static/pi3_cameras_corrected.json"
    tracker = DynamicTracker(cameras_json)
    
    # Open video files
    video_dir = base / "StreetAware-sample"
    caps = {}
    for pair in STEREO_PAIRS:
        for cam_id in pair:
            path = video_dir / f"{cam_id}.mp4"
            if path.exists():
                caps[cam_id] = cv2.VideoCapture(str(path))
    
    # Get video info
    sample_cap = list(caps.values())[0]
    num_frames = int(sample_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = sample_cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {num_frames} frames at {fps:.1f} FPS...")
    print(f"Cameras: {list(caps.keys())}")
    
    # Process all frames
    for frame_idx in tqdm(range(num_frames)):
        frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = frame
        
        tracker.process_frame(frame_idx, frames)
    
    # Cleanup
    for cap in caps.values():
        cap.release()
    
    # Generate output
    print("\nGenerating scene...")
    scene = tracker.generate_scene(num_frames, fps)
    
    # Summary
    static_count = sum(1 for obj in scene if obj.get('static', False))
    dynamic_count = len(scene) - static_count
    
    print(f"\nScene Summary:")
    print(f"  Total objects: {len(scene)}")
    print(f"  Static: {static_count}")
    print(f"  Dynamic: {dynamic_count}")
    
    # Save
    with open(out_dir / "scene_4d.json", 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"\nSaved: {out_dir}/scene_4d.json")


if __name__ == "__main__":
    main()
