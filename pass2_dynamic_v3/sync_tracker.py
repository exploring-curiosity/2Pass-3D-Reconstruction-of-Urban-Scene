#!/usr/bin/env python3
"""
Synchronized Multi-Camera 4D Tracker
======================================
Key improvement: Process ALL cameras simultaneously per frame.
No per-camera ByteTrack - we do our own tracking in 3D space.

Strategy:
1. Per frame: detect in all 8 cameras
2. Project detections to 3D
3. Cluster detections across cameras (same object seen by multiple)
4. Match clusters to existing global tracks using Hungarian algorithm
5. Smooth and output
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# =============================================================================
# CONFIG
# =============================================================================

MIN_CONFIDENCE = 0.45
CENTER_ZONE_RATIO = 0.7
MAX_3D_DISTANCE = 30.0

# Clustering
CLUSTER_EPS = 4.0        # meters - max distance for same object
CLUSTER_MIN_CAMS = 1     # minimum cameras to see object

# Tracking
MAX_GAP = 15             # frames to keep track alive without detection
POSITION_WEIGHT = 0.7    # weight for position in cost function
VELOCITY_WEIGHT = 0.3    # weight for velocity matching

# Output limits
MAX_STATIC = 15
MAX_DYNAMIC = 15
MIN_STATIC_FRAMES = 0.3  # 30% of frames
MIN_DYNAMIC_TRAVEL = 5.0  # meters

# =============================================================================
# GROUND MASK CHECKER
# =============================================================================

class GroundMaskChecker:
    def __init__(self, mask_dir: Path):
        road_path = mask_dir / "road_grid.npy"
        curb_path = mask_dir / "curb_grid.npy"
        info_path = mask_dir / "grid_info.json"
        
        if not road_path.exists():
            print("  WARNING: No ground masks found")
            self.road_grid = None
            self.curb_grid = None
            return
        
        self.road_grid = np.load(road_path)
        self.curb_grid = np.load(curb_path)
        with open(info_path) as f:
            self.info = json.load(f)
        print(f"  Loaded ground masks: {self.road_grid.shape}")
    
    def is_on_road(self, pos: np.ndarray) -> bool:
        if self.road_grid is None:
            return True
        gx = int((pos[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos[1] - self.info['origin'][1]) / self.info['resolution'])
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.road_grid[gy, gx])
        return False
    
    def is_on_curb(self, pos: np.ndarray) -> bool:
        if self.curb_grid is None:
            return False
        gx = int((pos[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos[1] - self.info['origin'][1]) / self.info['resolution'])
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.curb_grid[gy, gx])
        return False

# =============================================================================
# CAMERA PROJECTION
# =============================================================================

class CameraProjector:
    def __init__(self, cam_params, img_w=2592, img_h=1944):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]
        
        self.img_w = img_w
        self.img_h = img_h
        
        margin_x = img_w * (1 - CENTER_ZONE_RATIO) / 2
        margin_y = img_h * (1 - CENTER_ZONE_RATIO) / 2
        self.center_bounds = (margin_x, margin_y, img_w - margin_x, img_h - margin_y)
    
    def is_in_center(self, u: float, v: float) -> bool:
        return (self.center_bounds[0] <= u <= self.center_bounds[2] and
                self.center_bounds[1] <= v <= self.center_bounds[3])
    
    def project_to_ground(self, u: float, v: float, z: float = 0.0):
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        ray_world = self.R_c2w @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        if abs(ray_world[2]) < 1e-6:
            return None
        t = (z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
        point = self.t_c2w + t * ray_world
        if np.linalg.norm(point[:2]) > MAX_3D_DISTANCE:
            return None
        return point

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Detection3D:
    """Single 3D detection from one camera."""
    pos: np.ndarray
    cls: str
    conf: float
    camera: str

@dataclass
class Observation:
    """Clustered observation from multiple cameras at one frame."""
    pos: np.ndarray
    cls: str
    conf: float
    cameras: List[str]

@dataclass
class GlobalTrack:
    """Track with global ID maintained across cameras."""
    track_id: int
    cls: str
    is_static: bool = False
    frames: Dict[int, np.ndarray] = field(default_factory=dict)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: int = 0
    cameras_seen: Set[str] = field(default_factory=set)
    
    @property
    def length(self) -> int:
        return len(self.frames)
    
    def predict_position(self, current_frame: int) -> np.ndarray:
        """Predict position based on velocity."""
        if not self.frames:
            return np.zeros(3)
        last_pos = self.frames[self.last_seen]
        dt = current_frame - self.last_seen
        return last_pos + self.velocity * dt
    
    def update(self, frame_idx: int, pos: np.ndarray, cameras: List[str]):
        """Update track with new observation."""
        if self.frames:
            last_pos = self.frames[self.last_seen]
            dt = frame_idx - self.last_seen
            if dt > 0:
                self.velocity = 0.7 * self.velocity + 0.3 * (pos - last_pos) / dt
        
        self.frames[frame_idx] = pos.copy()
        self.last_seen = frame_idx
        self.cameras_seen.update(cameras)
    
    def get_travel_distance(self) -> float:
        """Total distance traveled."""
        if len(self.frames) < 2:
            return 0.0
        frames_list = sorted(self.frames.keys())
        positions = np.array([self.frames[f][:2] for f in frames_list])
        return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

# =============================================================================
# SYNCHRONIZED MULTI-CAMERA TRACKER
# =============================================================================

class SynchronizedTracker:
    """Main tracker that processes all cameras simultaneously."""
    
    def __init__(self, cameras: dict, ground_checker: GroundMaskChecker):
        self.projectors = {cam_id: CameraProjector(cameras[cam_id]) 
                          for cam_id in cameras.keys()}
        self.ground_checker = ground_checker
        self.yolo = YOLO('yolov8x.pt')
        
        self.tracks: Dict[int, GlobalTrack] = {}
        self.next_track_id = 1
    
    def detect_all_cameras(self, frames: Dict[str, np.ndarray]) -> List[Detection3D]:
        """Detect objects in all camera frames and project to 3D."""
        all_detections = []
        
        for cam_id, frame in frames.items():
            if cam_id not in self.projectors:
                continue
            
            proj = self.projectors[cam_id]
            
            results = self.yolo.predict(frame, conf=MIN_CONFIDENCE, verbose=False,
                                        classes=list(VALID_CLASSES.keys()))
            
            if results[0].boxes is None:
                continue
            
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                # Bottom center
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]
                
                # Center zone filter
                if not proj.is_in_center(cx, cy):
                    continue
                
                # Project to 3D
                pos_3d = proj.project_to_ground(cx, cy)
                if pos_3d is None:
                    continue
                
                all_detections.append(Detection3D(
                    pos=pos_3d,
                    cls=cls_name,
                    conf=conf,
                    camera=cam_id
                ))
        
        return all_detections
    
    def cluster_detections(self, detections: List[Detection3D]) -> List[Observation]:
        """Cluster detections across cameras."""
        if not detections:
            return []
        
        # Group by class first
        by_class = defaultdict(list)
        for det in detections:
            # Allow car/truck to be same
            key = 'vehicle' if det.cls in VEHICLE_CLASSES else det.cls
            by_class[key].append(det)
        
        observations = []
        
        for cls_key, class_dets in by_class.items():
            if len(class_dets) < CLUSTER_MIN_CAMS:
                continue
            
            positions = np.array([d.pos[:2] for d in class_dets])
            
            # DBSCAN clustering
            if len(positions) >= 2:
                db = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit(positions)
                labels = db.labels_
            else:
                labels = [0] * len(positions)
            
            for label in set(labels):
                if label == -1:
                    continue
                
                indices = [i for i, l in enumerate(labels) if l == label]
                cluster_dets = [class_dets[i] for i in indices]
                
                # Average position
                avg_pos = np.mean([d.pos for d in cluster_dets], axis=0)
                avg_conf = np.mean([d.conf for d in cluster_dets])
                cameras = list(set(d.camera for d in cluster_dets))
                
                # Get actual class (not 'vehicle' key)
                actual_cls = cluster_dets[0].cls
                
                observations.append(Observation(
                    pos=avg_pos,
                    cls=actual_cls,
                    conf=avg_conf,
                    cameras=cameras
                ))
        
        return observations
    
    def update_tracks(self, frame_idx: int, observations: List[Observation]):
        """Match observations to existing tracks and update."""
        
        # Remove dead tracks
        dead_ids = [tid for tid, track in self.tracks.items() 
                   if frame_idx - track.last_seen > MAX_GAP]
        for tid in dead_ids:
            del self.tracks[tid]
        
        if not observations:
            return
        
        # Get active tracks
        active_tracks = list(self.tracks.values())
        
        if not active_tracks:
            # Create new tracks for all observations
            for obs in observations:
                track = GlobalTrack(
                    track_id=self.next_track_id,
                    cls=obs.cls
                )
                track.update(frame_idx, obs.pos, obs.cameras)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
            return
        
        # Build cost matrix
        cost_matrix = np.zeros((len(observations), len(active_tracks)))
        
        for oi, obs in enumerate(observations):
            for ti, track in enumerate(active_tracks):
                # Class mismatch penalty
                if obs.cls != track.cls:
                    if not ({obs.cls, track.cls} <= {'car', 'truck'}):
                        cost_matrix[oi, ti] = 1000
                        continue
                
                # Position cost
                predicted_pos = track.predict_position(frame_idx)
                pos_dist = np.linalg.norm(obs.pos[:2] - predicted_pos[:2])
                
                # Velocity consistency
                if track.velocity is not None and np.linalg.norm(track.velocity) > 0.1:
                    obs_vel = obs.pos - track.frames[track.last_seen] if track.frames else np.zeros(3)
                    vel_diff = np.linalg.norm(obs_vel[:2] - track.velocity[:2])
                    cost_matrix[oi, ti] = POSITION_WEIGHT * pos_dist + VELOCITY_WEIGHT * vel_diff
                else:
                    cost_matrix[oi, ti] = pos_dist
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_obs = set()
        matched_tracks = set()
        
        for oi, ti in zip(row_ind, col_ind):
            if cost_matrix[oi, ti] < 10.0:  # Max distance threshold
                track = active_tracks[ti]
                obs = observations[oi]
                track.update(frame_idx, obs.pos, obs.cameras)
                matched_obs.add(oi)
                matched_tracks.add(ti)
        
        # Create new tracks for unmatched observations
        for oi, obs in enumerate(observations):
            if oi not in matched_obs:
                track = GlobalTrack(
                    track_id=self.next_track_id,
                    cls=obs.cls
                )
                track.update(frame_idx, obs.pos, obs.cameras)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
    
    def process_video(self, video_dir: Path, total_frames: int):
        """Process all frames with synchronized detection."""
        
        cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
                   's3-left', 's3-right', 's4-left', 's4-right']
        
        # Open all video captures
        caps = {}
        for cam_id in cam_ids:
            vpath = video_dir / f"{cam_id}.mp4"
            if vpath.exists():
                caps[cam_id] = cv2.VideoCapture(str(vpath))
        
        print(f"\n  Processing {len(caps)} cameras synchronously...")
        
        for frame_idx in tqdm(range(total_frames), desc="  Tracking"):
            # Read all frames
            frames = {}
            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[cam_id] = frame
            
            if not frames:
                break
            
            # Detect in all cameras
            detections = self.detect_all_cameras(frames)
            
            # Cluster across cameras
            observations = self.cluster_detections(detections)
            
            # Update tracks
            self.update_tracks(frame_idx, observations)
        
        # Cleanup
        for cap in caps.values():
            cap.release()
        
        print(f"  Total unique tracks: {len(self.tracks)}")
    
    def classify_and_smooth(self, total_frames: int) -> Tuple[List[GlobalTrack], List[GlobalTrack]]:
        """Classify tracks as static/dynamic and smooth."""
        
        static_tracks = []
        dynamic_tracks = []
        
        for track in self.tracks.values():
            if track.length < 10:
                continue
            
            travel = track.get_travel_distance()
            presence = track.length / total_frames
            
            # Get median position
            positions = list(track.frames.values())
            median_pos = np.median(positions, axis=0)
            
            is_on_curb = self.ground_checker.is_on_curb(median_pos)
            is_on_road = self.ground_checker.is_on_road(median_pos)
            
            # Static: low travel, high presence, on curb or not on road
            if travel < 3.0 and presence >= MIN_STATIC_FRAMES:
                if is_on_curb or (not is_on_road):
                    track.is_static = True
                    # Lock position
                    for f in range(total_frames):
                        track.frames[f] = median_pos.copy()
                    static_tracks.append(track)
                    continue
            
            # Dynamic: significant travel
            if travel >= MIN_DYNAMIC_TRAVEL:
                track.is_static = False
                
                # Fill gaps and smooth
                frames_list = sorted(track.frames.keys())
                
                # Fill gaps
                for i in range(len(frames_list) - 1):
                    gap = frames_list[i+1] - frames_list[i]
                    if gap > 1 and gap <= MAX_GAP:
                        p1 = track.frames[frames_list[i]]
                        p2 = track.frames[frames_list[i+1]]
                        for f in range(frames_list[i] + 1, frames_list[i+1]):
                            alpha = (f - frames_list[i]) / gap
                            track.frames[f] = p1 + alpha * (p2 - p1)
                
                # Smooth
                frames_list = sorted(track.frames.keys())
                if len(frames_list) >= 5:
                    for dim in range(3):
                        vals = np.array([track.frames[f][dim] for f in frames_list])
                        smoothed = uniform_filter1d(vals, size=11, mode='nearest')
                        for i, f in enumerate(frames_list):
                            track.frames[f][dim] = smoothed[i]
                
                dynamic_tracks.append(track)
        
        # Sort and limit
        static_tracks.sort(key=lambda t: t.length, reverse=True)
        dynamic_tracks.sort(key=lambda t: t.get_travel_distance(), reverse=True)
        
        static_tracks = static_tracks[:MAX_STATIC]
        dynamic_tracks = dynamic_tracks[:MAX_DYNAMIC]
        
        return static_tracks, dynamic_tracks

# =============================================================================
# OUTPUT
# =============================================================================

def generate_scene(static_tracks: List[GlobalTrack], dynamic_tracks: List[GlobalTrack],
                   total_frames: int, fps: float) -> dict:
    
    DIMS = {
        'car': [4.5, 1.8, 1.5],
        'truck': [7.0, 2.4, 2.8],
        'bus': [10.0, 2.5, 3.0],
        'motorcycle': [2.0, 0.8, 1.3],
        'bicycle': [1.8, 0.5, 1.4],
        'person': [0.5, 0.5, 1.7]
    }
    
    COLORS = {
        'car': [0, 0, 255],
        'truck': [255, 0, 200],
        'person': [0, 255, 0],
        'bicycle': [0, 255, 128],
        'motorcycle': [255, 128, 0]
    }
    
    scene = {
        'total_frames': total_frames,
        'fps': fps,
        'objects': {},
        'frames': {}
    }
    
    # Static
    for track in static_tracks:
        tid = f"S{track.track_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': [128, 128, 128],
            'is_stationary': True
        }
        
        for frame_idx, pos in track.frames.items():
            frame_key = str(frame_idx)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            quat = R.from_euler('z', 0.0).as_quat().tolist()
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': pos.tolist(),
                'rot': quat,
                'conf': 1.0
            })
    
    # Dynamic
    for track in dynamic_tracks:
        tid = f"D{track.track_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': COLORS.get(track.cls, [255, 255, 255]),
            'is_stationary': False
        }
        
        frames_list = sorted(track.frames.keys())
        positions = np.array([track.frames[f][:2] for f in frames_list])
        
        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([velocities, velocities[-1] if len(velocities) > 0 else [[1, 0]]])
        yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
        yaws = uniform_filter1d(np.unwrap(yaws), size=11, mode='nearest')
        
        for i, frame_idx in enumerate(frames_list):
            frame_key = str(frame_idx)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            pos = track.frames[frame_idx]
            quat = R.from_euler('z', float(yaws[i])).as_quat().tolist()
            
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': pos.tolist(),
                'rot': quat,
                'conf': 0.9
            })
    
    return scene

# =============================================================================
# MAIN
# =============================================================================

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = Path(__file__).parent
    out_dir = base_dir / "outputs" / "pass2_dynamic_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SYNCHRONIZED MULTI-CAMERA 4D TRACKER")
    print("All cameras processed simultaneously per frame")
    print("=" * 70)
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    mask_dir = base_dir / "outputs" / "pass1_static" / "ground_masks"
    ground_checker = GroundMaskChecker(mask_dir)
    
    video_dir = base_dir / "StreetAware-sample"
    
    test_cap = cv2.VideoCapture(str(video_dir / "s1-left.mp4"))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS)
    test_cap.release()
    
    print(f"\nTotal frames: {total_frames}, FPS: {fps:.1f}")
    
    # Initialize tracker
    tracker = SynchronizedTracker(cameras, ground_checker)
    
    # Process all frames
    print("\n=== SYNCHRONIZED TRACKING ===")
    tracker.process_video(video_dir, total_frames)
    
    # Classify and smooth
    print("\n=== CLASSIFICATION & SMOOTHING ===")
    static_tracks, dynamic_tracks = tracker.classify_and_smooth(total_frames)
    
    print(f"  Static: {len(static_tracks)}, Dynamic: {len(dynamic_tracks)}")
    
    # Generate scene
    print("\n=== GENERATING OUTPUT ===")
    scene = generate_scene(static_tracks, dynamic_tracks, total_frames, fps)
    
    scene_path = work_dir / "scene_4d.json"
    with open(scene_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    with open(out_dir / "scene_4d.json", 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"\nSaved: {scene_path}")
    print("Pipeline complete!")

if __name__ == "__main__":
    main()
