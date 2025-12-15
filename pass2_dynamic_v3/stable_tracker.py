#!/usr/bin/env python3
"""
URGENT FIX - Stable 4D Tracker
===============================
Fixes the dancing/rotating cars by:
1. Much stronger position smoothing (Savitzky-Golay filter)
2. Stable orientation from overall trajectory direction, not frame-to-frame
3. Static objects locked completely
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
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# Config
MIN_CONFIDENCE = 0.5
CENTER_ZONE_RATIO = 0.6  # Stricter center zone
MAX_3D_DISTANCE = 25.0
CLUSTER_EPS = 3.0
MAX_GAP = 10
MAX_STATIC = 15
MAX_DYNAMIC = 15

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
        
        margin_x = img_w * (1 - CENTER_ZONE_RATIO) / 2
        margin_y = img_h * (1 - CENTER_ZONE_RATIO) / 2
        self.center_bounds = (margin_x, margin_y, img_w - margin_x, img_h - margin_y)
    
    def is_in_center(self, u: float, v: float) -> bool:
        return (self.center_bounds[0] <= u <= self.center_bounds[2] and
                self.center_bounds[1] <= v <= self.center_bounds[3])
    
    def project_to_ground(self, u: float, v: float) -> Optional[np.ndarray]:
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        ray_world = self.R_c2w @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        if abs(ray_world[2]) < 1e-6:
            return None
        t = -self.t_c2w[2] / ray_world[2]
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
    pos: np.ndarray
    cls: str
    conf: float
    camera: str

@dataclass 
class GlobalTrack:
    track_id: int
    cls: str
    is_static: bool = False
    frames: Dict[int, np.ndarray] = field(default_factory=dict)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: int = 0
    
    def get_travel_distance(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        frames_list = sorted(self.frames.keys())
        positions = np.array([self.frames[f][:2] for f in frames_list])
        return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    def get_main_direction(self) -> float:
        """Get overall travel direction (stable yaw)."""
        if len(self.frames) < 2:
            return 0.0
        frames_list = sorted(self.frames.keys())
        positions = np.array([self.frames[f][:2] for f in frames_list])
        
        # Use first and last position for overall direction
        start = positions[0]
        end = positions[-1]
        delta = end - start
        
        if np.linalg.norm(delta) < 1.0:  # Barely moved
            return 0.0
        
        return np.arctan2(delta[1], delta[0])

# =============================================================================
# TRACKER
# =============================================================================

class StableTracker:
    def __init__(self, cameras: dict):
        self.projectors = {cam_id: CameraProjector(cameras[cam_id]) 
                          for cam_id in cameras.keys()}
        self.yolo = YOLO('yolov8x.pt')
        self.tracks: Dict[int, GlobalTrack] = {}
        self.next_track_id = 1
    
    def detect_all_cameras(self, frames: Dict[str, np.ndarray]) -> List[Detection3D]:
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
                
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]
                
                if not proj.is_in_center(cx, cy):
                    continue
                
                pos_3d = proj.project_to_ground(cx, cy)
                if pos_3d is not None:
                    all_detections.append(Detection3D(pos=pos_3d, cls=cls_name, 
                                                       conf=conf, camera=cam_id))
        
        return all_detections
    
    def cluster_detections(self, detections: List[Detection3D]) -> List[Tuple[np.ndarray, str, float]]:
        if not detections:
            return []
        
        by_class = defaultdict(list)
        for det in detections:
            key = 'vehicle' if det.cls in VEHICLE_CLASSES else det.cls
            by_class[key].append(det)
        
        observations = []
        
        for cls_key, class_dets in by_class.items():
            positions = np.array([d.pos[:2] for d in class_dets])
            
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
                avg_pos = np.mean([d.pos for d in cluster_dets], axis=0)
                avg_conf = np.mean([d.conf for d in cluster_dets])
                observations.append((avg_pos, cluster_dets[0].cls, avg_conf))
        
        return observations
    
    def update_tracks(self, frame_idx: int, observations):
        # Remove dead tracks
        dead = [tid for tid, t in self.tracks.items() if frame_idx - t.last_seen > MAX_GAP]
        for tid in dead:
            del self.tracks[tid]
        
        if not observations:
            return
        
        active = list(self.tracks.values())
        
        if not active:
            for pos, cls, conf in observations:
                track = GlobalTrack(track_id=self.next_track_id, cls=cls)
                track.frames[frame_idx] = pos.copy()
                track.last_seen = frame_idx
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
            return
        
        # Cost matrix
        cost = np.full((len(observations), len(active)), 1000.0)
        
        for oi, (pos, cls, conf) in enumerate(observations):
            for ti, track in enumerate(active):
                if cls != track.cls and not ({cls, track.cls} <= {'car', 'truck'}):
                    continue
                
                # Predict position
                if track.last_seen in track.frames:
                    last_pos = track.frames[track.last_seen]
                    dt = frame_idx - track.last_seen
                    predicted = last_pos + track.velocity * dt
                else:
                    predicted = pos
                
                dist = np.linalg.norm(pos[:2] - predicted[:2])
                cost[oi, ti] = dist
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        matched_obs = set()
        for oi, ti in zip(row_ind, col_ind):
            if cost[oi, ti] < 8.0:  # Max match distance
                track = active[ti]
                pos = observations[oi][0]
                
                # Update velocity
                if track.frames:
                    last = track.frames[track.last_seen]
                    dt = frame_idx - track.last_seen
                    if dt > 0:
                        track.velocity = 0.8 * track.velocity + 0.2 * (pos - last) / dt
                
                track.frames[frame_idx] = pos.copy()
                track.last_seen = frame_idx
                matched_obs.add(oi)
        
        # New tracks for unmatched
        for oi, (pos, cls, conf) in enumerate(observations):
            if oi not in matched_obs:
                track = GlobalTrack(track_id=self.next_track_id, cls=cls)
                track.frames[frame_idx] = pos.copy()
                track.last_seen = frame_idx
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
    
    def process_video(self, video_dir: Path, total_frames: int):
        cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
                   's3-left', 's3-right', 's4-left', 's4-right']
        
        caps = {}
        for cam_id in cam_ids:
            vpath = video_dir / f"{cam_id}.mp4"
            if vpath.exists():
                caps[cam_id] = cv2.VideoCapture(str(vpath))
        
        print(f"\n  Processing {len(caps)} cameras...")
        
        for frame_idx in tqdm(range(total_frames), desc="  Tracking"):
            frames = {}
            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[cam_id] = frame
            
            if not frames:
                break
            
            detections = self.detect_all_cameras(frames)
            observations = self.cluster_detections(detections)
            self.update_tracks(frame_idx, observations)
        
        for cap in caps.values():
            cap.release()
        
        print(f"  Tracks: {len(self.tracks)}")
    
    def stabilize_and_classify(self, total_frames: int):
        """Apply AGGRESSIVE smoothing and classify."""
        static_tracks = []
        dynamic_tracks = []
        
        for track in self.tracks.values():
            if len(track.frames) < 15:
                continue
            
            travel = track.get_travel_distance()
            presence = len(track.frames) / total_frames
            
            frames_list = sorted(track.frames.keys())
            positions = np.array([track.frames[f] for f in frames_list])
            
            # AGGRESSIVE SMOOTHING with Savitzky-Golay
            if len(positions) >= 15:
                window = min(15, (len(positions) // 2) * 2 - 1)  # Ensure odd and < length
                if window >= 5:
                    for dim in range(3):
                        positions[:, dim] = savgol_filter(positions[:, dim], window, 2, mode='nearest')
            
            # Update track with smoothed positions
            for i, f in enumerate(frames_list):
                track.frames[f] = positions[i].copy()
            
            # Recompute travel after smoothing
            travel = track.get_travel_distance()
            
            # Classify
            if travel < 2.0 and presence >= 0.15:  # Lower threshold for static
                track.is_static = True
                median_pos = np.median(positions, axis=0)
                for f in range(total_frames):
                    track.frames[f] = median_pos.copy()
                static_tracks.append(track)
            elif travel >= 3.0:
                track.is_static = False
                
                # Fill gaps
                for i in range(len(frames_list) - 1):
                    gap = frames_list[i+1] - frames_list[i]
                    if gap > 1 and gap <= MAX_GAP:
                        p1 = track.frames[frames_list[i]]
                        p2 = track.frames[frames_list[i+1]]
                        for f in range(frames_list[i] + 1, frames_list[i+1]):
                            alpha = (f - frames_list[i]) / gap
                            track.frames[f] = p1 + alpha * (p2 - p1)
                
                dynamic_tracks.append(track)
        
        static_tracks.sort(key=lambda t: len(t.frames), reverse=True)
        dynamic_tracks.sort(key=lambda t: t.get_travel_distance(), reverse=True)
        
        return static_tracks[:MAX_STATIC], dynamic_tracks[:MAX_DYNAMIC]

def generate_scene(static_tracks, dynamic_tracks, total_frames, fps):
    DIMS = {
        'car': [4.5, 1.8, 1.5], 'truck': [7.0, 2.4, 2.8],
        'bus': [10.0, 2.5, 3.0], 'motorcycle': [2.0, 0.8, 1.3],
        'bicycle': [1.8, 0.5, 1.4], 'person': [0.5, 0.5, 1.7]
    }
    COLORS = {
        'car': [0, 0, 255], 'truck': [255, 0, 200],
        'person': [0, 255, 0], 'bicycle': [0, 255, 128]
    }
    
    scene = {'total_frames': total_frames, 'fps': fps, 'objects': {}, 'frames': {}}
    
    # Static - locked position and orientation
    for track in static_tracks:
        tid = f"S{track.track_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': [128, 128, 128],
            'is_stationary': True
        }
        
        # Fixed orientation for static
        quat = R.from_euler('z', 0.0).as_quat().tolist()
        
        for frame_idx, pos in track.frames.items():
            fk = str(frame_idx)
            if fk not in scene['frames']:
                scene['frames'][fk] = []
            scene['frames'][fk].append({'id': tid, 'pos': pos.tolist(), 'rot': quat, 'conf': 1.0})
    
    # Dynamic - STABLE orientation from overall direction
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
        
        # Compute yaw from SMOOTHED velocity over large window
        if len(positions) >= 3:
            # Use difference over large window
            window = min(21, len(positions) // 2)
            yaws = []
            for i in range(len(positions)):
                start_idx = max(0, i - window)
                end_idx = min(len(positions) - 1, i + window)
                delta = positions[end_idx] - positions[start_idx]
                if np.linalg.norm(delta) > 0.5:
                    yaws.append(np.arctan2(delta[1], delta[0]))
                else:
                    yaws.append(yaws[-1] if yaws else 0.0)
            
            yaws = np.array(yaws)
            yaw_window = min(15, (len(yaws) // 2) * 2 - 1)
            if yaw_window >= 5:
                yaws = savgol_filter(np.unwrap(yaws), yaw_window, 2, mode='nearest')
        else:
            yaws = [0.0] * len(positions)
        
        for i, frame_idx in enumerate(frames_list):
            fk = str(frame_idx)
            if fk not in scene['frames']:
                scene['frames'][fk] = []
            
            pos = track.frames[frame_idx]
            quat = R.from_euler('z', float(yaws[i])).as_quat().tolist()
            scene['frames'][fk].append({'id': tid, 'pos': pos.tolist(), 'rot': quat, 'conf': 0.9})
    
    return scene

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = Path(__file__).parent
    out_dir = base_dir / "outputs" / "pass2_dynamic_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STABLE 4D TRACKER - AGGRESSIVE SMOOTHING")
    print("=" * 70)
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    video_dir = base_dir / "StreetAware-sample"
    
    cap = cv2.VideoCapture(str(video_dir / "s1-left.mp4"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"\nFrames: {total_frames}, FPS: {fps:.1f}")
    
    tracker = StableTracker(cameras)
    tracker.process_video(video_dir, total_frames)
    
    print("\n=== STABILIZING ===")
    static_tracks, dynamic_tracks = tracker.stabilize_and_classify(total_frames)
    print(f"  Static: {len(static_tracks)}, Dynamic: {len(dynamic_tracks)}")
    
    scene = generate_scene(static_tracks, dynamic_tracks, total_frames, fps)
    
    with open(work_dir / "scene_4d.json", 'w') as f:
        json.dump(scene, f, indent=2)
    with open(out_dir / "scene_4d.json", 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"\nSaved. Running visualizer...")

if __name__ == "__main__":
    main()
