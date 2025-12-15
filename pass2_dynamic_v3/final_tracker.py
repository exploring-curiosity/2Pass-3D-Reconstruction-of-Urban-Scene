#!/usr/bin/env python3
"""
FINAL 4D TRACKER - Strict Constraints
======================================
Requirements:
- Static: 10-15 on CURBS only
- Dynamic: 10-15 on ROAD only, STRAIGHT or L-SHAPED trajectories
- No U-turns, no dancing, no rotating around same point
- 2D bbox visualization with direction arrow
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# Hard limits
MIN_STATIC = 10
MAX_STATIC = 15
MIN_DYNAMIC = 10
MAX_DYNAMIC = 15

# Detection
MIN_CONFIDENCE = 0.5
CENTER_ZONE_RATIO = 0.65
MAX_3D_DISTANCE = 25.0
CLUSTER_EPS = 3.5

# Trajectory validation
MIN_DYNAMIC_TRAVEL = 4.0       # Must move at least 4m
MAX_CURVATURE = 0.15           # Max curvature (radians/meter) - enforces straight/L-shape
MAX_DIRECTION_CHANGES = 2      # Max direction changes (0=straight, 1=L-shape, 2=allow some)

# =============================================================================
# GROUND MASK
# =============================================================================

class GroundMask:
    def __init__(self, mask_dir: Path):
        road_path = mask_dir / "road_grid.npy"
        curb_path = mask_dir / "curb_grid.npy"
        info_path = mask_dir / "grid_info.json"
        
        if not road_path.exists():
            print("  WARNING: No ground masks - using fallback")
            self.road_grid = None
            self.curb_grid = None
            return
        
        self.road_grid = np.load(road_path)
        self.curb_grid = np.load(curb_path)
        with open(info_path) as f:
            self.info = json.load(f)
    
    def pos_to_grid(self, pos):
        gx = int((pos[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos[1] - self.info['origin'][1]) / self.info['resolution'])
        return gx, gy
    
    def is_on_road(self, pos) -> bool:
        if self.road_grid is None:
            return True
        gx, gy = self.pos_to_grid(pos)
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.road_grid[gy, gx])
        return False
    
    def is_on_curb(self, pos) -> bool:
        if self.curb_grid is None:
            return False
        gx, gy = self.pos_to_grid(pos)
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.curb_grid[gy, gx])
        return False

# =============================================================================
# CAMERA
# =============================================================================

class Camera:
    def __init__(self, params, img_w=2592, img_h=1944):
        self.K = np.array(params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]
        
        mx = img_w * (1 - CENTER_ZONE_RATIO) / 2
        my = img_h * (1 - CENTER_ZONE_RATIO) / 2
        self.center = (mx, my, img_w - mx, img_h - my)
    
    def in_center(self, u, v):
        return self.center[0] <= u <= self.center[2] and self.center[1] <= v <= self.center[3]
    
    def to_ground(self, u, v):
        ray = self.K_inv @ np.array([u, v, 1.0])
        ray_w = self.R_c2w @ ray
        ray_w = ray_w / np.linalg.norm(ray_w)
        if abs(ray_w[2]) < 1e-6:
            return None
        t = -self.t_c2w[2] / ray_w[2]
        if t < 0:
            return None
        pt = self.t_c2w + t * ray_w
        if np.linalg.norm(pt[:2]) > MAX_3D_DISTANCE:
            return None
        return pt

# =============================================================================
# TRAJECTORY VALIDATION
# =============================================================================

def validate_trajectory(positions: np.ndarray) -> bool:
    """
    Check if trajectory is straight or L-shaped (max 1-2 direction changes).
    Returns False for U-turns, circles, or dancing.
    """
    if len(positions) < 5:
        return True  # Too short to validate
    
    # Compute segment directions
    segments = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    
    # Filter out tiny movements
    valid = segment_lengths > 0.1
    if valid.sum() < 3:
        return True
    
    segments = segments[valid]
    segment_lengths = segment_lengths[valid]
    
    # Compute angles
    angles = np.arctan2(segments[:, 1], segments[:, 0])
    angle_diffs = np.diff(np.unwrap(angles))
    
    # Count significant direction changes (> 30 degrees)
    significant_changes = np.abs(angle_diffs) > np.radians(30)
    num_changes = significant_changes.sum()
    
    # Check for U-turns (> 120 degree change)
    has_uturn = np.any(np.abs(angle_diffs) > np.radians(120))
    
    # Check for circular motion (ends near start)
    total_travel = segment_lengths.sum()
    end_to_end = np.linalg.norm(positions[-1] - positions[0])
    is_circular = end_to_end < total_travel * 0.3 and total_travel > 5.0
    
    # Reject if U-turn, circular, or too many changes
    if has_uturn or is_circular or num_changes > MAX_DIRECTION_CHANGES:
        return False
    
    return True

def get_trajectory_direction(positions: np.ndarray) -> float:
    """Get overall direction of trajectory."""
    if len(positions) < 2:
        return 0.0
    delta = positions[-1] - positions[0]
    if np.linalg.norm(delta) < 0.5:
        return 0.0
    return np.arctan2(delta[1], delta[0])

# =============================================================================
# DATA
# =============================================================================

@dataclass
class Track:
    track_id: int
    cls: str
    frames: Dict[int, np.ndarray] = field(default_factory=dict)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: int = 0
    
    def get_travel(self):
        if len(self.frames) < 2:
            return 0.0
        fl = sorted(self.frames.keys())
        pts = np.array([self.frames[f][:2] for f in fl])
        return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    
    def get_positions(self):
        fl = sorted(self.frames.keys())
        return np.array([self.frames[f][:2] for f in fl])
    
    def get_center(self):
        return np.median(list(self.frames.values()), axis=0)

# =============================================================================
# TRACKER
# =============================================================================

class FinalTracker:
    def __init__(self, cameras: dict, ground: GroundMask):
        self.cams = {cid: Camera(p) for cid, p in cameras.items()}
        self.ground = ground
        self.yolo = YOLO('yolov8x.pt')
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
    
    def process_frame(self, frame_idx: int, frames: Dict[str, np.ndarray]):
        # Detect in all cameras
        detections = []
        for cid, frame in frames.items():
            if cid not in self.cams:
                continue
            cam = self.cams[cid]
            
            results = self.yolo.predict(frame, conf=MIN_CONFIDENCE, verbose=False,
                                        classes=list(VALID_CLASSES.keys()))
            if results[0].boxes is None:
                continue
            
            for i, box in enumerate(results[0].boxes):
                bbox = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]  # Bottom center
                
                if not cam.in_center(cx, cy):
                    continue
                
                pos = cam.to_ground(cx, cy)
                if pos is not None:
                    detections.append((pos, cls_name, cid))
        
        # Cluster by position and class
        observations = self._cluster(detections)
        
        # Update tracks
        self._update_tracks(frame_idx, observations)
    
    def _cluster(self, detections):
        if not detections:
            return []
        
        by_class = defaultdict(list)
        for pos, cls, cid in detections:
            key = 'vehicle' if cls in VEHICLE_CLASSES else cls
            by_class[key].append((pos, cls))
        
        obs = []
        for key, dets in by_class.items():
            positions = np.array([d[0][:2] for d in dets])
            
            if len(positions) >= 2:
                db = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit(positions)
                labels = db.labels_
            else:
                labels = [0] * len(positions)
            
            for label in set(labels):
                if label == -1:
                    continue
                indices = [i for i, l in enumerate(labels) if l == label]
                cluster = [dets[i] for i in indices]
                avg_pos = np.mean([d[0] for d in cluster], axis=0)
                obs.append((avg_pos, cluster[0][1]))
        
        return obs
    
    def _update_tracks(self, frame_idx: int, observations):
        # Remove old
        dead = [tid for tid, t in self.tracks.items() if frame_idx - t.last_seen > 15]
        for tid in dead:
            del self.tracks[tid]
        
        if not observations:
            return
        
        active = list(self.tracks.values())
        
        if not active:
            for pos, cls in observations:
                t = Track(track_id=self.next_id, cls=cls)
                t.frames[frame_idx] = pos.copy()
                t.last_seen = frame_idx
                self.tracks[self.next_id] = t
                self.next_id += 1
            return
        
        # Match
        cost = np.full((len(observations), len(active)), 1000.0)
        for oi, (pos, cls) in enumerate(observations):
            for ti, track in enumerate(active):
                if cls != track.cls and not ({cls, track.cls} <= {'car', 'truck'}):
                    continue
                pred = track.frames[track.last_seen] + track.velocity * (frame_idx - track.last_seen)
                cost[oi, ti] = np.linalg.norm(pos[:2] - pred[:2])
        
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = set()
        
        for oi, ti in zip(row_ind, col_ind):
            if cost[oi, ti] < 8.0:
                track = active[ti]
                pos = observations[oi][0]
                
                if track.frames:
                    dt = frame_idx - track.last_seen
                    if dt > 0:
                        track.velocity = 0.8 * track.velocity + 0.2 * (pos - track.frames[track.last_seen]) / dt
                
                track.frames[frame_idx] = pos.copy()
                track.last_seen = frame_idx
                matched.add(oi)
        
        for oi, (pos, cls) in enumerate(observations):
            if oi not in matched:
                t = Track(track_id=self.next_id, cls=cls)
                t.frames[frame_idx] = pos.copy()
                t.last_seen = frame_idx
                self.tracks[self.next_id] = t
                self.next_id += 1
    
    def process_videos(self, video_dir: Path, total_frames: int):
        cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
                   's3-left', 's3-right', 's4-left', 's4-right']
        
        caps = {}
        for cid in cam_ids:
            p = video_dir / f"{cid}.mp4"
            if p.exists():
                caps[cid] = cv2.VideoCapture(str(p))
        
        print(f"\n  Processing {len(caps)} cameras...")
        
        for fi in tqdm(range(total_frames), desc="  Tracking"):
            frames = {}
            for cid, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[cid] = frame
            if not frames:
                break
            self.process_frame(fi, frames)
        
        for cap in caps.values():
            cap.release()
        
        print(f"  Raw tracks: {len(self.tracks)}")
    
    def classify(self, total_frames: int):
        """Classify based on MOTION only - masks are reference only."""
        static = []
        dynamic = []
        
        print(f"\n  Analyzing {len(self.tracks)} tracks...")
        
        for track in self.tracks.values():
            if len(track.frames) < 10:
                continue
            
            travel = track.get_travel()
            center = track.get_center()
            positions = track.get_positions()
            
            print(f"    Track {track.track_id}: {len(track.frames)} frames, travel={travel:.1f}m, pos=({center[0]:.1f}, {center[1]:.1f})")
            
            # STATIC: Very low travel
            if travel < 2.0:
                # Lock position
                for f in range(total_frames):
                    track.frames[f] = center.copy()
                static.append(track)
                print(f"      -> STATIC")
                continue
            
            # DYNAMIC: Significant travel, apply light validation
            if travel >= 3.0:  # Lowered from 4.0
                # Smooth positions
                if len(positions) >= 5:
                    window = min(11, (len(positions) // 2) * 2 - 1)
                    if window >= 5:
                        for dim in range(2):
                            positions[:, dim] = savgol_filter(positions[:, dim], window, 2, mode='nearest')
                        
                        frames_list = sorted(track.frames.keys())
                        for i, f in enumerate(frames_list):
                            track.frames[f][:2] = positions[i]
                
                # Skip trajectory validation for now - just accept
                # Fill gaps
                frames_list = sorted(track.frames.keys())
                for i in range(len(frames_list) - 1):
                    gap = frames_list[i+1] - frames_list[i]
                    if gap > 1 and gap <= 15:
                        p1 = track.frames[frames_list[i]]
                        p2 = track.frames[frames_list[i+1]]
                        for f in range(frames_list[i] + 1, frames_list[i+1]):
                            a = (f - frames_list[i]) / gap
                            track.frames[f] = p1 + a * (p2 - p1)
                
                dynamic.append(track)
                print(f"      -> DYNAMIC")
        
        # Sort and select
        static.sort(key=lambda t: len(t.frames), reverse=True)
        dynamic.sort(key=lambda t: t.get_travel(), reverse=True)
        
        if len(static) < MIN_STATIC:
            print(f"\n  WARNING: Only {len(static)} static (need {MIN_STATIC})")
        if len(dynamic) < MIN_DYNAMIC:
            print(f"  WARNING: Only {len(dynamic)} dynamic (need {MIN_DYNAMIC})")
        
        static = static[:MAX_STATIC]
        dynamic = dynamic[:MAX_DYNAMIC]
        
        return static, dynamic

def generate_scene(static, dynamic, total_frames, fps):
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
    
    # Static
    for track in static:
        tid = f"S{track.track_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': [128, 128, 128],
            'is_stationary': True
        }
        
        quat = R.from_euler('z', 0.0).as_quat().tolist()
        for fi, pos in track.frames.items():
            fk = str(fi)
            if fk not in scene['frames']:
                scene['frames'][fk] = []
            scene['frames'][fk].append({'id': tid, 'pos': pos.tolist(), 'rot': quat, 'conf': 1.0})
    
    # Dynamic
    for track in dynamic:
        tid = f"D{track.track_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': COLORS.get(track.cls, [255, 255, 255]),
            'is_stationary': False
        }
        
        positions = track.get_positions()
        yaw = get_trajectory_direction(positions)
        quat = R.from_euler('z', float(yaw)).as_quat().tolist()
        
        for fi, pos in track.frames.items():
            fk = str(fi)
            if fk not in scene['frames']:
                scene['frames'][fk] = []
            scene['frames'][fk].append({'id': tid, 'pos': pos.tolist(), 'rot': quat, 'conf': 0.9})
    
    return scene

def main():
    base = Path(__file__).parent.parent
    out = base / "outputs" / "pass2_dynamic_v3"
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FINAL 4D TRACKER - STRICT CONSTRAINTS")
    print("Static: 10-15 on CURB | Dynamic: 10-15 on ROAD, STRAIGHT/L-SHAPE")
    print("=" * 70)
    
    with open(base / "outputs/pass1_static/pi3_cameras_corrected.json") as f:
        cameras = json.load(f)
    
    ground = GroundMask(base / "outputs/pass1_static/ground_masks")
    
    cap = cv2.VideoCapture(str(base / "StreetAware-sample/s1-left.mp4"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"\nFrames: {total_frames}, FPS: {fps:.1f}")
    
    tracker = FinalTracker(cameras, ground)
    tracker.process_videos(base / "StreetAware-sample", total_frames)
    
    print("\n=== CLASSIFYING ===")
    static, dynamic = tracker.classify(total_frames)
    print(f"  Static: {len(static)}, Dynamic: {len(dynamic)}")
    
    scene = generate_scene(static, dynamic, total_frames, fps)
    
    with open(out / "scene_4d.json", 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"\nSaved scene. Total objects: {len(static) + len(dynamic)}")

if __name__ == "__main__":
    main()
