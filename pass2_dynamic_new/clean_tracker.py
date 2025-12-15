#!/usr/bin/env python3
"""
Clean 4D Tracker with STRICT Validation
========================================
After debugging, the key issues are:
1. Projections from edge pixels create extreme 3D positions
2. Need STRICT bounds on valid detection area
3. Use image CENTER zone only for reliable projections
4. Require temporal consistency for all tracks
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
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# STRICT settings
MIN_CONF = 0.55                    # Higher confidence
IMAGE_CENTER_RATIO = 0.6           # Use only central 60% of image
MAX_GROUND_DIST = 25.0             # Max distance from origin (meters)
MIN_STATIC_FRAMES_RATIO = 0.7      # 70% of frames for static
MIN_DYNAMIC_LEN = 30               # Minimum frames for dynamic track
MAX_GAP = 8                        # Max interpolation gap
MIN_CAMERAS = 2                    # Require 2+ camera views
CLUSTER_DIST = 3.0                 # meters for clustering

class CameraProjector:
    def __init__(self, cam_params, img_w=2592, img_h=1944):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]
        
        self.R_w2c = self.R_c2w.T
        self.t_w2c = -self.R_w2c @ self.t_c2w
        
        self.img_w = img_w
        self.img_h = img_h
        
        # Compute valid image region (central 60%)
        margin_x = img_w * (1 - IMAGE_CENTER_RATIO) / 2
        margin_y = img_h * (1 - IMAGE_CENTER_RATIO) / 2
        self.valid_x_min = margin_x
        self.valid_x_max = img_w - margin_x
        self.valid_y_min = margin_y
        self.valid_y_max = img_h - margin_y
        
    def is_in_center(self, u: float, v: float) -> bool:
        return (self.valid_x_min <= u <= self.valid_x_max and
                self.valid_y_min <= v <= self.valid_y_max)
    
    def pixel_to_ground(self, u: float, v: float, z: float = 0.0) -> Optional[np.ndarray]:
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        ray_world = self.R_c2w @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        if abs(ray_world[2]) < 1e-6:
            return None
        
        t = (z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
        
        point = self.t_c2w + t * ray_world
        
        # Strict bounds check
        if abs(point[0]) > MAX_GROUND_DIST or abs(point[1]) > MAX_GROUND_DIST:
            return None
        
        return point
    
    def world_to_pixel(self, point: np.ndarray) -> Optional[np.ndarray]:
        p_cam = self.R_w2c @ point + self.t_w2c
        if p_cam[2] <= 0:
            return None
        p_img = self.K @ p_cam
        return (p_img[:2] / p_cam[2]).astype(int)

@dataclass
class Detection:
    frame: int
    camera: str
    cls: str
    conf: float
    pos_3d: np.ndarray

@dataclass
class Observation:
    frame: int
    pos_3d: np.ndarray
    cls: str
    cameras: List[str]
    conf: float

@dataclass
class Track:
    track_id: int
    cls: str
    is_static: bool
    frames: Dict[int, np.ndarray] = field(default_factory=dict)

def detect_all_frames(video_dir: Path, cameras: dict, yolo: YOLO, 
                      total_frames: int) -> Dict[int, List[Observation]]:
    """Detect with STRICT validation: center zone only, multi-camera required."""
    
    print("\n=== DETECTING WITH STRICT VALIDATION ===")
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    projectors = {cam_id: CameraProjector(cameras[cam_id]) 
                  for cam_id in cam_ids if cam_id in cameras}
    
    caps = {}
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            caps[cam_id] = cv2.VideoCapture(str(vpath))
    
    all_observations = {}
    total_raw = 0
    total_filtered = 0
    
    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        frame_dets = []
        
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            
            results = yolo.predict(frame, conf=MIN_CONF, iou=0.5, verbose=False,
                                  classes=list(VALID_CLASSES.keys()))
            
            if results[0].boxes is None:
                continue
            
            boxes = results[0].boxes
            proj = projectors.get(cam_id)
            if not proj:
                continue
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                total_raw += 1
                
                # Bottom center
                cx = (bbox[0] + bbox[2]) / 2
                cy = bbox[3]
                
                # STRICT: Must be in center zone
                if not proj.is_in_center(cx, cy):
                    continue
                
                # Project to ground
                pos_3d = proj.pixel_to_ground(cx, cy, 0.0)
                if pos_3d is None:
                    continue
                
                total_filtered += 1
                frame_dets.append(Detection(
                    frame=frame_idx,
                    camera=cam_id,
                    cls=cls_name,
                    conf=conf,
                    pos_3d=pos_3d
                ))
        
        # Cluster detections from multiple cameras
        if len(frame_dets) >= MIN_CAMERAS:
            observations = cluster_detections(frame_dets)
            if observations:
                all_observations[frame_idx] = observations
    
    for cap in caps.values():
        cap.release()
    
    print(f"  Raw detections: {total_raw}")
    print(f"  After center filter: {total_filtered}")
    print(f"  Multi-camera observations: {sum(len(o) for o in all_observations.values())}")
    
    return all_observations

def cluster_detections(dets: List[Detection]) -> List[Observation]:
    """Cluster detections, require 2+ cameras."""
    from sklearn.cluster import DBSCAN
    
    if len(dets) < MIN_CAMERAS:
        return []
    
    # Group by class
    by_class = defaultdict(list)
    for d in dets:
        by_class[d.cls].append(d)
    
    observations = []
    
    for cls, class_dets in by_class.items():
        if len(class_dets) < MIN_CAMERAS:
            continue
        
        points = np.array([d.pos_3d[:2] for d in class_dets])
        
        db = DBSCAN(eps=CLUSTER_DIST, min_samples=MIN_CAMERAS).fit(points)
        
        for label in set(db.labels_):
            if label == -1:
                continue
            
            indices = np.where(db.labels_ == label)[0]
            cluster_dets = [class_dets[i] for i in indices]
            
            cameras = set(d.camera for d in cluster_dets)
            if len(cameras) < MIN_CAMERAS:
                continue
            
            # Average position
            avg_pos = np.mean([d.pos_3d for d in cluster_dets], axis=0)
            avg_conf = np.mean([d.conf for d in cluster_dets])
            
            observations.append(Observation(
                frame=cluster_dets[0].frame,
                pos_3d=avg_pos,
                cls=cls,
                cameras=list(cameras),
                conf=avg_conf
            ))
    
    return observations

def identify_static_dynamic(observations: Dict[int, List[Observation]], 
                            total_frames: int) -> Tuple[List[Track], List[Track]]:
    """Separate static and dynamic objects."""
    
    print("\n=== SEPARATING STATIC AND DYNAMIC ===")
    
    # Group by spatial location
    position_groups = []
    
    for frame_idx, obs_list in sorted(observations.items()):
        for obs in obs_list:
            matched = False
            for group in position_groups:
                if obs.cls != group['cls']:
                    if not ({obs.cls, group['cls']} <= {'car', 'truck'}):
                        continue
                
                centroid = np.mean(group['positions'], axis=0)
                dist = np.linalg.norm(obs.pos_3d[:2] - centroid[:2])
                
                if dist < 2.5:
                    group['positions'].append(obs.pos_3d)
                    group['frames'].append(frame_idx)
                    matched = True
                    break
            
            if not matched:
                position_groups.append({
                    'positions': [obs.pos_3d],
                    'frames': [frame_idx],
                    'cls': obs.cls
                })
    
    # Classify as static or dynamic
    min_static_frames = int(total_frames * MIN_STATIC_FRAMES_RATIO)
    
    static_tracks = []
    dynamic_groups = []
    next_static_id = 1
    
    for group in position_groups:
        unique_frames = len(set(group['frames']))
        positions = np.array(group['positions'])
        variance = np.var(positions[:, :2], axis=0).sum()
        
        # Static: present in many frames with low variance
        if unique_frames >= min_static_frames and variance < 2.0:
            median_pos = np.median(positions, axis=0)
            
            track = Track(
                track_id=next_static_id,
                cls=group['cls'],
                is_static=True
            )
            
            for f in range(total_frames):
                track.frames[f] = median_pos.copy()
            
            static_tracks.append(track)
            next_static_id += 1
        else:
            dynamic_groups.append(group)
    
    print(f"  Static objects: {len(static_tracks)}")
    
    # Track dynamic objects
    static_positions = [np.median(np.array(t.frames[0])[:2].reshape(1, 2), axis=0) 
                        for t in static_tracks]
    
    # Filter out observations near static objects
    filtered_obs = {}
    for frame_idx, obs_list in observations.items():
        filtered = []
        for obs in obs_list:
            near_static = False
            for sp in static_positions:
                if np.linalg.norm(obs.pos_3d[:2] - sp) < 3.0:
                    near_static = True
                    break
            if not near_static:
                filtered.append(obs)
        if filtered:
            filtered_obs[frame_idx] = filtered
    
    # Track dynamic
    dynamic_tracks = []
    next_dyn_id = 100
    active = []
    
    for frame_idx in range(total_frames):
        current = filtered_obs.get(frame_idx, [])
        
        still_active = []
        for track in active:
            last_frame = max(track.frames.keys())
            if frame_idx - last_frame <= MAX_GAP:
                still_active.append(track)
            else:
                dynamic_tracks.append(track)
        active = still_active
        
        if not current:
            continue
        
        if active:
            cost = np.zeros((len(current), len(active)))
            
            for oi, obs in enumerate(current):
                for ti, track in enumerate(active):
                    if track.cls != obs.cls:
                        if not ({track.cls, obs.cls} <= {'car', 'truck'}):
                            cost[oi, ti] = 1000
                            continue
                    
                    last_frame = max(track.frames.keys())
                    last_pos = track.frames[last_frame]
                    dist = np.linalg.norm(obs.pos_3d[:2] - last_pos[:2])
                    cost[oi, ti] = dist
            
            rows, cols = linear_sum_assignment(cost)
            matched = set()
            
            for oi, ti in zip(rows, cols):
                if cost[oi, ti] < 5.0:
                    active[ti].frames[frame_idx] = current[oi].pos_3d.copy()
                    matched.add(oi)
            
            for oi, obs in enumerate(current):
                if oi not in matched:
                    track = Track(track_id=next_dyn_id, cls=obs.cls, is_static=False)
                    next_dyn_id += 1
                    track.frames[frame_idx] = obs.pos_3d.copy()
                    active.append(track)
        else:
            for obs in current:
                track = Track(track_id=next_dyn_id, cls=obs.cls, is_static=False)
                next_dyn_id += 1
                track.frames[frame_idx] = obs.pos_3d.copy()
                active.append(track)
    
    dynamic_tracks.extend(active)
    
    # Filter short tracks
    dynamic_tracks = [t for t in dynamic_tracks if len(t.frames) >= MIN_DYNAMIC_LEN]
    
    print(f"  Dynamic tracks: {len(dynamic_tracks)}")
    
    return static_tracks, dynamic_tracks

def smooth_tracks(tracks: List[Track]) -> List[Track]:
    """Smooth track positions and fill gaps."""
    for track in tracks:
        if track.is_static:
            continue
        
        frames_list = sorted(track.frames.keys())
        if len(frames_list) < 3:
            continue
        
        # Fill gaps
        new_frames = dict(track.frames)
        for i in range(len(frames_list) - 1):
            gap = frames_list[i+1] - frames_list[i]
            if gap > 1 and gap <= MAX_GAP:
                p1 = track.frames[frames_list[i]]
                p2 = track.frames[frames_list[i+1]]
                for f in range(frames_list[i] + 1, frames_list[i+1]):
                    alpha = (f - frames_list[i]) / gap
                    new_frames[f] = p1 + alpha * (p2 - p1)
        track.frames = new_frames
        
        # Smooth
        frames_list = sorted(track.frames.keys())
        if len(frames_list) >= 5:
            for dim in range(3):
                vals = np.array([track.frames[f][dim] for f in frames_list])
                smoothed = uniform_filter1d(vals, size=5, mode='nearest')
                for i, f in enumerate(frames_list):
                    track.frames[f][dim] = smoothed[i]
    
    return tracks

def generate_scene(static_tracks: List[Track], dynamic_tracks: List[Track],
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
        'bus': [0, 200, 255],
        'motorcycle': [255, 128, 0],
        'bicycle': [0, 255, 128],
        'person': [0, 255, 0]
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
        yaws = uniform_filter1d(np.unwrap(yaws), size=5, mode='nearest')
        
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
                'conf': 0.8
            })
    
    return scene

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    
    print("=" * 70)
    print("CLEAN 4D TRACKER - STRICT VALIDATION")
    print("- Center zone only (60% of image)")
    print("- Require 2+ cameras")
    print("- Max 25m from origin")
    print("=" * 70)
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    video_dir = base_dir / "StreetAware-sample"
    
    test_cap = cv2.VideoCapture(str(video_dir / "s1-left.mp4"))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS)
    test_cap.release()
    
    print(f"\nTotal frames: {total_frames}, FPS: {fps:.1f}")
    
    yolo = YOLO('yolov8x.pt')
    
    observations = detect_all_frames(video_dir, cameras, yolo, total_frames)
    
    static_tracks, dynamic_tracks = identify_static_dynamic(observations, total_frames)
    
    print("\n=== SMOOTHING ===")
    dynamic_tracks = smooth_tracks(dynamic_tracks)
    
    print("\n=== GENERATING SCENE ===")
    scene = generate_scene(static_tracks, dynamic_tracks, total_frames, fps)
    
    print(f"  Static: {len(static_tracks)}, Dynamic: {len(dynamic_tracks)}")
    print(f"  Total: {len(static_tracks) + len(dynamic_tracks)} objects")
    
    out_path = work_dir / "scene_4d.json"
    with open(out_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"\nSaved to {out_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
