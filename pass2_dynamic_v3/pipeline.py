#!/usr/bin/env python3
"""
Pass2 Dynamic V3 - WITH ROAD/CURB MASKS + TRAJECTORY VALIDATION
================================================================
Uses semantic segmentation masks for:
- Static: Objects on curbs (not roads)
- Dynamic: Objects on roads with smooth trajectories
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# =============================================================================
# THRESHOLDS
# =============================================================================

MIN_CONFIDENCE = 0.5
SPATIAL_MATCH_THRESHOLD = 4.0
TEMPORAL_OVERLAP_MIN = 15
STATIC_MOTION_THRESHOLD = 0.03
MIN_STATIC_PRESENCE = 0.4
MIN_DYNAMIC_TRAVEL = 5.0
MAX_CIRCULAR_RATIO = 0.25  # Stricter - must be more linear
MAX_STATIC = 15
MAX_DYNAMIC = 15
CENTER_ZONE_RATIO = 0.7
MAX_3D_DISTANCE = 25.0

# Trajectory smoothness - detect haywire
MAX_ACCELERATION = 3.0  # m/frame² - max sudden direction change
MAX_JUMP = 8.0         # m - max jump between frames

# =============================================================================
# ROAD/CURB MASK LOADER
# =============================================================================

class GroundMaskChecker:
    """Checks if 3D positions are on road or curb using precomputed masks."""
    
    def __init__(self, mask_dir: Path):
        road_path = mask_dir / "road_grid.npy"
        curb_path = mask_dir / "curb_grid.npy"
        info_path = mask_dir / "grid_info.json"
        
        if not road_path.exists():
            print("  WARNING: No ground masks found, using fallback heuristics")
            self.road_grid = None
            self.curb_grid = None
            return
        
        self.road_grid = np.load(road_path)
        self.curb_grid = np.load(curb_path)
        
        with open(info_path) as f:
            self.info = json.load(f)
        
        print(f"  Loaded ground masks: {self.road_grid.shape}")
    
    def is_on_road(self, pos_3d: np.ndarray) -> bool:
        if self.road_grid is None:
            return abs(pos_3d[0]) < 10 and abs(pos_3d[1]) < 10  # Fallback
        
        gx = int((pos_3d[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos_3d[1] - self.info['origin'][1]) / self.info['resolution'])
        
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.road_grid[gy, gx])
        return False
    
    def is_on_curb(self, pos_3d: np.ndarray) -> bool:
        if self.curb_grid is None:
            return abs(pos_3d[0]) > 8 or abs(pos_3d[1]) > 8  # Fallback
        
        gx = int((pos_3d[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos_3d[1] - self.info['origin'][1]) / self.info['resolution'])
        
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.curb_grid[gy, gx])
        return False

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BBox2D:
    x1: float
    y1: float  
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1

@dataclass  
class Track2D:
    track_id: int
    camera: str
    cls: str
    frames: Dict[int, BBox2D] = field(default_factory=dict)
    
    @property
    def frame_range(self) -> Tuple[int, int]:
        if not self.frames:
            return (0, 0)
        return (min(self.frames.keys()), max(self.frames.keys()))
    
    @property
    def length(self) -> int:
        return len(self.frames)
    
    def get_motion_score(self) -> float:
        if len(self.frames) < 5:
            return 0.5
        centers = [self.frames[f].center for f in sorted(self.frames.keys())]
        centers = np.array(centers)
        displacements = np.diff(centers, axis=0)
        motion = np.std(displacements, axis=0).mean()
        avg_width = np.mean([self.frames[f].width for f in self.frames])
        return min(1.0, motion / (avg_width * STATIC_MOTION_THRESHOLD))

@dataclass
class GlobalTrack:
    global_id: int
    cls: str
    camera_tracks: Dict[str, int] = field(default_factory=dict)
    is_static: bool = False
    frames_3d: Dict[int, np.ndarray] = field(default_factory=dict)
    
    def get_trajectory_stats(self) -> Tuple[float, float, float]:
        if len(self.frames_3d) < 2:
            return (0.0, 0.0, 1.0)
        
        frames_list = sorted(self.frames_3d.keys())
        positions = np.array([self.frames_3d[f][:2] for f in frames_list])
        
        diffs = np.diff(positions, axis=0)
        total_travel = np.sum(np.linalg.norm(diffs, axis=1))
        end_to_end = np.linalg.norm(positions[-1] - positions[0])
        circular_ratio = end_to_end / (total_travel + 1e-6)
        
        return (total_travel, end_to_end, circular_ratio)
    
    def fix_haywire_points(self) -> int:
        """Remove or fix points that cause haywire motion. Returns count of fixed."""
        if len(self.frames_3d) < 5:
            return 0
        
        frames_list = sorted(self.frames_3d.keys())
        positions = np.array([self.frames_3d[f][:2] for f in frames_list])
        
        # Compute velocities
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Find outlier speeds (haywire)
        median_speed = np.median(speeds)
        outliers = speeds > max(MAX_JUMP, median_speed * 5)
        
        fixed_count = 0
        
        # For each outlier, interpolate instead
        for i in np.where(outliers)[0]:
            # Frame at i+1 has the issue
            frame_idx = frames_list[i + 1]
            
            # Find good neighbors
            if i > 0 and i + 2 < len(frames_list):
                prev_pos = positions[i]
                next_pos = positions[i + 2]
                # Interpolate
                self.frames_3d[frame_idx][:2] = (prev_pos + next_pos) / 2
                fixed_count += 1
        
        return fixed_count

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
        self.R_w2c = self.R_c2w.T
        self.t_w2c = -self.R_w2c @ self.t_c2w
        
        self.img_w = img_w
        self.img_h = img_h
        
        margin_x = img_w * (1 - CENTER_ZONE_RATIO) / 2
        margin_y = img_h * (1 - CENTER_ZONE_RATIO) / 2
        self.center_bounds = (margin_x, margin_y, img_w - margin_x, img_h - margin_y)
    
    def is_in_center(self, u: float, v: float) -> bool:
        return (self.center_bounds[0] <= u <= self.center_bounds[2] and
                self.center_bounds[1] <= v <= self.center_bounds[3])
    
    def project_to_ground(self, u: float, v: float, z: float = 0.0) -> Optional[np.ndarray]:
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
# STAGE 1: 2D TRACKING
# =============================================================================

def run_2d_tracking(video_dir: Path, total_frames: int) -> Dict[str, List[Track2D]]:
    print("\n=== STAGE 1: 2D TRACKING ===")
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    yolo = YOLO('yolov8x.pt')
    all_tracks = {}
    
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if not vpath.exists():
            continue
        
        print(f"  {cam_id}...")
        cap = cv2.VideoCapture(str(vpath))
        tracks_dict = {}
        
        for frame_idx in tqdm(range(total_frames), desc=f"    {cam_id}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            
            results = yolo.track(frame, conf=MIN_CONFIDENCE, persist=True, 
                                tracker="bytetrack.yaml", verbose=False,
                                classes=list(VALID_CLASSES.keys()))
            
            if results[0].boxes is None or results[0].boxes.id is None:
                continue
            
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if boxes.id is None:
                    continue
                
                track_id = int(boxes.id[i].cpu())
                bbox = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = Track2D(track_id=track_id, camera=cam_id, cls=cls_name)
                
                tracks_dict[track_id].frames[frame_idx] = BBox2D(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]
                )
        
        cap.release()
        valid_tracks = [t for t in tracks_dict.values() if t.length >= 15]
        all_tracks[cam_id] = valid_tracks
        print(f"    {len(valid_tracks)} tracks")
    
    return all_tracks

# =============================================================================
# STAGE 2: CROSS-CAMERA ASSOCIATION
# =============================================================================

def associate_across_cameras(tracks_2d: Dict[str, List[Track2D]], 
                              cameras: dict) -> List[GlobalTrack]:
    print("\n=== STAGE 2: CROSS-CAMERA ASSOCIATION ===")
    
    projectors = {cam_id: CameraProjector(cameras[cam_id]) 
                  for cam_id in tracks_2d.keys() if cam_id in cameras}
    
    all_tracks = []
    for cam_id, cam_tracks in tracks_2d.items():
        all_tracks.extend(cam_tracks)
    
    all_tracks.sort(key=lambda t: t.frame_range[0])
    
    global_tracks = []
    used_tracks = set()
    next_global_id = 1
    
    for track in all_tracks:
        if id(track) in used_tracks:
            continue
        
        global_track = GlobalTrack(global_id=next_global_id, cls=track.cls)
        next_global_id += 1
        
        global_track.camera_tracks[track.camera] = track.track_id
        used_tracks.add(id(track))
        
        for other_track in all_tracks:
            if id(other_track) in used_tracks:
                continue
            if other_track.camera == track.camera:
                continue
            if other_track.cls != track.cls:
                if not ({other_track.cls, track.cls} <= {'car', 'truck'}):
                    continue
            
            r1 = track.frame_range
            r2 = other_track.frame_range
            overlap_start = max(r1[0], r2[0])
            overlap_end = min(r1[1], r2[1])
            
            if overlap_end - overlap_start < TEMPORAL_OVERLAP_MIN:
                continue
            
            proj1 = projectors.get(track.camera)
            proj2 = projectors.get(other_track.camera)
            
            if not proj1 or not proj2:
                continue
            
            distances = []
            for frame in range(overlap_start, overlap_end + 1, 5):
                if frame not in track.frames or frame not in other_track.frames:
                    continue
                
                bc1 = track.frames[frame].bottom_center
                bc2 = other_track.frames[frame].bottom_center
                
                if not proj1.is_in_center(*bc1) or not proj2.is_in_center(*bc2):
                    continue
                
                p1 = proj1.project_to_ground(*bc1)
                p2 = proj2.project_to_ground(*bc2)
                
                if p1 is not None and p2 is not None:
                    dist = np.linalg.norm(p1[:2] - p2[:2])
                    distances.append(dist)
            
            if distances and np.median(distances) < SPATIAL_MATCH_THRESHOLD:
                global_track.camera_tracks[other_track.camera] = other_track.track_id
                used_tracks.add(id(other_track))
        
        global_tracks.append(global_track)
    
    print(f"  Created {len(global_tracks)} global tracks")
    return global_tracks

# =============================================================================
# STAGE 3: ESTIMATE 3D + CLASSIFY WITH ROAD/CURB MASKS
# =============================================================================

def estimate_and_classify(global_tracks: List[GlobalTrack],
                           tracks_2d: Dict[str, List[Track2D]],
                           cameras: dict,
                           total_frames: int,
                           ground_checker: GroundMaskChecker) -> Tuple[List[GlobalTrack], List[GlobalTrack]]:
    print("\n=== STAGE 3: 3D ESTIMATION + CLASSIFICATION ===")
    
    projectors = {cam_id: CameraProjector(cameras[cam_id]) 
                  for cam_id in tracks_2d.keys() if cam_id in cameras}
    
    track_lookup = {}
    for cam_id, cam_tracks in tracks_2d.items():
        for t in cam_tracks:
            track_lookup[(cam_id, t.track_id)] = t
    
    # Estimate 3D positions
    for global_track in tqdm(global_tracks, desc="  3D estimation"):
        frame_observations = defaultdict(list)
        
        for cam_id, track_id in global_track.camera_tracks.items():
            key = (cam_id, track_id)
            if key not in track_lookup:
                continue
            
            track = track_lookup[key]
            proj = projectors.get(cam_id)
            if not proj:
                continue
            
            for frame_idx, bbox in track.frames.items():
                bc = bbox.bottom_center
                if not proj.is_in_center(*bc):
                    continue
                pos_3d = proj.project_to_ground(*bc)
                if pos_3d is not None:
                    frame_observations[frame_idx].append(pos_3d)
        
        for frame_idx, positions in frame_observations.items():
            global_track.frames_3d[frame_idx] = np.median(positions, axis=0)
    
    # Classify with road/curb masks
    static_tracks = []
    dynamic_tracks = []
    
    for global_track in global_tracks:
        if not global_track.frames_3d or len(global_track.frames_3d) < 10:
            continue
        
        # Compute motion score
        motion_scores = []
        for cam_id, track_id in global_track.camera_tracks.items():
            key = (cam_id, track_id)
            if key in track_lookup:
                motion_scores.append(track_lookup[key].get_motion_score())
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0.5
        presence = len(global_track.frames_3d) / total_frames
        total_travel, end_to_end, circular_ratio = global_track.get_trajectory_stats()
        
        # Get median position
        all_pos = list(global_track.frames_3d.values())
        median_pos = np.median(all_pos, axis=0)
        
        # USE MASKS for classification
        is_on_road = ground_checker.is_on_road(median_pos)
        is_on_curb = ground_checker.is_on_curb(median_pos)
        
        # STATIC: Low motion + on curb (or not on road)
        if avg_motion < 0.4 and presence >= MIN_STATIC_PRESENCE:
            if is_on_curb or (not is_on_road):
                global_track.is_static = True
                
                for f in range(total_frames):
                    global_track.frames_3d[f] = median_pos.copy()
                static_tracks.append(global_track)
                continue
        
        # DYNAMIC: Significant travel + on road + linear trajectory
        if total_travel >= MIN_DYNAMIC_TRAVEL:
            if circular_ratio >= MAX_CIRCULAR_RATIO:
                # FIX HAYWIRE before accepting
                fixed = global_track.fix_haywire_points()
                if fixed > 0:
                    print(f"    Fixed {fixed} haywire points in D{global_track.global_id}")
                
                global_track.is_static = False
                
                # Fill gaps
                frames_list = sorted(global_track.frames_3d.keys())
                for i in range(len(frames_list) - 1):
                    gap = frames_list[i+1] - frames_list[i]
                    if gap > 1 and gap <= 10:
                        p1 = global_track.frames_3d[frames_list[i]]
                        p2 = global_track.frames_3d[frames_list[i+1]]
                        for f in range(frames_list[i] + 1, frames_list[i+1]):
                            alpha = (f - frames_list[i]) / gap
                            global_track.frames_3d[f] = p1 + alpha * (p2 - p1)
                
                # Smooth with STRONGER filter
                frames_list = sorted(global_track.frames_3d.keys())
                if len(frames_list) >= 5:
                    for dim in range(3):
                        vals = np.array([global_track.frames_3d[f][dim] for f in frames_list])
                        smoothed = uniform_filter1d(vals, size=11, mode='nearest')  # Stronger
                        for i, f in enumerate(frames_list):
                            global_track.frames_3d[f][dim] = smoothed[i]
                
                dynamic_tracks.append(global_track)
    
    # Limit counts
    static_tracks.sort(key=lambda t: len(t.frames_3d), reverse=True)
    dynamic_tracks.sort(key=lambda t: t.get_trajectory_stats()[0], reverse=True)
    
    static_tracks = static_tracks[:MAX_STATIC]
    dynamic_tracks = dynamic_tracks[:MAX_DYNAMIC]
    
    print(f"  Static: {len(static_tracks)}, Dynamic: {len(dynamic_tracks)}")
    
    return static_tracks, dynamic_tracks

# =============================================================================
# OUTPUT
# =============================================================================

def generate_scene(static_tracks: List[GlobalTrack], dynamic_tracks: List[GlobalTrack],
                   total_frames: int, fps: float) -> dict:
    print("\n=== GENERATING OUTPUT ===")
    
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
        tid = f"S{track.global_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': [128, 128, 128],
            'is_stationary': True
        }
        
        for frame_idx, pos in track.frames_3d.items():
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
        tid = f"D{track.global_id}"
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': COLORS.get(track.cls, [255, 255, 255]),
            'is_stationary': False
        }
        
        frames_list = sorted(track.frames_3d.keys())
        positions = np.array([track.frames_3d[f][:2] for f in frames_list])
        
        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([velocities, velocities[-1] if len(velocities) > 0 else [[1, 0]]])
        yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
        yaws = uniform_filter1d(np.unwrap(yaws), size=11, mode='nearest')  # Stronger
        
        for i, frame_idx in enumerate(frames_list):
            frame_key = str(frame_idx)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            pos = track.frames_3d[frame_idx]
            quat = R.from_euler('z', float(yaws[i])).as_quat().tolist()
            
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': pos.tolist(),
                'rot': quat,
                'conf': 0.9
            })
    
    print(f"  Total: {len(static_tracks) + len(dynamic_tracks)} objects")
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
    print("PASS2 DYNAMIC V3 - WITH ROAD/CURB MASKS + HAYWIRE FIX")
    print("=" * 70)
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    # Load ground masks
    mask_dir = base_dir / "outputs" / "pass1_static" / "ground_masks"
    ground_checker = GroundMaskChecker(mask_dir)
    
    video_dir = base_dir / "StreetAware-sample"
    
    test_cap = cv2.VideoCapture(str(video_dir / "s1-left.mp4"))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS)
    test_cap.release()
    
    print(f"\nTotal frames: {total_frames}, FPS: {fps:.1f}")
    
    tracks_2d = run_2d_tracking(video_dir, total_frames)
    global_tracks = associate_across_cameras(tracks_2d, cameras)
    static_tracks, dynamic_tracks = estimate_and_classify(
        global_tracks, tracks_2d, cameras, total_frames, ground_checker
    )
    
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
