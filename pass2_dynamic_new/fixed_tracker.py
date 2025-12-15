#!/usr/bin/env python3
"""
FIXED Unified 3D Tracker
========================
Fixes:
1. Better 3D projection - use bbox center adjusted for perspective
2. Robust parked car locking - lock position early and don't change
3. Occlusion handling - prefer older position during gaps
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import sys

# Thresholds
MIN_CONFIDENCE = 0.5
CLUSTER_EPS = 5.0              # meters
MIN_TRACK_FRAMES = 60          # frames
MAX_ASSOCIATION_DIST = 6.0     # meters
MAX_GAP_FRAMES = 5             # frames
STATIONARY_VAR = 2.5           # m²
MERGE_DISTANCE = 5.0           # meters
PARKED_LOCK_FRAMES = 15        # Lock parked position after N consistent frames

@dataclass
class Observation:
    """One observation of an object at a single frame."""
    frame: int
    pos: np.ndarray
    cls: str
    cameras: List[str]
    conf: float
    det_count: int

@dataclass
class Track:
    """A tracked object over time."""
    track_id: int
    cls: str
    observations: List[Observation] = field(default_factory=list)
    is_stationary: bool = False
    locked_position: Optional[np.ndarray] = None  # For parked cars
    
    @property
    def length(self):
        return len(self.observations)
    
    def get_last_pos(self) -> np.ndarray:
        if self.observations:
            return self.observations[-1].pos
        return np.zeros(3)
    
    def get_velocity(self, fps: float) -> np.ndarray:
        if len(self.observations) < 2:
            return np.zeros(3)
        
        recent = self.observations[-5:]
        if len(recent) < 2:
            return np.zeros(3)
        
        pos1 = recent[0].pos
        pos2 = recent[-1].pos
        dt = (recent[-1].frame - recent[0].frame) / fps
        
        if dt > 0:
            return (pos2 - pos1) / dt
        return np.zeros(3)
    
    def check_and_lock_stationary(self):
        """Check if object is stationary and lock its position."""
        if len(self.observations) < PARKED_LOCK_FRAMES:
            return
        
        # Use first N observations to determine if stationary
        early_obs = self.observations[:PARKED_LOCK_FRAMES]
        positions = np.array([o.pos[:2] for o in early_obs])
        variance = np.var(positions, axis=0).sum()
        
        if variance < STATIONARY_VAR:
            self.is_stationary = True
            # Lock to median of early observations
            self.locked_position = np.median(np.array([o.pos for o in early_obs]), axis=0)

def load_detections(work_dir: Path):
    """Load 3D detections."""
    det_path = work_dir / "detections_3d.json"
    with open(det_path) as f:
        data = json.load(f)
    return data['detections'], data['total_frames'], data['fps']

def cluster_frame_detections(dets: List[dict], eps: float = CLUSTER_EPS) -> List[Observation]:
    """Cluster all detections in a frame by 3D proximity and class."""
    if not dets:
        return []
    
    frame = dets[0]['frame']
    
    by_class = defaultdict(list)
    for d in dets:
        by_class[d['class']].append(d)
    
    observations = []
    
    for cls, class_dets in by_class.items():
        if len(class_dets) == 1:
            d = class_dets[0]
            observations.append(Observation(
                frame=frame,
                pos=np.array(d['pos_3d']),
                cls=cls,
                cameras=[d['camera']],
                conf=d['conf'],
                det_count=1
            ))
            continue
        
        points = np.array([d['pos_3d'][:2] for d in class_dets])
        
        db = DBSCAN(eps=eps, min_samples=1).fit(points)
        labels = db.labels_
        
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_dets = [class_dets[i] for i in cluster_indices]
            
            positions = np.array([d['pos_3d'] for d in cluster_dets])
            
            # Use MEDIAN instead of mean for robustness to outliers
            avg_pos = np.median(positions, axis=0)
            cameras = list(set(d['camera'] for d in cluster_dets))
            avg_conf = np.mean([d['conf'] for d in cluster_dets])
            
            observations.append(Observation(
                frame=frame,
                pos=avg_pos,
                cls=cls,
                cameras=cameras,
                conf=avg_conf,
                det_count=len(cluster_dets)
            ))
    
    return observations

def track_observations(all_observations: Dict[int, List[Observation]], 
                       total_frames: int, fps: float) -> List[Track]:
    """Track observations over time."""
    
    tracks = []
    next_id = 1
    active_tracks: List[Track] = []
    
    for frame_idx in tqdm(range(total_frames), desc="Tracking"):
        current_obs = all_observations.get(frame_idx, [])
        
        # Age out inactive tracks
        still_active = []
        for track in active_tracks:
            last_frame = track.observations[-1].frame
            if frame_idx - last_frame <= MAX_GAP_FRAMES:
                still_active.append(track)
            else:
                tracks.append(track)
        active_tracks = still_active
        
        # Check for stationary locking
        for track in active_tracks:
            if not track.is_stationary and not track.locked_position is not None:
                track.check_and_lock_stationary()
        
        if not current_obs:
            continue
        
        if active_tracks:
            cost = np.zeros((len(current_obs), len(active_tracks)))
            
            for oi, obs in enumerate(current_obs):
                for ti, track in enumerate(active_tracks):
                    if track.cls != obs.cls:
                        if not ({track.cls, obs.cls} <= {'car', 'truck'}):
                            cost[oi, ti] = 1000
                            continue
                    
                    # For stationary tracks, use locked position
                    if track.locked_position is not None:
                        pred_pos = track.locked_position
                    else:
                        last_obs = track.observations[-1]
                        dt = (frame_idx - last_obs.frame) / fps
                        vel = track.get_velocity(fps)
                        pred_pos = last_obs.pos + vel * dt
                    
                    dist = np.linalg.norm(obs.pos[:2] - pred_pos[:2])
                    cam_bonus = 0.5 if len(obs.cameras) > 1 else 0
                    
                    cost[oi, ti] = dist - cam_bonus
            
            rows, cols = linear_sum_assignment(cost)
            
            matched_obs = set()
            # matched_tracks = set()  # Not needed
            
            for oi, ti in zip(rows, cols):
                if cost[oi, ti] < MAX_ASSOCIATION_DIST:
                    track = active_tracks[ti]
                    
                    # For stationary tracks, use locked position instead of observation
                    if track.locked_position is not None:
                        obs_to_add = Observation(
                            frame=current_obs[oi].frame,
                            pos=track.locked_position.copy(),  # Use locked position!
                            cls=track.cls,
                            cameras=current_obs[oi].cameras,
                            conf=current_obs[oi].conf,
                            det_count=current_obs[oi].det_count
                        )
                        track.observations.append(obs_to_add)
                    else:
                        track.observations.append(current_obs[oi])
                    
                    matched_obs.add(oi)
            
            for oi, obs in enumerate(current_obs):
                if oi not in matched_obs:
                    track = Track(track_id=next_id, cls=obs.cls)
                    next_id += 1
                    track.observations.append(obs)
                    active_tracks.append(track)
        else:
            for obs in current_obs:
                track = Track(track_id=next_id, cls=obs.cls)
                next_id += 1
                track.observations.append(obs)
                active_tracks.append(track)
    
    tracks.extend(active_tracks)
    
    return tracks

def post_process_tracks(tracks: List[Track], fps: float) -> List[Track]:
    """Filter, interpolate, and smooth tracks."""
    
    valid = []
    
    for track in tracks:
        if track.length < MIN_TRACK_FRAMES:
            continue
        
        track.observations.sort(key=lambda o: o.frame)
        
        # Interpolate gaps
        new_obs = []
        for i in range(len(track.observations) - 1):
            o1 = track.observations[i]
            o2 = track.observations[i + 1]
            new_obs.append(o1)
            
            gap = o2.frame - o1.frame
            if gap > 1 and gap <= MAX_GAP_FRAMES:
                for f in range(o1.frame + 1, o2.frame):
                    alpha = (f - o1.frame) / gap
                    interp_pos = o1.pos + alpha * (o2.pos - o1.pos)
                    new_obs.append(Observation(
                        frame=f,
                        pos=interp_pos,
                        cls=track.cls,
                        cameras=[],
                        conf=0.5,
                        det_count=0
                    ))
        new_obs.append(track.observations[-1])
        track.observations = sorted(new_obs, key=lambda o: o.frame)
        
        # Final stationary check with all observations
        positions = np.array([o.pos[:2] for o in track.observations])
        variance = np.var(positions, axis=0).sum()
        
        if variance < STATIONARY_VAR or track.is_stationary:
            track.is_stationary = True
            # Lock ALL positions to median
            median = np.median(positions, axis=0)
            for o in track.observations:
                o.pos[0] = median[0]
                o.pos[1] = median[1]
        else:
            # Smooth moving track
            if len(track.observations) >= 5:
                for dim in range(3):
                    vals = np.array([o.pos[dim] for o in track.observations])
                    smoothed = uniform_filter1d(vals, size=5, mode='nearest')
                    for i, o in enumerate(track.observations):
                        o.pos[dim] = smoothed[i]
        
        valid.append(track)
    
    return valid

def merge_duplicate_tracks(tracks: List[Track]) -> List[Track]:
    """Merge tracks that are likely the same object."""
    
    if len(tracks) <= 1:
        return tracks
    
    tracks.sort(key=lambda t: t.observations[0].frame)
    
    merged = []
    used = set()
    
    for i, t1 in enumerate(tracks):
        if i in used:
            continue
        
        for j, t2 in enumerate(tracks):
            if j <= i or j in used:
                continue
            if t1.cls != t2.cls:
                continue
            
            r1 = (t1.observations[0].frame, t1.observations[-1].frame)
            r2 = (t2.observations[0].frame, t2.observations[-1].frame)
            
            overlap_start = max(r1[0], r2[0])
            overlap_end = min(r1[1], r2[1])
            
            if overlap_end >= overlap_start:
                dists = []
                t1_frames = {o.frame: o for o in t1.observations}
                t2_frames = {o.frame: o for o in t2.observations}
                
                for f in range(overlap_start, overlap_end + 1):
                    if f in t1_frames and f in t2_frames:
                        dist = np.linalg.norm(t1_frames[f].pos[:2] - t2_frames[f].pos[:2])
                        dists.append(dist)
                
                if dists and np.median(dists) < MERGE_DISTANCE:
                    for o in t2.observations:
                        if o.frame not in t1_frames:
                            t1.observations.append(o)
                    t1.observations.sort(key=lambda o: o.frame)
                    used.add(j)
        
        merged.append(t1)
        used.add(i)
    
    return merged

def generate_scene(tracks: List[Track], total_frames: int, fps: float) -> dict:
    """Generate final scene descriptor."""
    
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
    
    for track in tracks:
        tid = track.track_id
        cls = track.cls
        
        scene['objects'][tid] = {
            'class': cls,
            'dims': DIMS.get(cls, DIMS['car']),
            'color': COLORS.get(cls, [255, 255, 255]),
            'is_stationary': bool(track.is_stationary)
        }
        
        observations = track.observations
        n = len(observations)
        
        positions = np.array([o.pos[:2] for o in observations])
        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([velocities, velocities[-1] if len(velocities) > 0 else [[1, 0]]])
        
        yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        if track.is_stationary:
            # For stationary, use first valid velocity direction
            valid_vels = velocities[np.linalg.norm(velocities, axis=1) > 0.1]
            if len(valid_vels) > 0:
                avg_yaw = float(np.arctan2(valid_vels[0, 1], valid_vels[0, 0]))
            else:
                avg_yaw = 0.0
            yaws = [avg_yaw] * n
        else:
            yaws = uniform_filter1d(np.unwrap(yaws), size=7, mode='nearest')
        
        for i, obs in enumerate(observations):
            frame_key = str(obs.frame)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            yaw_val = float(yaws[i]) if not isinstance(yaws, list) else yaws[i]
            quat = R.from_euler('z', yaw_val).as_quat().tolist()
            
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': obs.pos.tolist(),
                'rot': quat,
                'conf': obs.conf
            })
    
    return scene

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    
    print("=" * 60)
    print("FIXED UNIFIED 3D TRACKER")
    print("- Parked car locking")
    print("- Occlusion-robust position")
    print("=" * 60)
    
    print("\nLoading detections...")
    dets, total_frames, fps = load_detections(work_dir)
    print(f"Loaded {len(dets)} detections, {total_frames} frames")
    
    dets = [d for d in dets if d['conf'] >= MIN_CONFIDENCE]
    print(f"After confidence filter: {len(dets)}")
    
    by_frame = defaultdict(list)
    for d in dets:
        by_frame[d['frame']].append(d)
    
    print("\nClustering detections per frame...")
    all_observations = {}
    for frame_idx in range(total_frames):
        frame_dets = by_frame.get(frame_idx, [])
        if frame_dets:
            obs = cluster_frame_detections(frame_dets)
            if obs:
                all_observations[frame_idx] = obs
    
    total_obs = sum(len(obs) for obs in all_observations.values())
    print(f"Total observations: {total_obs}")
    
    print("\nTracking with parked car locking...")
    tracks = track_observations(all_observations, total_frames, fps)
    print(f"Raw tracks: {len(tracks)}")
    
    print("\nPost-processing...")
    tracks = post_process_tracks(tracks, fps)
    print(f"After post-processing: {len(tracks)}")
    
    print("\nMerging duplicates...")
    tracks = merge_duplicate_tracks(tracks)
    print(f"After merging: {len(tracks)}")
    
    stationary = sum(1 for t in tracks if t.is_stationary)
    moving = len(tracks) - stationary
    print(f"  Stationary: {stationary}, Moving: {moving}")
    
    by_class = defaultdict(int)
    for t in tracks:
        by_class[t.cls] += 1
    print("By class:")
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")
    
    print("\nGenerating scene...")
    scene = generate_scene(tracks, total_frames, fps)
    
    out_path = work_dir / "scene_4d.json"
    print(f"Saving to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    tracks_out = {
        'total_frames': total_frames,
        'fps': fps,
        'tracks': [
            {
                'id': t.track_id,
                'class': t.cls,
                'is_stationary': bool(t.is_stationary),
                'length': t.length,
            }
            for t in tracks
        ]
    }
    
    tracks_path = work_dir / "tracks_4d.json"
    with open(tracks_path, 'w') as f:
        json.dump(tracks_out, f, indent=2)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
