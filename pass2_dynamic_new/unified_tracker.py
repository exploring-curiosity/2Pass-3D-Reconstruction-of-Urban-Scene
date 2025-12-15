#!/usr/bin/env python3
"""
Unified 3D Tracker - Work Entirely in World Coordinates
========================================================
Key insight: All 8 cameras view the SAME intersection.
Expected: ~15-20 unique objects, NOT 87.

Strategy:
1. Per-frame: Cluster ALL detections from ALL cameras in 3D
2. One cluster = one object observation
3. Track clusters over time using Kalman filter
4. Result: Correct number of unique objects
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

# Thresholds - VERY AGGRESSIVE for ~15-20 objects
MIN_CONFIDENCE = 0.5           # High confidence only
CLUSTER_EPS = 5.0              # meters - VERY LARGE to merge aggressively
MIN_TRACK_FRAMES = 60          # VERY LONG minimum (2 seconds at 30fps)
MAX_ASSOCIATION_DIST = 6.0     # meters for frame-to-frame association
MAX_GAP_FRAMES = 5             # Short gaps only
STATIONARY_VAR = 2.5           # m² variance threshold
MERGE_DISTANCE = 5.0           # meters for track merging

@dataclass
class Observation:
    """One observation of an object at a single frame (from 1+ cameras)."""
    frame: int
    pos: np.ndarray          # Average 3D position
    cls: str                  # Class (majority vote)
    cameras: List[str]        # Which cameras saw it
    conf: float               # Average confidence
    det_count: int            # Number of detections

@dataclass
class Track:
    """A tracked object over time."""
    track_id: int
    cls: str
    observations: List[Observation] = field(default_factory=list)
    is_stationary: bool = False
    
    @property
    def length(self):
        return len(self.observations)
    
    def get_last_pos(self) -> np.ndarray:
        if self.observations:
            return self.observations[-1].pos
        return np.zeros(3)
    
    def get_velocity(self, fps: float) -> np.ndarray:
        """Estimate current velocity from recent observations."""
        if len(self.observations) < 2:
            return np.zeros(3)
        
        # Use last 5 observations
        recent = self.observations[-5:]
        if len(recent) < 2:
            return np.zeros(3)
        
        pos1 = recent[0].pos
        pos2 = recent[-1].pos
        dt = (recent[-1].frame - recent[0].frame) / fps
        
        if dt > 0:
            return (pos2 - pos1) / dt
        return np.zeros(3)

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
    
    # Group by class first
    by_class = defaultdict(list)
    for d in dets:
        by_class[d['class']].append(d)
    
    observations = []
    
    for cls, class_dets in by_class.items():
        if len(class_dets) == 1:
            # Single detection
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
        
        # Multiple detections - cluster
        points = np.array([d['pos_3d'][:2] for d in class_dets])
        
        db = DBSCAN(eps=eps, min_samples=1).fit(points)
        labels = db.labels_
        
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_dets = [class_dets[i] for i in cluster_indices]
            
            # Compute observation
            positions = np.array([d['pos_3d'] for d in cluster_dets])
            avg_pos = np.mean(positions, axis=0)
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
    """Track observations over time using greedy association."""
    
    tracks = []
    next_id = 1
    active_tracks: List[Track] = []  # Currently active tracks
    
    for frame_idx in tqdm(range(total_frames), desc="Tracking"):
        current_obs = all_observations.get(frame_idx, [])
        
        # Age out inactive tracks
        still_active = []
        for track in active_tracks:
            last_frame = track.observations[-1].frame
            if frame_idx - last_frame <= MAX_GAP_FRAMES:
                still_active.append(track)
            else:
                tracks.append(track)  # Finalize
        active_tracks = still_active
        
        if not current_obs:
            continue
        
        # Associate observations to tracks
        if active_tracks:
            cost = np.zeros((len(current_obs), len(active_tracks)))
            
            for oi, obs in enumerate(current_obs):
                for ti, track in enumerate(active_tracks):
                    # Class check
                    if track.cls != obs.cls:
                        # Allow car<->truck
                        if not ({track.cls, obs.cls} <= {'car', 'truck'}):
                            cost[oi, ti] = 1000
                            continue
                    
                    # Predict position
                    last_obs = track.observations[-1]
                    dt = (frame_idx - last_obs.frame) / fps
                    vel = track.get_velocity(fps)
                    pred_pos = last_obs.pos + vel * dt
                    
                    # Distance
                    dist = np.linalg.norm(obs.pos[:2] - pred_pos[:2])
                    
                    # Prefer multi-camera observations
                    cam_bonus = 0.5 if len(obs.cameras) > 1 else 0
                    
                    cost[oi, ti] = dist - cam_bonus
            
            rows, cols = linear_sum_assignment(cost)
            
            matched_obs = set()
            matched_tracks = set()
            
            for oi, ti in zip(rows, cols):
                if cost[oi, ti] < MAX_ASSOCIATION_DIST:
                    track = active_tracks[ti]
                    track.observations.append(current_obs[oi])
                    matched_obs.add(oi)
                    matched_tracks.add(ti)
            
            # Keep unmatched tracks active
            new_active = []
            for ti, track in enumerate(active_tracks):
                if ti in matched_tracks or (frame_idx - track.observations[-1].frame <= MAX_GAP_FRAMES):
                    new_active.append(track)
                else:
                    tracks.append(track)
            active_tracks = new_active
            
            # Create new tracks for unmatched observations
            for oi, obs in enumerate(current_obs):
                if oi not in matched_obs:
                    track = Track(track_id=next_id, cls=obs.cls)
                    next_id += 1
                    track.observations.append(obs)
                    active_tracks.append(track)
        else:
            # No active tracks
            for obs in current_obs:
                track = Track(track_id=next_id, cls=obs.cls)
                next_id += 1
                track.observations.append(obs)
                active_tracks.append(track)
    
    # Collect remaining active tracks
    tracks.extend(active_tracks)
    
    return tracks

def post_process_tracks(tracks: List[Track], fps: float) -> List[Track]:
    """Filter, interpolate, and smooth tracks."""
    
    valid = []
    
    for track in tracks:
        # Length filter
        if track.length < MIN_TRACK_FRAMES:
            continue
        
        # Sort observations
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
        
        # Detect stationary
        positions = np.array([o.pos[:2] for o in track.observations])
        variance = np.var(positions, axis=0).sum()
        track.is_stationary = variance < STATIONARY_VAR
        
        # Lock stationary to median
        if track.is_stationary:
            median = np.median(positions, axis=0)
            for o in track.observations:
                o.pos[0] = median[0]
                o.pos[1] = median[1]
        else:
            # Smooth moving objects
            if len(track.observations) >= 5:
                for dim in range(3):
                    vals = np.array([o.pos[dim] for o in track.observations])
                    smoothed = uniform_filter1d(vals, size=5, mode='nearest')
                    for i, o in enumerate(track.observations):
                        o.pos[dim] = smoothed[i]
        
        valid.append(track)
    
    return valid

def merge_duplicate_tracks(tracks: List[Track]) -> List[Track]:
    """Merge tracks that are likely the same object (close in space/time)."""
    
    if len(tracks) <= 1:
        return tracks
    
    # Sort by start frame
    tracks.sort(key=lambda t: t.observations[0].frame)
    
    merged = []
    used = set()
    
    for i, t1 in enumerate(tracks):
        if i in used:
            continue
        
        # Find overlapping tracks
        for j, t2 in enumerate(tracks):
            if j <= i or j in used:
                continue
            if t1.cls != t2.cls:
                continue
            
            # Check temporal overlap
            r1 = (t1.observations[0].frame, t1.observations[-1].frame)
            r2 = (t2.observations[0].frame, t2.observations[-1].frame)
            
            overlap_start = max(r1[0], r2[0])
            overlap_end = min(r1[1], r2[1])
            
            if overlap_end >= overlap_start:
                # Check spatial proximity during overlap
                dists = []
                t1_frames = {o.frame: o for o in t1.observations}
                t2_frames = {o.frame: o for o in t2.observations}
                
                for f in range(overlap_start, overlap_end + 1):
                    if f in t1_frames and f in t2_frames:
                        dist = np.linalg.norm(t1_frames[f].pos[:2] - t2_frames[f].pos[:2])
                        dists.append(dist)
                
                if dists and np.median(dists) < MERGE_DISTANCE:
                    # Merge t2 into t1
                    # Add non-overlapping observations from t2
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
        
        # Compute orientations
        observations = track.observations
        n = len(observations)
        
        positions = np.array([o.pos[:2] for o in observations])
        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([velocities, velocities[-1] if len(velocities) > 0 else [[1, 0]]])
        
        yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        if track.is_stationary:
            avg_yaw = float(np.mean(yaws))
            yaws = [avg_yaw] * n
        else:
            yaws = uniform_filter1d(np.unwrap(yaws), size=7, mode='nearest')
        
        for i, obs in enumerate(observations):
            frame_key = str(obs.frame)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            quat = R.from_euler('z', float(yaws[i])).as_quat().tolist()
            
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
    print("UNIFIED 3D TRACKER")
    print("=" * 60)
    
    print("\nLoading detections...")
    dets, total_frames, fps = load_detections(work_dir)
    print(f"Loaded {len(dets)} detections, {total_frames} frames")
    
    # Filter by confidence
    dets = [d for d in dets if d['conf'] >= MIN_CONFIDENCE]
    print(f"After confidence filter: {len(dets)}")
    
    # Group by frame
    by_frame = defaultdict(list)
    for d in dets:
        by_frame[d['frame']].append(d)
    
    print("\nClustering detections per frame (all cameras together)...")
    all_observations = {}
    for frame_idx in range(total_frames):
        frame_dets = by_frame.get(frame_idx, [])
        if frame_dets:
            obs = cluster_frame_detections(frame_dets)
            if obs:
                all_observations[frame_idx] = obs
    
    total_obs = sum(len(obs) for obs in all_observations.values())
    print(f"Total observations: {total_obs}")
    
    avg_per_frame = total_obs / max(1, len(all_observations))
    print(f"Average per frame: {avg_per_frame:.1f}")
    
    print("\nTracking observations over time...")
    tracks = track_observations(all_observations, total_frames, fps)
    print(f"Raw tracks: {len(tracks)}")
    
    print("\nPost-processing tracks...")
    tracks = post_process_tracks(tracks, fps)
    print(f"After post-processing: {len(tracks)}")
    
    print("\nMerging duplicate tracks...")
    tracks = merge_duplicate_tracks(tracks)
    print(f"After merging: {len(tracks)}")
    
    # Stats
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
