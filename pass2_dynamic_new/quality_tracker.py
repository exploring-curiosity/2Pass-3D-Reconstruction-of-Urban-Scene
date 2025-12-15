#!/usr/bin/env python3
"""
High-Quality 4D Tracker - PRESENTATION READY
=============================================
Aggressive filtering for clean, presentable output:
1. Higher confidence threshold
2. Strict multi-camera validation  
3. Temporal consistency requirements
4. Parked vehicle stability
5. Remove ALL spurious/short tracks
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import cv2
import sys

# STRICT THRESHOLDS
MIN_CONFIDENCE = 0.4           # Higher confidence
MIN_TRACK_LENGTH = 30          # Much longer tracks required
MIN_ANCHOR_CAMERAS = 2         # MUST have 2+ camera views
MAX_SPEED_CAR = 15.0           # m/s - conservative
MAX_SPEED_PERSON = 2.5         # m/s
STATIONARY_THRESHOLD = 1.0     # m variance for stationary detection
SPATIAL_CLUSTER_EPS = 2.0      # m for clustering
MAX_GAP_FRAMES = 10            # Max gap to interpolate

@dataclass
class Detection:
    frame: int
    camera: str
    cls: str
    conf: float
    bbox: List[float]
    pos_3d: np.ndarray
    idx: int

@dataclass
class Track:
    track_id: int
    cls: str
    nodes: List[dict] = field(default_factory=list)
    is_stationary: bool = False
    confidence_score: float = 0.0

def load_data(work_dir: Path):
    """Load detections."""
    det_path = work_dir / "detections_3d.json"
    with open(det_path) as f:
        data = json.load(f)
    return data['detections'], data['total_frames'], data['fps']

def filter_detections(dets: List[dict], min_conf: float = MIN_CONFIDENCE):
    """Filter low-confidence detections."""
    filtered = []
    for i, d in enumerate(dets):
        if d['conf'] >= min_conf:
            det = Detection(
                frame=d['frame'],
                camera=d['camera'],
                cls=d['class'],
                conf=d['conf'],
                bbox=d['bbox'],
                pos_3d=np.array(d['pos_3d']),
                idx=i
            )
            filtered.append(det)
    return filtered

def find_strict_anchors(dets: List[Detection], total_frames: int) -> List[dict]:
    """Find anchors with STRICT multi-camera requirement."""
    by_frame = defaultdict(list)
    for d in dets:
        by_frame[d.frame].append(d)
    
    anchors = []
    
    for frame_idx in range(total_frames):
        frame_dets = by_frame[frame_idx]
        if len(frame_dets) < 2:
            continue
        
        # Group by class
        by_class = defaultdict(list)
        for d in frame_dets:
            by_class[d.cls].append(d)
        
        for cls, class_dets in by_class.items():
            if len(class_dets) < 2:
                continue
            
            # Spatial clustering
            points = np.array([d.pos_3d[:2] for d in class_dets])
            
            if len(points) >= 2:
                db = DBSCAN(eps=SPATIAL_CLUSTER_EPS, min_samples=2).fit(points)
                labels = db.labels_
                
                for label in set(labels):
                    if label == -1:
                        continue
                    
                    cluster_indices = np.where(labels == label)[0]
                    cluster_dets = [class_dets[i] for i in cluster_indices]
                    
                    # STRICT: Must have 2+ different cameras
                    cameras = set(d.camera for d in cluster_dets)
                    if len(cameras) >= MIN_ANCHOR_CAMERAS:
                        # Compute centroid
                        positions = np.array([d.pos_3d for d in cluster_dets])
                        centroid = np.mean(positions, axis=0)
                        
                        # Average confidence
                        avg_conf = np.mean([d.conf for d in cluster_dets])
                        
                        # Only accept high-confidence anchors
                        if avg_conf >= MIN_CONFIDENCE:
                            anchors.append({
                                'frame': frame_idx,
                                'pos': centroid,
                                'class': cls,
                                'cameras': list(cameras),
                                'conf': avg_conf,
                                'det_indices': [d.idx for d in cluster_dets]
                            })
    
    return anchors

def build_clean_tracks(anchors: List[dict], total_frames: int, fps: float) -> List[Track]:
    """Build tracks with strict physics and temporal constraints."""
    if not anchors:
        return []
    
    anchors_by_frame = defaultdict(list)
    for a in anchors:
        anchors_by_frame[a['frame']].append(a)
    
    tracks = []
    next_id = 1
    active = []  # (track_idx, last_anchor, velocity)
    
    for frame_idx in tqdm(range(total_frames), desc="Building tracks"):
        current = anchors_by_frame[frame_idx]
        
        # Age out old tracks
        new_active = []
        for track_idx, last_anchor, vel in active:
            gap = frame_idx - last_anchor['frame']
            if gap <= MAX_GAP_FRAMES:
                new_active.append((track_idx, last_anchor, vel))
        active = new_active
        
        if not current:
            continue
        
        # Match to active tracks
        if active:
            cost = np.zeros((len(current), len(active)))
            
            for ai, anchor in enumerate(current):
                for ti, (track_idx, last_anchor, vel) in enumerate(active):
                    track = tracks[track_idx]
                    
                    # Class check
                    if track.cls != anchor['class']:
                        if not ({track.cls, anchor['class']} <= {'car', 'truck'}):
                            cost[ai, ti] = 1000
                            continue
                    
                    # Predict position
                    dt = (anchor['frame'] - last_anchor['frame']) / fps
                    if vel is not None and np.linalg.norm(vel) > 0.1:
                        pred_pos = last_anchor['pos'] + vel * dt
                    else:
                        pred_pos = last_anchor['pos']
                    
                    dist = np.linalg.norm(anchor['pos'][:2] - pred_pos[:2])
                    
                    # Speed check
                    speed = dist / dt if dt > 0 else 0
                    max_speed = MAX_SPEED_CAR if track.cls in ['car', 'truck', 'bus'] else MAX_SPEED_PERSON
                    
                    if speed > max_speed * 1.3:
                        cost[ai, ti] = 500
                    else:
                        cost[ai, ti] = dist
            
            rows, cols = linear_sum_assignment(cost)
            
            matched_anchors = set()
            new_active_list = []
            
            for ai, ti in zip(rows, cols):
                if cost[ai, ti] < 5.0:
                    track_idx, last_anchor, old_vel = active[ti]
                    anchor = current[ai]
                    
                    # Compute new velocity
                    dt = (anchor['frame'] - last_anchor['frame']) / fps
                    if dt > 0:
                        new_vel = (anchor['pos'] - last_anchor['pos']) / dt
                    else:
                        new_vel = old_vel
                    
                    # Add to track
                    tracks[track_idx].nodes.append({
                        'frame': anchor['frame'],
                        'pos': anchor['pos'].tolist(),
                        'cameras': anchor['cameras'],
                        'conf': anchor['conf']
                    })
                    tracks[track_idx].confidence_score += anchor['conf']
                    
                    new_active_list.append((track_idx, anchor, new_vel))
                    matched_anchors.add(ai)
            
            # Keep unmatched active tracks
            for ti, (track_idx, last_anchor, vel) in enumerate(active):
                if ti not in [c for _, c in zip(rows, cols) if cost[_, c] < 5.0]:
                    gap = frame_idx - last_anchor['frame']
                    if gap <= MAX_GAP_FRAMES:
                        new_active_list.append((track_idx, last_anchor, vel))
            
            active = new_active_list
            
            # Create new tracks for unmatched
            for ai, anchor in enumerate(current):
                if ai not in matched_anchors:
                    track = Track(track_id=next_id, cls=anchor['class'])
                    next_id += 1
                    track.nodes.append({
                        'frame': anchor['frame'],
                        'pos': anchor['pos'].tolist(),
                        'cameras': anchor['cameras'],
                        'conf': anchor['conf']
                    })
                    track.confidence_score = anchor['conf']
                    tracks.append(track)
                    active.append((len(tracks) - 1, anchor, None))
        else:
            # No active, create new
            for anchor in current:
                track = Track(track_id=next_id, cls=anchor['class'])
                next_id += 1
                track.nodes.append({
                    'frame': anchor['frame'],
                    'pos': anchor['pos'].tolist(),
                    'cameras': anchor['cameras'],
                    'conf': anchor['conf']
                })
                track.confidence_score = anchor['conf']
                tracks.append(track)
                active.append((len(tracks) - 1, anchor, None))
    
    return tracks

def filter_and_smooth_tracks(tracks: List[Track], min_length: int = MIN_TRACK_LENGTH) -> List[Track]:
    """Aggressive filtering and smoothing."""
    valid = []
    
    for track in tracks:
        # Length filter
        if len(track.nodes) < min_length:
            continue
        
        # Sort nodes
        track.nodes.sort(key=lambda n: n['frame'])
        
        # Compute statistics
        positions = np.array([n['pos'][:2] for n in track.nodes])
        variance = np.var(positions, axis=0).sum()
        
        # Detect stationary
        track.is_stationary = variance < STATIONARY_THRESHOLD
        
        # Smooth positions
        if len(track.nodes) >= 5:
            for dim in range(3):
                vals = np.array([n['pos'][dim] for n in track.nodes])
                smoothed = uniform_filter1d(vals, size=7, mode='nearest')
                for i, n in enumerate(track.nodes):
                    n['pos'][dim] = float(smoothed[i])
        
        # For stationary objects, lock to median position
        if track.is_stationary:
            median_pos = np.median(positions, axis=0)
            for n in track.nodes:
                n['pos'][0] = float(median_pos[0])
                n['pos'][1] = float(median_pos[1])
        
        valid.append(track)
    
    return valid

def compute_orientations(track: Track) -> List[float]:
    """Compute heading from velocity, with smoothing."""
    nodes = track.nodes
    n = len(nodes)
    
    if n < 2:
        return [0.0] * n
    
    positions = np.array([node['pos'][:2] for node in nodes])
    
    # Compute velocities
    velocities = np.diff(positions, axis=0)
    velocities = np.vstack([velocities, velocities[-1]])  # Pad last
    
    # Compute yaw
    yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
    
    # For stationary, use average
    if track.is_stationary:
        avg_yaw = np.mean(yaws)
        return [float(avg_yaw)] * n
    
    # Smooth
    yaws_unwrap = np.unwrap(yaws)
    yaws_smooth = uniform_filter1d(yaws_unwrap, size=9, mode='nearest')
    
    return [float(y) for y in yaws_smooth]

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
        yaws = compute_orientations(track)
        
        for i, node in enumerate(track.nodes):
            frame_key = str(node['frame'])
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            quat = R.from_euler('z', yaws[i]).as_quat().tolist()
            
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': node['pos'],
                'rot': quat,
                'conf': node['conf']
            })
    
    return scene

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    
    print("=" * 60)
    print("HIGH-QUALITY 4D TRACKER - PRESENTATION READY")
    print("=" * 60)
    
    print("\nLoading detections...")
    dets, total_frames, fps = load_data(work_dir)
    print(f"Loaded {len(dets)} raw detections")
    
    print(f"\nFiltering (conf >= {MIN_CONFIDENCE})...")
    filtered_dets = filter_detections(dets, MIN_CONFIDENCE)
    print(f"After filtering: {len(filtered_dets)} detections")
    
    print("\nFinding strict multi-camera anchors...")
    anchors = find_strict_anchors(filtered_dets, total_frames)
    print(f"Found {len(anchors)} high-quality anchors")
    
    print("\nBuilding tracks with physics constraints...")
    tracks = build_clean_tracks(anchors, total_frames, fps)
    print(f"Built {len(tracks)} tracks")
    
    print(f"\nFiltering (length >= {MIN_TRACK_LENGTH}) and smoothing...")
    tracks = filter_and_smooth_tracks(tracks, MIN_TRACK_LENGTH)
    print(f"Final: {len(tracks)} clean tracks")
    
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
    
    # Also save tracks
    tracks_out = {
        'total_frames': total_frames,
        'fps': fps,
        'tracks': [
            {
                'id': t.track_id,
                'class': t.cls,
                'is_stationary': bool(t.is_stationary),
                'length': len(t.nodes),
                'trajectory': t.nodes
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
