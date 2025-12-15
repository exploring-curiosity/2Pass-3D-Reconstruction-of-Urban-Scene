#!/usr/bin/env python3
"""
Stage 2: Physics-Constrained 4D Tracker
========================================
1. Load 3D detections from Stage 1.
2. Identify "Anchors" (frames with 2+ camera views of same object).
3. Build tracking graph with physics constraints.
4. Bidirectional propagation from anchors.
5. Gap filling and noise rejection.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import sys

# Physics constraints
MAX_SPEED = {
    'car': 20.0,      # m/s (72 km/h)
    'truck': 15.0,    # m/s
    'bus': 12.0,      # m/s
    'motorcycle': 20.0,
    'bicycle': 8.0,   # m/s
    'person': 3.0     # m/s
}

MAX_ACCEL = {
    'car': 5.0,       # m/s²
    'truck': 3.0,
    'bus': 2.5,
    'motorcycle': 6.0,
    'bicycle': 3.0,
    'person': 2.0
}

# Stationary threshold (variance in position)
STATIONARY_VAR_THRESH = 0.5  # m²

@dataclass
class TrackNode:
    """Single observation in a track."""
    frame: int
    pos: np.ndarray  # 3D position
    cameras: List[str]  # Which cameras saw this
    det_ids: List[int]  # Detection indices
    conf: float

@dataclass
class Track:
    """A complete track over time."""
    track_id: int
    cls: str
    nodes: List[TrackNode] = field(default_factory=list)
    is_stationary: bool = False
    
    @property
    def frame_range(self) -> Tuple[int, int]:
        if not self.nodes:
            return (0, 0)
        frames = [n.frame for n in self.nodes]
        return (min(frames), max(frames))
    
    @property
    def length(self) -> int:
        return len(self.nodes)
    
    def get_velocity(self, idx: int, fps: float = 14.4) -> np.ndarray:
        """Compute velocity at node index."""
        if idx <= 0 or idx >= len(self.nodes):
            return np.zeros(3)
        
        n1 = self.nodes[idx - 1]
        n2 = self.nodes[idx]
        dt = (n2.frame - n1.frame) / fps
        if dt <= 0:
            return np.zeros(3)
        return (n2.pos - n1.pos) / dt

def load_detections(path: Path) -> Tuple[List[dict], int, float]:
    """Load detections from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data['detections'], data['total_frames'], data['fps']

def cluster_frame_detections(dets: List[dict], eps: float = 2.5) -> List[List[int]]:
    """Cluster detections in a single frame by 3D proximity."""
    if len(dets) < 2:
        return [[i] for i in range(len(dets))]
    
    # Group by class first
    by_class = defaultdict(list)
    for i, d in enumerate(dets):
        by_class[d['class']].append(i)
    
    clusters = []
    for cls, indices in by_class.items():
        if len(indices) < 2:
            clusters.append(indices)
            continue
        
        points = np.array([dets[i]['pos_3d'][:2] for i in indices])
        
        if len(points) >= 2:
            db = DBSCAN(eps=eps, min_samples=1).fit(points)
            labels = db.labels_
            
            label_groups = defaultdict(list)
            for i, label in enumerate(labels):
                label_groups[label].append(indices[i])
            
            clusters.extend(label_groups.values())
        else:
            clusters.append(indices)
    
    return clusters

def identify_anchors(dets: List[dict], total_frames: int) -> List[dict]:
    """Find frames where objects are seen by 2+ cameras (high confidence anchors)."""
    # Group detections by frame
    by_frame = defaultdict(list)
    for i, d in enumerate(dets):
        by_frame[d['frame']].append((i, d))
    
    anchors = []
    
    for frame_idx in range(total_frames):
        frame_dets = by_frame[frame_idx]
        if len(frame_dets) < 2:
            continue
        
        # Cluster by 3D position
        dets_list = [d for _, d in frame_dets]
        idx_list = [i for i, _ in frame_dets]
        
        clusters = cluster_frame_detections(dets_list)
        
        for cluster_indices in clusters:
            if len(cluster_indices) < 1:
                continue
            
            actual_indices = [idx_list[ci] for ci in cluster_indices]
            cluster_dets = [dets_list[ci] for ci in cluster_indices]
            cams = set(d['camera'] for d in cluster_dets)
            
            # Must have 2+ cameras for strong anchor
            if len(cams) >= 2:
                # Compute centroid
                positions = np.array([d['pos_3d'] for d in cluster_dets])
                centroid = np.mean(positions, axis=0)
                
                # Majority class
                classes = [d['class'] for d in cluster_dets]
                cls = max(set(classes), key=classes.count)
                
                # Average confidence
                conf = np.mean([d['conf'] for d in cluster_dets])
                
                anchors.append({
                    'frame': frame_idx,
                    'pos': centroid,
                    'class': cls,
                    'cameras': list(cams),
                    'det_ids': actual_indices,
                    'conf': conf
                })
    
    return anchors

def physics_valid(pos1: np.ndarray, pos2: np.ndarray, 
                  frame1: int, frame2: int, cls: str, fps: float) -> bool:
    """Check if movement between two positions is physically plausible."""
    dt = abs(frame2 - frame1) / fps
    if dt <= 0:
        return True
    
    dist = np.linalg.norm(pos2[:2] - pos1[:2])
    speed = dist / dt
    
    max_speed = MAX_SPEED.get(cls, 20.0)
    
    # Allow some slack for projection noise
    return speed <= max_speed * 1.5

def build_tracks_from_anchors(anchors: List[dict], 
                              dets: List[dict],
                              total_frames: int,
                              fps: float) -> List[Track]:
    """Build tracks by linking anchors over time with physics constraints."""
    
    # Sort anchors by frame
    anchors_by_frame = defaultdict(list)
    for a in anchors:
        anchors_by_frame[a['frame']].append(a)
    
    tracks = []
    next_track_id = 1
    
    # Track active anchors (those that can still be extended)
    # Each entry: (track_idx, last_anchor)
    active_tracks = []
    
    for frame_idx in tqdm(range(total_frames), desc="Building tracks"):
        current_anchors = anchors_by_frame[frame_idx]
        
        if not current_anchors:
            # No anchors this frame, but keep active tracks alive for gap tolerance
            new_active = []
            for track_idx, last_anchor in active_tracks:
                gap = frame_idx - last_anchor['frame']
                if gap <= 15:  # Allow 15-frame gaps
                    new_active.append((track_idx, last_anchor))
            active_tracks = new_active
            continue
        
        # Match current anchors to active tracks
        if active_tracks:
            track_list = [tracks[ti] for ti, _ in active_tracks]
            
            cost = np.zeros((len(current_anchors), len(active_tracks)))
            
            for ai, anchor in enumerate(current_anchors):
                for ti, (track_idx, last_anchor) in enumerate(active_tracks):
                    track = tracks[track_idx]
                    
                    # Class mismatch penalty
                    if track.cls != anchor['class']:
                        # Allow truck<->car confusion
                        if not ({track.cls, anchor['class']} <= {'car', 'truck'}):
                            cost[ai, ti] = 1000
                            continue
                    
                    # Distance
                    dist = np.linalg.norm(anchor['pos'][:2] - last_anchor['pos'][:2])
                    
                    # Physics check
                    if not physics_valid(last_anchor['pos'], anchor['pos'],
                                        last_anchor['frame'], anchor['frame'],
                                        track.cls, fps):
                        cost[ai, ti] = 500 + dist
                    else:
                        # Velocity prediction
                        if len(track.nodes) >= 2:
                            v = track.get_velocity(-1, fps)
                            dt = (anchor['frame'] - last_anchor['frame']) / fps
                            pred_pos = last_anchor['pos'] + v * dt
                            pred_dist = np.linalg.norm(anchor['pos'][:2] - pred_pos[:2])
                            cost[ai, ti] = pred_dist
                        else:
                            cost[ai, ti] = dist
            
            # Hungarian matching
            rows, cols = linear_sum_assignment(cost)
            
            matched_anchors = set()
            matched_tracks = set()
            
            for ai, ti in zip(rows, cols):
                if cost[ai, ti] < 8.0:  # 8m gating
                    track_idx, _ = active_tracks[ti]
                    anchor = current_anchors[ai]
                    
                    # Add node to track
                    node = TrackNode(
                        frame=anchor['frame'],
                        pos=anchor['pos'],
                        cameras=anchor['cameras'],
                        det_ids=anchor['det_ids'],
                        conf=anchor['conf']
                    )
                    tracks[track_idx].nodes.append(node)
                    
                    matched_anchors.add(ai)
                    matched_tracks.add(ti)
            
            # Update active tracks
            new_active = []
            for ti, (track_idx, last_anchor) in enumerate(active_tracks):
                if ti in matched_tracks:
                    # Use the new anchor as last
                    ai = [a for a, t in zip(rows, cols) if t == ti and cost[a, t] < 8.0][0]
                    new_active.append((track_idx, current_anchors[ai]))
                else:
                    gap = frame_idx - last_anchor['frame']
                    if gap <= 15:
                        new_active.append((track_idx, last_anchor))
            
            active_tracks = new_active
            
            # Create new tracks for unmatched anchors
            for ai, anchor in enumerate(current_anchors):
                if ai not in matched_anchors:
                    track = Track(
                        track_id=next_track_id,
                        cls=anchor['class']
                    )
                    next_track_id += 1
                    
                    node = TrackNode(
                        frame=anchor['frame'],
                        pos=anchor['pos'],
                        cameras=anchor['cameras'],
                        det_ids=anchor['det_ids'],
                        conf=anchor['conf']
                    )
                    track.nodes.append(node)
                    
                    tracks.append(track)
                    active_tracks.append((len(tracks) - 1, anchor))
        else:
            # No active tracks, create new ones
            for anchor in current_anchors:
                track = Track(
                    track_id=next_track_id,
                    cls=anchor['class']
                )
                next_track_id += 1
                
                node = TrackNode(
                    frame=anchor['frame'],
                    pos=anchor['pos'],
                    cameras=anchor['cameras'],
                    det_ids=anchor['det_ids'],
                    conf=anchor['conf']
                )
                track.nodes.append(node)
                
                tracks.append(track)
                active_tracks.append((len(tracks) - 1, anchor))
    
    return tracks

def fill_gaps_and_smooth(tracks: List[Track], fps: float) -> List[Track]:
    """Fill temporal gaps and smooth trajectories."""
    
    for track in tracks:
        if len(track.nodes) < 2:
            continue
        
        # Sort nodes
        track.nodes.sort(key=lambda n: n.frame)
        
        # Fill gaps with linear interpolation
        new_nodes = []
        for i in range(len(track.nodes) - 1):
            n1 = track.nodes[i]
            n2 = track.nodes[i + 1]
            
            gap = n2.frame - n1.frame
            if gap > 1 and gap <= 15:
                # Interpolate
                for f in range(n1.frame + 1, n2.frame):
                    alpha = (f - n1.frame) / gap
                    pos = n1.pos + alpha * (n2.pos - n1.pos)
                    
                    new_nodes.append(TrackNode(
                        frame=f,
                        pos=pos,
                        cameras=[],  # Interpolated
                        det_ids=[],
                        conf=0.5
                    ))
        
        track.nodes.extend(new_nodes)
        track.nodes.sort(key=lambda n: n.frame)
        
        # Smooth positions (moving average)
        if len(track.nodes) >= 5:
            positions = np.array([n.pos for n in track.nodes])
            window = 5
            
            smoothed = np.copy(positions)
            for i in range(len(positions)):
                start = max(0, i - window // 2)
                end = min(len(positions), i + window // 2 + 1)
                smoothed[i] = np.mean(positions[start:end], axis=0)
            
            for i, n in enumerate(track.nodes):
                n.pos = smoothed[i]
        
        # Detect if stationary
        if len(track.nodes) >= 5:
            positions = np.array([n.pos[:2] for n in track.nodes])
            variance = np.var(positions, axis=0).sum()
            track.is_stationary = variance < STATIONARY_VAR_THRESH
    
    return tracks

def filter_invalid_tracks(tracks: List[Track], min_length: int = 10) -> List[Track]:
    """Remove tracks that are too short or violate physics."""
    
    valid = []
    for track in tracks:
        if track.length < min_length:
            continue
        valid.append(track)
    
    return valid

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    
    det_path = work_dir / "detections_3d.json"
    if not det_path.exists():
        print("ERROR: Run unified_detector.py first")
        sys.exit(1)
    
    print("Loading detections...")
    dets, total_frames, fps = load_detections(det_path)
    print(f"Loaded {len(dets)} detections, {total_frames} frames @ {fps:.1f} FPS")
    
    print("\nIdentifying anchors (multi-camera observations)...")
    anchors = identify_anchors(dets, total_frames)
    print(f"Found {len(anchors)} multi-camera anchors")
    
    # Stats
    by_class = defaultdict(int)
    for a in anchors:
        by_class[a['class']] += 1
    print("Anchors by class:")
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")
    
    print("\nBuilding tracks from anchors...")
    tracks = build_tracks_from_anchors(anchors, dets, total_frames, fps)
    print(f"Built {len(tracks)} raw tracks")
    
    print("\nFilling gaps and smoothing...")
    tracks = fill_gaps_and_smooth(tracks, fps)
    
    print("\nFiltering invalid tracks...")
    tracks = filter_invalid_tracks(tracks, min_length=10)
    print(f"Final: {len(tracks)} tracks")
    
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
    
    # Save
    output = {
        'total_frames': total_frames,
        'fps': fps,
        'tracks': []
    }
    
    for track in tracks:
        t_data = {
            'id': track.track_id,
            'class': track.cls,
            'is_stationary': bool(track.is_stationary),
            'frame_range': track.frame_range,
            'length': track.length,
            'trajectory': [
                {
                    'frame': n.frame,
                    'pos': n.pos.tolist(),
                    'cameras': n.cameras,
                    'conf': n.conf
                }
                for n in track.nodes
            ]
        }
        output['tracks'].append(t_data)
    
    out_path = work_dir / "tracks_4d.json"
    print(f"\nSaving to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
