#!/usr/bin/env python3
"""
Balanced 4D Tracker - Per-Camera First, Then Associate
=======================================================
Strategy:
1. Run per-camera tracking (using YOLO tracker or IoU-based)
2. Build per-camera tracks with temporal consistency
3. Associate across cameras using 3D overlap
4. Use image center zone for reliable projections
5. Filter by track quality (length + consistency)
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
import cv2
from tqdm import tqdm
import sys

# Thresholds - BALANCED
MIN_CONFIDENCE = 0.35        # Reasonable confidence
MIN_TRACK_LENGTH = 15        # Shorter tracks OK
MIN_TRACK_LENGTH_SINGLE_CAM = 25  # But single-cam needs longer
IOU_THRESHOLD = 0.3          # For frame-to-frame association
MAX_SPEED_CAR = 18.0         # m/s
MAX_SPEED_PERSON = 3.0       # m/s
EDGE_MARGIN_RATIO = 0.15     # Ignore detections in outer 15% of image

# Image dimensions (will be updated from video)
IMG_W = 2592
IMG_H = 1944

@dataclass
class Detection:
    frame: int
    camera: str
    cls: str
    conf: float
    bbox: List[float]  # [x1, y1, x2, y2]
    pos_3d: np.ndarray
    idx: int
    
    @property
    def center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    def is_in_center_zone(self, img_w, img_h, margin=EDGE_MARGIN_RATIO):
        """Check if detection center is in reliable zone (not at edges)."""
        cx, cy = self.center
        x_margin = img_w * margin
        y_margin = img_h * margin
        return (x_margin < cx < img_w - x_margin and 
                y_margin < cy < img_h - y_margin)

@dataclass
class PerCameraTrack:
    track_id: int
    camera: str
    cls: str
    detections: Dict[int, Detection] = field(default_factory=dict)  # frame -> det
    
    @property
    def length(self):
        return len(self.detections)
    
    @property
    def frame_range(self):
        if not self.detections:
            return (0, 0)
        frames = list(self.detections.keys())
        return (min(frames), max(frames))
    
    @property 
    def avg_confidence(self):
        if not self.detections:
            return 0
        return np.mean([d.conf for d in self.detections.values()])
    
    def get_position_at(self, frame: int) -> Optional[np.ndarray]:
        if frame in self.detections:
            return self.detections[frame].pos_3d
        return None

@dataclass
class GlobalTrack:
    track_id: int
    cls: str
    camera_tracks: Dict[str, int] = field(default_factory=dict)  # cam -> track_id
    nodes: List[dict] = field(default_factory=list)  # [{frame, pos, cameras}]
    is_stationary: bool = False

def iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def load_data(work_dir: Path):
    """Load detections."""
    det_path = work_dir / "detections_3d.json"
    with open(det_path) as f:
        data = json.load(f)
    
    dets = []
    for i, d in enumerate(data['detections']):
        det = Detection(
            frame=d['frame'],
            camera=d['camera'],
            cls=d['class'],
            conf=d['conf'],
            bbox=d['bbox'],
            pos_3d=np.array(d['pos_3d']),
            idx=i
        )
        dets.append(det)
    
    return dets, data['total_frames'], data['fps']

def build_per_camera_tracks(dets: List[Detection], total_frames: int) -> Dict[str, List[PerCameraTrack]]:
    """Build tracks for each camera independently using IoU association."""
    
    # Group by camera
    by_camera = defaultdict(list)
    for d in dets:
        by_camera[d.camera].append(d)
    
    all_tracks = {}
    next_track_id = 1
    
    for cam_id, cam_dets in by_camera.items():
        # Group by frame
        by_frame = defaultdict(list)
        for d in cam_dets:
            by_frame[d.frame].append(d)
        
        active_tracks = []  # List of (PerCameraTrack, last_det)
        finished_tracks = []
        
        for frame_idx in range(total_frames):
            current_dets = by_frame[frame_idx]
            
            # Age out old tracks (gap > 10 frames)
            still_active = []
            for track, last_det in active_tracks:
                if frame_idx - last_det.frame <= 10:
                    still_active.append((track, last_det))
                else:
                    finished_tracks.append(track)
            active_tracks = still_active
            
            if not current_dets:
                continue
            
            # Match current detections to active tracks using IoU
            if active_tracks:
                cost = np.zeros((len(current_dets), len(active_tracks)))
                
                for di, det in enumerate(current_dets):
                    for ti, (track, last_det) in enumerate(active_tracks):
                        # Class must match
                        if det.cls != track.cls:
                            cost[di, ti] = 100
                            continue
                        
                        # Compute IoU
                        iou_score = iou(det.bbox, last_det.bbox)
                        cost[di, ti] = 1 - iou_score
                
                rows, cols = linear_sum_assignment(cost)
                
                matched_dets = set()
                matched_tracks = set()
                
                new_active = []
                for di, ti in zip(rows, cols):
                    if cost[di, ti] < 1 - IOU_THRESHOLD:  # IoU > threshold
                        track, _ = active_tracks[ti]
                        det = current_dets[di]
                        track.detections[frame_idx] = det
                        new_active.append((track, det))
                        matched_dets.add(di)
                        matched_tracks.add(ti)
                
                # Keep unmatched tracks active for potential future matching
                for ti, (track, last_det) in enumerate(active_tracks):
                    if ti not in matched_tracks:
                        if frame_idx - last_det.frame <= 10:
                            new_active.append((track, last_det))
                        else:
                            finished_tracks.append(track)
                
                active_tracks = new_active
                
                # Create new tracks for unmatched detections
                for di, det in enumerate(current_dets):
                    if di not in matched_dets:
                        track = PerCameraTrack(
                            track_id=next_track_id,
                            camera=cam_id,
                            cls=det.cls
                        )
                        next_track_id += 1
                        track.detections[frame_idx] = det
                        active_tracks.append((track, det))
            else:
                # No active tracks, create new ones
                for det in current_dets:
                    track = PerCameraTrack(
                        track_id=next_track_id,
                        camera=cam_id,
                        cls=det.cls
                    )
                    next_track_id += 1
                    track.detections[frame_idx] = det
                    active_tracks.append((track, det))
        
        # Collect remaining active tracks
        for track, _ in active_tracks:
            finished_tracks.append(track)
        
        all_tracks[cam_id] = finished_tracks
    
    return all_tracks

def filter_per_camera_tracks(tracks: Dict[str, List[PerCameraTrack]], 
                             min_len: int = MIN_TRACK_LENGTH) -> Dict[str, List[PerCameraTrack]]:
    """Filter per-camera tracks by length and consistency."""
    filtered = {}
    
    for cam_id, cam_tracks in tracks.items():
        valid = []
        for track in cam_tracks:
            if track.length >= min_len:
                valid.append(track)
        filtered[cam_id] = valid
    
    return filtered

def associate_across_cameras(tracks: Dict[str, List[PerCameraTrack]], 
                            total_frames: int, fps: float) -> List[GlobalTrack]:
    """Associate per-camera tracks into global tracks using 3D proximity."""
    
    global_tracks = []
    next_global_id = 1
    
    # Flatten all tracks
    all_cam_tracks = []
    for cam_id, cam_tracks in tracks.items():
        for track in cam_tracks:
            all_cam_tracks.append(track)
    
    # Sort by earliest frame
    all_cam_tracks.sort(key=lambda t: t.frame_range[0])
    
    used_tracks = set()
    
    for track in all_cam_tracks:
        if track.track_id in used_tracks:
            continue
        
        # Start a new global track
        global_track = GlobalTrack(
            track_id=next_global_id,
            cls=track.cls
        )
        next_global_id += 1
        
        global_track.camera_tracks[track.camera] = track.track_id
        used_tracks.add(track.track_id)
        
        # Try to find overlapping tracks from other cameras
        for other_track in all_cam_tracks:
            if other_track.track_id in used_tracks:
                continue
            if other_track.camera == track.camera:
                continue
            if other_track.cls != track.cls:
                # Allow car<->truck confusion
                if not ({other_track.cls, track.cls} <= {'car', 'truck'}):
                    continue
            
            # Check temporal overlap
            r1 = track.frame_range
            r2 = other_track.frame_range
            overlap_start = max(r1[0], r2[0])
            overlap_end = min(r1[1], r2[1])
            
            if overlap_end < overlap_start:
                continue  # No temporal overlap
            
            # Check spatial proximity during overlap
            dists = []
            for f in range(overlap_start, overlap_end + 1):
                p1 = track.get_position_at(f)
                p2 = other_track.get_position_at(f)
                if p1 is not None and p2 is not None:
                    dist = np.linalg.norm(p1[:2] - p2[:2])
                    dists.append(dist)
            
            if dists and np.median(dists) < 3.0:  # Within 3m
                global_track.camera_tracks[other_track.camera] = other_track.track_id
                used_tracks.add(other_track.track_id)
        
        global_tracks.append(global_track)
    
    return global_tracks

def build_global_trajectories(global_tracks: List[GlobalTrack],
                              per_cam_tracks: Dict[str, List[PerCameraTrack]],
                              total_frames: int, fps: float) -> List[GlobalTrack]:
    """Build smooth trajectories for global tracks."""
    
    # Create lookup
    track_lookup = {}
    for cam_id, cam_tracks in per_cam_tracks.items():
        for track in cam_tracks:
            track_lookup[(cam_id, track.track_id)] = track
    
    for global_track in global_tracks:
        # Collect all positions across cameras
        frame_positions = defaultdict(list)  # frame -> [positions]
        frame_cameras = defaultdict(list)
        
        for cam_id, track_id in global_track.camera_tracks.items():
            key = (cam_id, track_id)
            if key not in track_lookup:
                continue
            
            cam_track = track_lookup[key]
            for frame, det in cam_track.detections.items():
                # Only use center-zone detections for position estimation
                if det.is_in_center_zone(IMG_W, IMG_H, margin=0.1):
                    frame_positions[frame].append(det.pos_3d)
                    frame_cameras[frame].append(cam_id)
        
        # Build nodes
        nodes = []
        for frame in sorted(frame_positions.keys()):
            positions = frame_positions[frame]
            cameras = frame_cameras[frame]
            
            if positions:
                # Average position (weighted by number of cameras)
                avg_pos = np.mean(positions, axis=0)
                nodes.append({
                    'frame': frame,
                    'pos': avg_pos.tolist(),
                    'cameras': cameras,
                    'conf': len(cameras) / len(global_track.camera_tracks)
                })
        
        # Interpolate gaps
        if len(nodes) >= 2:
            new_nodes = []
            for i in range(len(nodes) - 1):
                new_nodes.append(nodes[i])
                
                gap = nodes[i+1]['frame'] - nodes[i]['frame']
                if gap > 1 and gap <= 15:
                    p1 = np.array(nodes[i]['pos'])
                    p2 = np.array(nodes[i+1]['pos'])
                    for f in range(nodes[i]['frame'] + 1, nodes[i+1]['frame']):
                        alpha = (f - nodes[i]['frame']) / gap
                        interp_pos = p1 + alpha * (p2 - p1)
                        new_nodes.append({
                            'frame': f,
                            'pos': interp_pos.tolist(),
                            'cameras': [],
                            'conf': 0.5
                        })
            new_nodes.append(nodes[-1])
            nodes = sorted(new_nodes, key=lambda n: n['frame'])
        
        global_track.nodes = nodes
        
        # Detect stationary
        if len(nodes) >= 5:
            positions = np.array([n['pos'][:2] for n in nodes])
            variance = np.var(positions, axis=0).sum()
            global_track.is_stationary = variance < 1.0
            
            # Lock stationary to median
            if global_track.is_stationary:
                median = np.median(positions, axis=0)
                for n in nodes:
                    n['pos'][0] = float(median[0])
                    n['pos'][1] = float(median[1])
        
        # Smooth moving objects
        if not global_track.is_stationary and len(nodes) >= 5:
            for dim in range(3):
                vals = np.array([n['pos'][dim] for n in nodes])
                smoothed = uniform_filter1d(vals, size=5, mode='nearest')
                for i, n in enumerate(nodes):
                    n['pos'][dim] = float(smoothed[i])
    
    return global_tracks

def filter_global_tracks(tracks: List[GlobalTrack], min_length: int = 10) -> List[GlobalTrack]:
    """Final filtering of global tracks."""
    valid = []
    for track in tracks:
        if len(track.nodes) >= min_length:
            valid.append(track)
    return valid

def compute_orientations(track: GlobalTrack) -> List[float]:
    """Compute heading from velocity, with smoothing."""
    nodes = track.nodes
    n = len(nodes)
    
    if n < 2:
        return [0.0] * n
    
    positions = np.array([node['pos'][:2] for node in nodes])
    
    velocities = np.diff(positions, axis=0)
    velocities = np.vstack([velocities, velocities[-1]])
    
    yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
    
    if track.is_stationary:
        avg_yaw = np.mean(yaws)
        return [float(avg_yaw)] * n
    
    yaws_unwrap = np.unwrap(yaws)
    yaws_smooth = uniform_filter1d(yaws_unwrap, size=7, mode='nearest')
    
    return [float(y) for y in yaws_smooth]

def generate_scene(tracks: List[GlobalTrack], total_frames: int, fps: float) -> dict:
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
    print("BALANCED 4D TRACKER")
    print("=" * 60)
    
    print("\nLoading detections...")
    dets, total_frames, fps = load_data(work_dir)
    print(f"Loaded {len(dets)} detections")
    
    # Filter by confidence
    dets = [d for d in dets if d.conf >= MIN_CONFIDENCE]
    print(f"After confidence filter: {len(dets)}")
    
    print("\nBuilding per-camera tracks...")
    per_cam_tracks = build_per_camera_tracks(dets, total_frames)
    
    total_raw = sum(len(t) for t in per_cam_tracks.values())
    print(f"Raw per-camera tracks: {total_raw}")
    
    for cam_id, tracks in per_cam_tracks.items():
        print(f"  {cam_id}: {len(tracks)} tracks")
    
    print("\nFiltering per-camera tracks...")
    per_cam_tracks = filter_per_camera_tracks(per_cam_tracks, MIN_TRACK_LENGTH)
    
    total_filtered = sum(len(t) for t in per_cam_tracks.values())
    print(f"Filtered per-camera tracks: {total_filtered}")
    
    for cam_id, tracks in per_cam_tracks.items():
        if tracks:
            by_class = defaultdict(int)
            for t in tracks:
                by_class[t.cls] += 1
            print(f"  {cam_id}: {len(tracks)} - {dict(by_class)}")
    
    print("\nAssociating across cameras...")
    global_tracks = associate_across_cameras(per_cam_tracks, total_frames, fps)
    print(f"Global tracks (before trajectories): {len(global_tracks)}")
    
    print("\nBuilding trajectories...")
    global_tracks = build_global_trajectories(global_tracks, per_cam_tracks, total_frames, fps)
    
    print("\nFinal filtering...")
    global_tracks = filter_global_tracks(global_tracks, min_length=10)
    print(f"Final tracks: {len(global_tracks)}")
    
    # Stats
    stationary = sum(1 for t in global_tracks if t.is_stationary)
    moving = len(global_tracks) - stationary
    print(f"  Stationary: {stationary}, Moving: {moving}")
    
    by_class = defaultdict(int)
    for t in global_tracks:
        by_class[t.cls] += 1
    print("By class:")
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")
    
    print("\nGenerating scene...")
    scene = generate_scene(global_tracks, total_frames, fps)
    
    out_path = work_dir / "scene_4d.json"
    print(f"Saving to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    # Save tracks too
    tracks_out = {
        'total_frames': total_frames,
        'fps': fps,
        'tracks': [
            {
                'id': t.track_id,
                'class': t.cls,
                'is_stationary': bool(t.is_stationary),
                'num_cameras': len(t.camera_tracks),
                'length': len(t.nodes),
                'trajectory': t.nodes
            }
            for t in global_tracks
        ]
    }
    
    tracks_path = work_dir / "tracks_4d.json"
    with open(tracks_path, 'w') as f:
        json.dump(tracks_out, f, indent=2)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
