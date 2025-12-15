#!/usr/bin/env python3
"""
Multi-Camera Triangulation Tracker
===================================
Key fix: Use MULTIPLE cameras simultaneously to triangulate 3D positions.
- Single-camera depth estimation is unreliable
- Multi-camera views give accurate 3D via ray intersection
- Require 2+ cameras for all detections
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

# Strict thresholds for ~15 objects
MIN_CONFIDENCE = 0.5
MIN_CAMERAS_FOR_VALID = 2           # MUST be seen by 2+ cameras
SPATIAL_MATCH_THRESH = 3.0          # meters - distance threshold for matching
MIN_STATIC_FRAMES = 0.8             # 80% of frames for static object
MIN_DYNAMIC_TRACK_LEN = 40          # frames
MAX_GAP = 8                         # frames

class CameraProjector:
    """Projects pixels to rays and 3D points."""
    
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R = pose[:3, :3]
        self.t = pose[:3, 3]  # Camera center in world
        
    def pixel_to_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get ray origin and direction for a pixel."""
        ray_cam = self.K_inv @ np.array([u, v, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = self.R @ ray_cam
        return self.t.copy(), ray_world
    
    def project_to_ground(self, u: float, v: float, z: float = 0.0) -> Optional[np.ndarray]:
        """Project pixel to ground plane Z=z."""
        origin, direction = self.pixel_to_ray(u, v)
        if abs(direction[2]) < 1e-6:
            return None
        t = (z - origin[2]) / direction[2]
        if t < 0:
            return None
        return origin + t * direction
    
    def project_3d_to_2d(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D world point to 2D pixel."""
        R_w2c = self.R.T
        t_w2c = -R_w2c @ self.t
        p_cam = R_w2c @ point + t_w2c
        if p_cam[2] <= 0:
            return None
        p_img = self.K @ p_cam
        return p_img[:2] / p_cam[2]

def triangulate_from_rays(rays: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Triangulate 3D point from multiple ray observations.
    rays: list of (origin, direction) tuples
    Returns: 3D point that minimizes distance to all rays
    """
    if len(rays) < 2:
        # Fallback to ground projection for single ray
        origin, direction = rays[0]
        if abs(direction[2]) < 1e-6:
            return origin
        t = -origin[2] / direction[2]
        return origin + t * direction
    
    # Linear least squares triangulation
    # Minimize sum of squared distances to all rays
    # Point P lies on ray: P = O + t*D
    # Distance from point X to ray: ||(X-O) - ((X-O).D)D||
    
    # Build system: (I - D*D^T) * X = (I - D*D^T) * O
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for origin, direction in rays:
        d = direction / np.linalg.norm(direction)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ origin
    
    # Solve linear system
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to average of ground projections
        points = []
        for origin, direction in rays:
            if abs(direction[2]) > 1e-6:
                t = -origin[2] / direction[2]
                if t > 0:
                    points.append(origin + t * direction)
        if points:
            point = np.mean(points, axis=0)
        else:
            point = rays[0][0]
    
    # Clamp to ground level for vehicles
    point[2] = max(0, min(point[2], 2.0))
    
    return point

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter)

@dataclass
class MultiCamDetection:
    """A detection seen by multiple cameras."""
    frame: int
    cls: str
    pos_3d: np.ndarray
    cameras: List[str]
    avg_conf: float

@dataclass
class Track:
    track_id: int
    cls: str
    is_static: bool
    frames: Dict[int, np.ndarray] = field(default_factory=dict)  # frame -> pos

def detect_all_frames(video_dir: Path, cameras: dict, yolo: YOLO, 
                      total_frames: int) -> Dict[int, List[MultiCamDetection]]:
    """Detect objects across all cameras for each frame, triangulate."""
    
    print("\n=== DETECTING AND TRIANGULATING ===")
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    projectors = {cam_id: CameraProjector(cameras[cam_id]) for cam_id in cam_ids if cam_id in cameras}
    
    # Open all videos
    caps = {}
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            caps[cam_id] = cv2.VideoCapture(str(vpath))
    
    all_detections = {}
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        # Read all camera frames
        cam_frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                cam_frames[cam_id] = frame
        
        # Detect in all cameras
        cam_dets = {}  # cam_id -> list of {bbox, cls, conf}
        for cam_id, frame in cam_frames.items():
            results = yolo.predict(frame, conf=MIN_CONFIDENCE, iou=0.5, verbose=False,
                                  classes=list(VALID_CLASSES.keys()))
            
            dets = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())
                    cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                    
                    # Bottom center for ground projection
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = bbox[3]
                    
                    dets.append({
                        'bbox': bbox,
                        'cls': cls_name,
                        'conf': conf,
                        'bottom_center': (cx, cy)
                    })
            cam_dets[cam_id] = dets
        
        # Match detections across cameras by ground projection proximity
        # First project all to ground
        for cam_id, dets in cam_dets.items():
            proj = projectors.get(cam_id)
            if not proj:
                continue
            for det in dets:
                cx, cy = det['bottom_center']
                ground_pt = proj.project_to_ground(cx, cy, 0.0)
                if ground_pt is not None:
                    det['ground_pos'] = ground_pt
                    det['ray'] = proj.pixel_to_ray(cx, cy)
                else:
                    det['ground_pos'] = None
                    det['ray'] = None
        
        # Group detections by class and spatial proximity
        all_dets_flat = []
        for cam_id, dets in cam_dets.items():
            for det in dets:
                if det.get('ground_pos') is not None:
                    det['camera'] = cam_id
                    all_dets_flat.append(det)
        
        # Cluster by class and 3D position
        frame_multi_dets = []
        used = set()
        
        for i, det1 in enumerate(all_dets_flat):
            if i in used:
                continue
            
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(all_dets_flat):
                if j in used or j <= i:
                    continue
                
                # Same class or car/truck
                if det1['cls'] != det2['cls']:
                    if not ({det1['cls'], det2['cls']} <= {'car', 'truck'}):
                        continue
                
                # Spatial proximity
                dist = np.linalg.norm(det1['ground_pos'][:2] - det2['ground_pos'][:2])
                if dist < SPATIAL_MATCH_THRESH:
                    cluster.append(det2)
                    used.add(j)
            
            # Only accept if 2+ cameras
            cameras_in_cluster = set(d['camera'] for d in cluster)
            if len(cameras_in_cluster) >= MIN_CAMERAS_FOR_VALID:
                # Triangulate from rays
                rays = [d['ray'] for d in cluster if d['ray'] is not None]
                if len(rays) >= 2:
                    pos_3d = triangulate_from_rays(rays)
                else:
                    pos_3d = np.mean([d['ground_pos'] for d in cluster], axis=0)
                
                # Check bounds
                if abs(pos_3d[0]) > 35 or abs(pos_3d[1]) > 35:
                    continue
                
                # Majority class
                classes = [d['cls'] for d in cluster]
                cls = max(set(classes), key=classes.count)
                
                avg_conf = np.mean([d['conf'] for d in cluster])
                
                frame_multi_dets.append(MultiCamDetection(
                    frame=frame_idx,
                    cls=cls,
                    pos_3d=pos_3d,
                    cameras=list(cameras_in_cluster),
                    avg_conf=avg_conf
                ))
        
        if frame_multi_dets:
            all_detections[frame_idx] = frame_multi_dets
    
    for cap in caps.values():
        cap.release()
    
    total = sum(len(d) for d in all_detections.values())
    print(f"  Total multi-camera detections: {total}")
    
    return all_detections

def identify_static_objects(all_detections: Dict[int, List[MultiCamDetection]], 
                            total_frames: int) -> List[Track]:
    """Identify static objects (present in most frames at same position)."""
    
    print("\n=== IDENTIFYING STATIC OBJECTS ===")
    
    # Collect all unique positions
    # Group by spatial proximity and track occurrence count
    
    position_groups = []  # list of {positions: [], frames: [], cls: str}
    
    for frame_idx, dets in sorted(all_detections.items()):
        for det in dets:
            if det.cls not in VEHICLE_CLASSES:
                continue
            
            # Try to match to existing group
            matched = False
            for group in position_groups:
                if det.cls != group['cls']:
                    if not ({det.cls, group['cls']} <= {'car', 'truck'}):
                        continue
                
                # Check distance to group centroid
                centroid = np.mean(group['positions'], axis=0)
                dist = np.linalg.norm(det.pos_3d[:2] - centroid[:2])
                
                if dist < 2.0:  # Within 2m = same static object
                    group['positions'].append(det.pos_3d)
                    group['frames'].append(frame_idx)
                    matched = True
                    break
            
            if not matched:
                position_groups.append({
                    'positions': [det.pos_3d],
                    'frames': [frame_idx],
                    'cls': det.cls
                })
    
    # Filter to truly static objects (present in many frames)
    min_frames = int(total_frames * MIN_STATIC_FRAMES)
    
    static_tracks = []
    next_id = 1
    
    for group in position_groups:
        unique_frames = len(set(group['frames']))
        if unique_frames >= min_frames:
            # Check position variance (should be low for static)
            positions = np.array(group['positions'])
            variance = np.var(positions[:, :2], axis=0).sum()
            
            if variance < 2.0:  # Low variance = truly static
                median_pos = np.median(positions, axis=0)
                
                track = Track(
                    track_id=next_id,
                    cls=group['cls'],
                    is_static=True
                )
                
                # Add to all frames
                for f in range(total_frames):
                    track.frames[f] = median_pos.copy()
                
                static_tracks.append(track)
                next_id += 1
    
    print(f"  Found {len(static_tracks)} static objects")
    
    return static_tracks

def track_dynamic_objects(all_detections: Dict[int, List[MultiCamDetection]],
                          static_tracks: List[Track],
                          total_frames: int) -> List[Track]:
    """Track dynamic objects, excluding static ones."""
    
    print("\n=== TRACKING DYNAMIC OBJECTS ===")
    
    # Get static positions
    static_positions = []
    for track in static_tracks:
        if track.frames:
            pos = list(track.frames.values())[0]
            static_positions.append(pos[:2])
    
    # Filter detections that are NOT near static positions
    dynamic_dets = {}
    
    for frame_idx, dets in all_detections.items():
        filtered = []
        for det in dets:
            # Check if near any static object
            is_static = False
            for static_pos in static_positions:
                dist = np.linalg.norm(det.pos_3d[:2] - static_pos)
                if dist < 2.5:
                    is_static = True
                    break
            
            if not is_static:
                filtered.append(det)
        
        if filtered:
            dynamic_dets[frame_idx] = filtered
    
    # Track dynamic objects
    tracks = []
    next_id = 100  # Start from 100 to distinguish from static
    active: List[Track] = []
    
    for frame_idx in tqdm(range(total_frames), desc="Tracking dynamic"):
        current = dynamic_dets.get(frame_idx, [])
        
        # Age out old tracks
        still_active = []
        for track in active:
            last_frame = max(track.frames.keys())
            if frame_idx - last_frame <= MAX_GAP:
                still_active.append(track)
            else:
                tracks.append(track)
        active = still_active
        
        if not current:
            continue
        
        if active:
            cost = np.zeros((len(current), len(active)))
            
            for di, det in enumerate(current):
                for ti, track in enumerate(active):
                    if track.cls != det.cls:
                        if not ({track.cls, det.cls} <= {'car', 'truck'}):
                            cost[di, ti] = 1000
                            continue
                    
                    last_frame = max(track.frames.keys())
                    last_pos = track.frames[last_frame]
                    dist = np.linalg.norm(det.pos_3d[:2] - last_pos[:2])
                    cost[di, ti] = dist
            
            rows, cols = linear_sum_assignment(cost)
            
            matched = set()
            for di, ti in zip(rows, cols):
                if cost[di, ti] < 6.0:
                    active[ti].frames[frame_idx] = current[di].pos_3d.copy()
                    matched.add(di)
            
            for di, det in enumerate(current):
                if di not in matched:
                    track = Track(track_id=next_id, cls=det.cls, is_static=False)
                    next_id += 1
                    track.frames[frame_idx] = det.pos_3d.copy()
                    active.append(track)
        else:
            for det in current:
                track = Track(track_id=next_id, cls=det.cls, is_static=False)
                next_id += 1
                track.frames[frame_idx] = det.pos_3d.copy()
                active.append(track)
    
    tracks.extend(active)
    
    # Filter short tracks
    tracks = [t for t in tracks if len(t.frames) >= MIN_DYNAMIC_TRACK_LEN]
    
    print(f"  Found {len(tracks)} dynamic tracks")
    
    return tracks

def generate_scene(static_tracks: List[Track], dynamic_tracks: List[Track],
                   total_frames: int, fps: float) -> dict:
    """Generate final scene."""
    
    DIMS = {
        'car': [4.5, 1.8, 1.5],
        'truck': [7.0, 2.4, 2.8],
        'bus': [10.0, 2.5, 3.0],
        'motorcycle': [2.0, 0.8, 1.3],
        'bicycle': [1.8, 0.5, 1.4],
        'person': [0.5, 0.5, 1.7]
    }
    
    scene = {
        'total_frames': total_frames,
        'fps': fps,
        'objects': {},
        'frames': {}
    }
    
    # Static objects (gray)
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
    
    # Dynamic objects (colored)
    COLORS = {
        'car': [0, 0, 255],
        'truck': [255, 0, 200],
        'bus': [0, 200, 255],
        'motorcycle': [255, 128, 0],
        'bicycle': [0, 255, 128],
        'person': [0, 255, 0]
    }
    
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
        
        # Compute orientations
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
    print("MULTI-CAMERA TRIANGULATION TRACKER")
    print("- Requires 2+ camera views for valid detection")
    print("- Ray intersection for accurate 3D")
    print("=" * 70)
    
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    video_dir = base_dir / "StreetAware-sample"
    
    # Get frame count
    test_cap = cv2.VideoCapture(str(video_dir / "s1-left.mp4"))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS)
    test_cap.release()
    
    print(f"\nTotal frames: {total_frames}, FPS: {fps:.1f}")
    
    # Initialize YOLO
    print("\nLoading YOLO...")
    yolo = YOLO('yolov8x.pt')
    
    # Stage 1: Detect all frames with multi-camera validation
    all_detections = detect_all_frames(video_dir, cameras, yolo, total_frames)
    
    # Stage 2: Identify static objects
    static_tracks = identify_static_objects(all_detections, total_frames)
    
    # Stage 3: Track dynamic objects
    dynamic_tracks = track_dynamic_objects(all_detections, static_tracks, total_frames)
    
    # Stage 4: Generate scene
    print("\n=== GENERATING SCENE ===")
    scene = generate_scene(static_tracks, dynamic_tracks, total_frames, fps)
    
    print(f"  Static: {len(static_tracks)}, Dynamic: {len(dynamic_tracks)}")
    print(f"  Total: {len(static_tracks) + len(dynamic_tracks)} objects")
    
    out_path = work_dir / "scene_4d.json"
    print(f"\nSaving to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
