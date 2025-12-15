#!/usr/bin/env python3
"""
Pass2 Dynamic V4 - Complete Pipeline
=====================================
Based on cross_camera_tracker.py with improvements:

1. Static objects from background images (not video tracking)
2. Per-camera tracking with IoU matching
3. Cross-camera association using bbox similarity
4. Proper road/curb constraints:
   - Curbs: only people/bicycles
   - Road: vehicles
   - Static vehicles: NOT in middle of road
5. Simple 2D visualization
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}
PEOPLE_CLASSES = {'person', 'bicycle'}

# =============================================================================
# CAMERA PROJECTION
# =============================================================================

class CameraProjector:
    def __init__(self, params: dict, image_size: Tuple[int, int]):
        self.K = np.array(params['K']).reshape(3, 3)
        pose = np.array(params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.cam_pos = pose[:3, 3]
        self.image_size = image_size
    
    def project_to_ground(self, pixel: np.ndarray, ground_z: float = 0.0) -> Optional[np.ndarray]:
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        ray_cam = np.linalg.inv(self.K) @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = self.R_c2w @ ray_cam
        
        if abs(ray_world[2]) < 1e-6:
            return None
        
        s = (ground_z - self.cam_pos[2]) / ray_world[2]
        if s < 0:
            return None
        
        point = self.cam_pos + s * ray_world
        if np.linalg.norm(point[:2]) > 35:  # Max 35m from origin
            return None
        
        return point

# =============================================================================
# GROUND MASK
# =============================================================================

class GroundMask:
    def __init__(self, mask_dir: Path):
        road_path = mask_dir / "road_grid.npy"
        curb_path = mask_dir / "curb_grid.npy"
        info_path = mask_dir / "grid_info.json"
        
        if not road_path.exists():
            self.road_grid = None
            self.curb_grid = None
            return
        
        self.road_grid = np.load(road_path)
        self.curb_grid = np.load(curb_path)
        with open(info_path) as f:
            self.info = json.load(f)
    
    def is_on_road(self, pos) -> bool:
        if self.road_grid is None:
            return True
        gx = int((pos[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos[1] - self.info['origin'][1]) / self.info['resolution'])
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.road_grid[gy, gx])
        return False
    
    def is_on_curb(self, pos) -> bool:
        if self.curb_grid is None:
            return False
        gx = int((pos[0] - self.info['origin'][0]) / self.info['resolution'])
        gy = int((pos[1] - self.info['origin'][1]) / self.info['resolution'])
        if 0 <= gx < self.info['dim'] and 0 <= gy < self.info['dim']:
            return bool(self.curb_grid[gy, gx])
        return False

# =============================================================================
# STATIC OBJECT DETECTION (from backgrounds)
# =============================================================================

def matches_l_shape(x, y):
    """Check if position matches expected pipe-shape =||= parking pattern.
    
    Pipe shape has roads on 4 sides (from point cloud analysis):
    - Bottom edge: Y < -8, X between -6 and 16
    - Top edge: Y > 12, X between -10 and 14
    - Left edge: X < -8, Y between -10 and 10
    - Right edge: X > 10, Y between -12 and 14
    """
    # Bottom edge
    if y < -8 and -6 < x < 16:
        return True, "bottom"
    # Top edge
    if y > 12 and -10 < x < 14:
        return True, "top"
    # Left edge
    if x < -8 and -10 < y < 10:
        return True, "left"
    # Right edge
    if x > 10 and -12 < y < 14:
        return True, "right"
    
    return False, None

def detect_static_objects(bg_dir: Path, cameras: dict, ground: GroundMask,
                          yolo: YOLO) -> List[dict]:
    """Detect parked vehicles from static background images.
    
    Key improvements:
    1. Better clustering using DBSCAN
    2. Orientation from bbox aspect ratio (horizontal vs vertical parking)
    3. Only keep cars near road edges (not in middle)
    """
    print("\n=== DETECTING STATIC OBJECTS FROM BACKGROUNDS ===")
    
    from sklearn.cluster import DBSCAN
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    # Collect all high-confidence detections
    all_dets = []
    
    for cam_id in cam_ids:
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists() or cam_id not in cameras:
            continue
        
        print(f"  {cam_id}...")
        
        bg_img = cv2.imread(str(bg_path))
        h, w = bg_img.shape[:2]
        
        K = np.array(cameras[cam_id]['K']).reshape(3, 3)
        pose = np.array(cameras[cam_id]['pose_c2w'])
        R_c2w = pose[:3, :3]
        cam_pos = pose[:3, 3]
        
        results = yolo.predict(bg_img, conf=0.4, verbose=False,
                              classes=[2, 3, 5, 7])  # vehicles only
        
        if results[0].boxes is None:
            continue
        
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu())
            cls_name = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}[cls_id]
            conf = float(box.conf[0].cpu())
            
            # Skip low confidence
            if conf < 0.45:
                continue
            
            # Get bottom center
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]
            
            # Bbox dimensions for orientation estimate
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            aspect = bw / bh if bh > 0 else 1.0
            
            # Estimate orientation from aspect ratio:
            # aspect > 1.5 = car seen from side (horizontal parking)
            # aspect < 0.8 = car seen from front/back
            if aspect > 1.5:
                view_angle = 0  # Side view - car is horizontal
            elif aspect < 0.8:
                view_angle = np.pi / 2  # Front view - car is vertical
            else:
                view_angle = np.pi / 4  # Diagonal
            
            # Project to 3D
            ray_cam = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            ray_world = R_c2w @ ray_cam
            
            if abs(ray_world[2]) < 1e-6:
                continue
            
            s = -cam_pos[2] / ray_world[2]
            if s < 0:
                continue
            
            pos = cam_pos + s * ray_world
            
            # Filter by distance
            if np.linalg.norm(pos[:2]) > 30:
                continue
            
            all_dets.append({
                'pos': pos[:2].copy(),
                'cls': cls_name,
                'conf': conf,
                'camera': cam_id,
                'aspect': aspect,
                'view_angle': view_angle
            })
    
    print(f"  Total raw detections: {len(all_dets)}")
    
    if len(all_dets) < 2:
        return []
    
    # DBSCAN clustering to merge same car from different cameras
    positions = np.array([d['pos'] for d in all_dets])
    db = DBSCAN(eps=2.5, min_samples=1).fit(positions)
    labels = db.labels_
    
    # Build clusters
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            clusters[label].append(all_dets[i])
    
    print(f"  Clusters: {len(clusters)}")
    
    # Process each cluster
    static_objects = []
    
    for cluster_id, cluster in clusters.items():
        # Average position
        avg_pos = np.mean([d['pos'] for d in cluster], axis=0)
        
        # Best confidence and class
        best_det = max(cluster, key=lambda x: x['conf'])
        cls_name = best_det['cls']
        best_conf = best_det['conf']
        
        # Number of cameras seeing this object
        num_cams = len(set(d['camera'] for d in cluster))
        
        # Check if matches L-shape pattern
        x, y = avg_pos
        matches, edge = matches_l_shape(x, y)
        if not matches:
            continue  # Skip objects outside L-shape
        
        # Orientation: Top/Bottom = Horizontal (0), Left/Right = Vertical (90)
        if edge in ("bottom", "top"):
            yaw = 0.0  # Horizontal parking (0°)
        else:  # left, right
            yaw = np.pi / 2  # Vertical parking (90°)
        
        static_objects.append({
            'pos': avg_pos,
            'cls': cls_name,
            'conf': best_conf,
            'num_cameras': num_cams,
            'yaw': yaw,
            'edge': edge
        })
    
    # Sort by number of cameras, then confidence
    static_objects.sort(key=lambda x: (x['num_cameras'], x['conf']), reverse=True)
    
    # Count by edge
    from collections import Counter
    edge_counts = Counter(o['edge'] for o in static_objects)
    
    print(f"  Pipe-shape static: {len(static_objects)} ({dict(edge_counts)})")
    for i, obj in enumerate(static_objects[:15]):
        print(f"    {i+1}: pos=({obj['pos'][0]:.1f}, {obj['pos'][1]:.1f}), edge={obj['edge']}, cams={obj['num_cameras']}")
    
    return static_objects

# =============================================================================
# DYNAMIC TRACKING (per-camera then merge)
# =============================================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

def track_single_camera(video_path: Path, cam_id: str, projector: CameraProjector,
                        yolo: YOLO) -> Dict[int, dict]:
    """Track objects in a single camera using IoU matching."""
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracks = {}  # {track_id: {'class': str, 'frames': {frame: (bbox, pos_3d, conf)}}}
    next_id = 1
    active = {}  # {track_id: (last_bbox, last_frame, velocity)}
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        results = yolo.predict(frame, conf=0.35, verbose=False,
                              classes=list(VALID_CLASSES.keys()))
        
        dets = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                bbox = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                conf = float(box.conf[0].cpu())
                
                bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                pos = projector.project_to_ground(bottom_center)
                
                if pos is not None:
                    dets.append((bbox, cls_name, conf, pos))
        
        # Match to active tracks
        matched_dets = set()
        
        if active and dets:
            track_ids = list(active.keys())
            costs = np.zeros((len(track_ids), len(dets)))
            
            for ti, tid in enumerate(track_ids):
                last_bbox, last_frame, velocity = active[tid]
                gap = frame_idx - last_frame
                
                # Predict bbox with velocity
                if velocity is not None:
                    pred_bbox = last_bbox.copy()
                    pred_bbox[0] += velocity[0] * gap
                    pred_bbox[1] += velocity[1] * gap
                    pred_bbox[2] += velocity[0] * gap
                    pred_bbox[3] += velocity[1] * gap
                else:
                    pred_bbox = last_bbox
                
                for di, (det_bbox, _, _, _) in enumerate(dets):
                    iou = compute_iou(pred_bbox, det_bbox)
                    costs[ti, di] = 1 - iou
            
            # Greedy matching
            while True:
                if costs.size == 0:
                    break
                min_idx = np.unravel_index(np.argmin(costs), costs.shape)
                min_cost = costs[min_idx]
                
                if min_cost > 0.7:
                    break
                
                ti, di = min_idx
                tid = track_ids[ti]
                det_bbox, det_cls, det_conf, det_pos = dets[di]
                
                tracks[tid]['frames'][frame_idx] = (det_bbox, det_pos, det_conf)
                
                # Update velocity
                last_bbox, last_frame, old_vel = active[tid]
                if frame_idx > last_frame:
                    dt = frame_idx - last_frame
                    dx = (det_bbox[0] + det_bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2
                    dy = (det_bbox[1] + det_bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2
                    new_vel = np.array([dx / dt, dy / dt])
                    if old_vel is not None:
                        new_vel = 0.7 * old_vel + 0.3 * new_vel
                    active[tid] = (det_bbox, frame_idx, new_vel)
                else:
                    active[tid] = (det_bbox, frame_idx, old_vel)
                
                matched_dets.add(di)
                costs[ti, :] = np.inf
                costs[:, di] = np.inf
        
        # Create new tracks
        for di, (det_bbox, det_cls, det_conf, det_pos) in enumerate(dets):
            if di not in matched_dets:
                tid = next_id
                next_id += 1
                tracks[tid] = {
                    'class': det_cls,
                    'frames': {frame_idx: (det_bbox, det_pos, det_conf)}
                }
                active[tid] = (det_bbox, frame_idx, None)
        
        # Remove stale tracks
        stale = [tid for tid, (_, lf, _) in active.items() if frame_idx - lf > 30]
        for tid in stale:
            del active[tid]
    
    cap.release()
    
    # Filter short tracks
    tracks = {tid: data for tid, data in tracks.items() if len(data['frames']) >= 10}
    
    return tracks

def run_dynamic_tracking(video_dir: Path, cameras: dict, ground: GroundMask,
                         yolo: YOLO) -> List[dict]:
    """Run per-camera tracking and merge across cameras."""
    print("\n=== DYNAMIC TRACKING (PER-CAMERA) ===")
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    # Get video info
    test_cap = cv2.VideoCapture(str(video_dir / "s1-left.mp4"))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    test_cap.release()
    
    # Phase 1: Per-camera tracking
    per_cam_tracks = {}
    
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if not vpath.exists() or cam_id not in cameras:
            continue
        
        print(f"\n  {cam_id}...")
        proj = CameraProjector(cameras[cam_id], (width, height))
        tracks = track_single_camera(vpath, cam_id, proj, yolo)
        per_cam_tracks[cam_id] = tracks
        print(f"    {len(tracks)} tracks")
    
    # Phase 2: Cross-camera association (simple: merge within pairs)
    print("\n  Merging across camera pairs...")
    
    camera_pairs = [
        ('s1-left', 's1-right'),
        ('s2-left', 's2-right'),
        ('s3-left', 's3-right'),
        ('s4-left', 's4-right'),
    ]
    
    global_tracks = []
    local_to_global = {}  # (cam_id, local_id) -> global_idx
    
    for left_cam, right_cam in camera_pairs:
        if left_cam not in per_cam_tracks:
            continue
        
        left_tracks = per_cam_tracks.get(left_cam, {})
        right_tracks = per_cam_tracks.get(right_cam, {})
        
        matched_right = set()
        
        for left_id, left_data in left_tracks.items():
            left_frames = set(left_data['frames'].keys())
            
            best_match = None
            best_overlap = 0
            
            for right_id, right_data in right_tracks.items():
                if right_id in matched_right:
                    continue
                if right_data['class'] != left_data['class']:
                    continue
                
                right_frames = set(right_data['frames'].keys())
                overlap = len(left_frames & right_frames)
                
                if overlap > best_overlap and overlap >= 10:
                    # Check 3D position similarity
                    common = left_frames & right_frames
                    dists = []
                    for f in list(common)[:30]:
                        lp = left_data['frames'][f][1][:2]
                        rp = right_data['frames'][f][1][:2]
                        dists.append(np.linalg.norm(lp - rp))
                    
                    if np.median(dists) < 5.0:
                        best_match = right_id
                        best_overlap = overlap
            
            # Create global track
            gidx = len(global_tracks)
            
            # Merge positions from both cameras
            merged_frames = {}
            for f, (bbox, pos, conf) in left_data['frames'].items():
                merged_frames[f] = [pos[:2]]
            
            if best_match is not None:
                matched_right.add(best_match)
                for f, (bbox, pos, conf) in right_tracks[best_match]['frames'].items():
                    if f in merged_frames:
                        merged_frames[f].append(pos[:2])
                    else:
                        merged_frames[f] = [pos[:2]]
            
            # Average positions
            final_frames = {}
            for f, positions in merged_frames.items():
                final_frames[f] = np.mean(positions, axis=0)
            
            # Compute travel distance
            frames_list = sorted(final_frames.keys())
            if len(frames_list) >= 2:
                pts = np.array([final_frames[f] for f in frames_list])
                travel = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            else:
                travel = 0.0
            
            # Only keep if significant travel (moving vehicle)
            if travel >= 4.0:
                # Check if mainly on road
                center = np.mean(list(final_frames.values()), axis=0)
                if ground.is_on_road(center):
                    global_tracks.append({
                        'class': left_data['class'],
                        'frames': final_frames,
                        'travel': travel
                    })
        
        # Add unmatched right tracks
        for right_id, right_data in right_tracks.items():
            if right_id in matched_right:
                continue
            
            frames_data = {f: pos[:2] for f, (_, pos, _) in right_data['frames'].items()}
            frames_list = sorted(frames_data.keys())
            
            if len(frames_list) >= 2:
                pts = np.array([frames_data[f] for f in frames_list])
                travel = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            else:
                travel = 0.0
            
            if travel >= 4.0:
                center = np.mean(list(frames_data.values()), axis=0)
                if ground.is_on_road(center):
                    global_tracks.append({
                        'class': right_data['class'],
                        'frames': frames_data,
                        'travel': travel
                    })
    
    # Sort by travel and take top
    global_tracks.sort(key=lambda x: x['travel'], reverse=True)
    
    print(f"\n  Total global dynamic tracks: {len(global_tracks)}")
    
    return global_tracks, total_frames

# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def smooth_track(frames_data: dict) -> dict:
    """Apply Savitzky-Golay smoothing to track positions."""
    frames_list = sorted(frames_data.keys())
    if len(frames_list) < 5:
        return frames_data
    
    positions = np.array([frames_data[f] for f in frames_list])
    
    window = min(11, (len(positions) // 2) * 2 - 1)
    if window >= 5:
        for dim in range(2):
            positions[:, dim] = savgol_filter(positions[:, dim], window, 2, mode='nearest')
    
    return {f: positions[i] for i, f in enumerate(frames_list)}

def fill_gaps(frames_data: dict, max_gap: int = 15) -> dict:
    """Fill gaps in track by interpolation."""
    frames_list = sorted(frames_data.keys())
    result = dict(frames_data)
    
    for i in range(len(frames_list) - 1):
        gap = frames_list[i+1] - frames_list[i]
        if gap > 1 and gap <= max_gap:
            p1 = frames_data[frames_list[i]]
            p2 = frames_data[frames_list[i+1]]
            for f in range(frames_list[i] + 1, frames_list[i+1]):
                a = (f - frames_list[i]) / gap
                result[f] = p1 + a * (p2 - p1)
    
    return result

def generate_scene_json(static_objects: List[dict], dynamic_tracks: List[dict],
                        total_frames: int, fps: float) -> dict:
    """Generate scene_4d.json format."""
    
    DIMS = {
        'car': [4.5, 1.8, 1.5], 'truck': [7.0, 2.4, 2.8],
        'bus': [10.0, 2.5, 3.0], 'motorcycle': [2.0, 0.8, 1.3],
        'bicycle': [1.8, 0.5, 1.4], 'person': [0.5, 0.5, 1.7]
    }
    COLORS = {
        'car': [0, 0, 255], 'truck': [255, 0, 200], 'bus': [200, 0, 200],
        'person': [0, 255, 0], 'bicycle': [0, 255, 128]
    }
    
    scene = {
        'total_frames': total_frames,
        'fps': fps,
        'objects': {},
        'frames': {}
    }
    
    # Static objects
    for i, obj in enumerate(static_objects[:15]):  # Max 15 static
        oid = f"S{i+1}"
        scene['objects'][oid] = {
            'class': obj['cls'],
            'dims': DIMS.get(obj['cls'], DIMS['car']),
            'color': [128, 128, 128],
            'is_stationary': True
        }
        
        # Use yaw from detection (based on position and aspect ratio)
        yaw = obj.get('yaw', 0.0)
        quat = R.from_euler('z', float(yaw)).as_quat().tolist()
        pos = [float(obj['pos'][0]), float(obj['pos'][1]), 0.0]
        
        for f in range(total_frames):
            fk = str(f)
            if fk not in scene['frames']:
                scene['frames'][fk] = []
            scene['frames'][fk].append({'id': oid, 'pos': pos, 'rot': quat, 'conf': 1.0})
    
    # Dynamic objects
    for i, track in enumerate(dynamic_tracks[:15]):  # Max 15 dynamic
        oid = f"D{i+1}"
        scene['objects'][oid] = {
            'class': track['class'],
            'dims': DIMS.get(track['class'], DIMS['car']),
            'color': COLORS.get(track['class'], [255, 255, 255]),
            'is_stationary': False
        }
        
        # Smooth and fill
        frames_data = smooth_track(track['frames'])
        frames_data = fill_gaps(frames_data)
        
        # Get direction
        frames_list = sorted(frames_data.keys())
        positions = np.array([frames_data[f] for f in frames_list])
        
        if len(positions) >= 2:
            delta = positions[-1] - positions[0]
            yaw = np.arctan2(delta[1], delta[0])
        else:
            yaw = 0.0
        
        quat = R.from_euler('z', float(yaw)).as_quat().tolist()
        
        for f, pos in frames_data.items():
            fk = str(f)
            if fk not in scene['frames']:
                scene['frames'][fk] = []
            scene['frames'][fk].append({
                'id': oid,
                'pos': [float(pos[0]), float(pos[1]), 0.0],
                'rot': quat,
                'conf': 0.9
            })
    
    return scene

# =============================================================================
# MAIN
# =============================================================================

def main():
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "outputs" / "pass2_dynamic_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PASS2 DYNAMIC V4 - Per-Camera Tracking + Static from Backgrounds")
    print("=" * 70)
    
    # Load config
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    ground = GroundMask(base_dir / "outputs" / "pass1_static" / "ground_masks")
    
    print("\nLoading YOLO...")
    yolo = YOLO('yolov8x.pt')
    
    # Video info
    cap = cv2.VideoCapture(str(base_dir / "StreetAware-sample/s1-left.mp4"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Frames: {total_frames}, FPS: {fps:.1f}")
    
    # Detect static objects from backgrounds
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    static_objects = detect_static_objects(bg_dir, cameras, ground, yolo)
    
    # Track dynamic objects
    video_dir = base_dir / "StreetAware-sample"
    dynamic_tracks, _ = run_dynamic_tracking(video_dir, cameras, ground, yolo)
    
    # Generate scene
    print("\n=== GENERATING SCENE ===")
    scene = generate_scene_json(static_objects, dynamic_tracks, total_frames, fps)
    
    static_count = sum(1 for o in scene['objects'].values() if o['is_stationary'])
    dynamic_count = len(scene['objects']) - static_count
    print(f"  Static: {static_count}, Dynamic: {dynamic_count}")
    
    # Save
    scene_path = out_dir / "scene_4d.json"
    with open(scene_path, 'w') as f:
        json.dump(scene, f, indent=2)
    print(f"\nSaved: {scene_path}")

if __name__ == "__main__":
    main()
