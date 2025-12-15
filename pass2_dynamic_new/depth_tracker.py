#!/usr/bin/env python3
"""
Depth-Aware 4D Tracking Pipeline
=================================
Key improvements:
1. Learn STATIC objects from static backgrounds (parked cars are fixed)
2. Use Depth Anything V2 for accurate 3D position estimation
3. Separate static vs dynamic clearly
4. Handle occlusion by temporarily stopping objects
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

# Constants
VALID_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}
MIN_CONFIDENCE = 0.4
IOU_MATCH_THRESH = 0.4  # For matching detections to static objects
MIN_TRACK_FRAMES = 30
MAX_GAP_FRAMES = 10
CLUSTER_DIST = 4.0  # meters

class DepthEstimator:
    """Depth Anything V2 for monocular depth estimation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Depth Anything V2...")
        self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
        
    def estimate_depth(self, img_rgb: np.ndarray) -> np.ndarray:
        """Returns depth map (H, W) normalized to 0-1 range."""
        inputs = self.processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
        
        # Interpolate to original size
        h, w = img_rgb.shape[:2]
        depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
        depth = depth.squeeze().cpu().numpy()
        
        # Normalize to 0-1 (inverse depth, so larger = closer)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

class CameraProjector:
    """Projects 2D pixels to 3D using depth."""
    
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        self.K_inv = np.linalg.inv(self.K)
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]
        
        # Estimate focal length for depth scaling
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
    def to_3d_with_depth(self, u: float, v: float, depth_value: float, 
                         max_depth: float = 50.0) -> np.ndarray:
        """
        Project pixel (u, v) to 3D using depth value.
        depth_value: normalized depth (0-1 where 1 is closest)
        """
        # Convert normalized inverse depth to metric depth
        # Assume depth 1.0 = 2m, depth 0.1 = 20m (rough calibration)
        metric_depth = 2.0 / (depth_value + 0.04)  # Avoid division by zero
        metric_depth = min(metric_depth, max_depth)
        
        # Back-project to camera coordinates
        x_cam = (u - self.cx) * metric_depth / self.fx
        y_cam = (v - self.cy) * metric_depth / self.fy
        z_cam = metric_depth
        
        p_cam = np.array([x_cam, y_cam, z_cam])
        
        # Transform to world
        p_world = self.R_c2w @ p_cam + self.t_c2w
        
        # Clamp Z to ground level
        p_world[2] = max(0, min(p_world[2], 3.0))  # Vehicle height
        
        return p_world
    
    def project_to_image(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D world point to image pixel."""
        R_w2c = self.R_c2w.T
        t_w2c = -R_w2c @ self.t_c2w
        p_cam = R_w2c @ point_3d + t_w2c
        if p_cam[2] <= 0:
            return None
        p_img = self.K @ p_cam
        return p_img[:2] / p_cam[2]

@dataclass
class StaticObject:
    """A parked/static object detected in background."""
    obj_id: int
    cls: str
    camera: str
    bbox: np.ndarray  # [x1, y1, x2, y2]
    pos_3d: np.ndarray
    conf: float

@dataclass
class DynamicTrack:
    """A moving object tracked over time."""
    track_id: int
    cls: str
    frames: Dict[int, dict] = field(default_factory=dict)  # frame -> {pos, cameras, conf}
    is_temporarily_stopped: bool = False
    stopped_position: Optional[np.ndarray] = None
    
    @property
    def length(self):
        return len(self.frames)

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

def detect_static_objects(bg_dir: Path, cameras: dict, yolo: YOLO, 
                          depth_estimator: DepthEstimator) -> Dict[str, List[StaticObject]]:
    """Detect parked vehicles in static background images."""
    
    print("\n=== DETECTING STATIC OBJECTS FROM BACKGROUNDS ===")
    
    static_objects = {}
    next_id = 1
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    for cam_id in cam_ids:
        if cam_id not in cameras:
            continue
            
        bg_path = bg_dir / f"{cam_id}_bg.png"
        if not bg_path.exists():
            continue
        
        print(f"\n  Processing {cam_id}...")
        
        # Load background
        bg_img = cv2.imread(str(bg_path))
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        
        # Get depth
        depth_map = depth_estimator.estimate_depth(bg_rgb)
        
        # Detect vehicles
        results = yolo.predict(bg_img, conf=0.3, iou=0.5, verbose=False,
                              classes=list(VALID_CLASSES.keys()))
        
        projector = CameraProjector(cameras[cam_id])
        cam_statics = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                # Only consider vehicles as static objects
                if cls_name not in VEHICLE_CLASSES:
                    continue
                
                # Get object center and bottom
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2  # Center, not bottom
                cy_bottom = bbox[3]
                
                # Sample depth at object center
                dx, dy = int(cx), int(cy)
                dx = max(0, min(depth_map.shape[1] - 1, dx))
                dy = max(0, min(depth_map.shape[0] - 1, dy))
                depth_val = depth_map[dy, dx]
                
                # Project to 3D using depth
                pos_3d = projector.to_3d_with_depth(cx, cy_bottom, depth_val)
                
                # Clamp to reasonable range
                if abs(pos_3d[0]) > 40 or abs(pos_3d[1]) > 40:
                    continue
                
                static_obj = StaticObject(
                    obj_id=next_id,
                    cls=cls_name,
                    camera=cam_id,
                    bbox=bbox,
                    pos_3d=pos_3d,
                    conf=conf
                )
                cam_statics.append(static_obj)
                next_id += 1
                
        static_objects[cam_id] = cam_statics
        print(f"    Found {len(cam_statics)} static vehicles")
    
    # Merge across cameras by 3D proximity
    print("\n  Merging static objects across cameras...")
    all_statics = []
    for cam_statics in static_objects.values():
        all_statics.extend(cam_statics)
    
    # Cluster by 3D position
    merged = []
    used = set()
    
    for i, s1 in enumerate(all_statics):
        if i in used:
            continue
        
        cluster = [s1]
        used.add(i)
        
        for j, s2 in enumerate(all_statics):
            if j in used or j <= i:
                continue
            
            dist = np.linalg.norm(s1.pos_3d[:2] - s2.pos_3d[:2])
            if dist < 3.0:  # Within 3m = same object
                cluster.append(s2)
                used.add(j)
        
        # Take the detection with highest confidence
        best = max(cluster, key=lambda x: x.conf)
        # Average position
        avg_pos = np.mean([s.pos_3d for s in cluster], axis=0)
        best.pos_3d = avg_pos
        merged.append(best)
    
    print(f"  Total unique static objects: {len(merged)}")
    
    return {'all': merged, 'by_camera': static_objects}

def is_detection_static(bbox: np.ndarray, cam_id: str, 
                        static_by_camera: Dict[str, List[StaticObject]]) -> bool:
    """Check if a detection matches a known static object."""
    if cam_id not in static_by_camera:
        return False
    
    for static_obj in static_by_camera[cam_id]:
        if iou(bbox, static_obj.bbox) > IOU_MATCH_THRESH:
            return True
    
    return False

def process_videos(video_dir: Path, cameras: dict, static_objects: dict,
                   yolo: YOLO, depth_estimator: DepthEstimator) -> Tuple[List[dict], int, float]:
    """Process video frames, detecting dynamic objects only."""
    
    print("\n=== PROCESSING VIDEO FRAMES ===")
    
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right', 
               's3-left', 's3-right', 's4-left', 's4-right']
    
    static_by_camera = static_objects['by_camera']
    
    # Open videos
    video_caps = {}
    total_frames = 0
    fps = 14.4
    
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            video_caps[cam_id] = cap
            total_frames = max(total_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Processing {len(video_caps)} cameras, {total_frames} frames @ {fps:.1f} FPS")
    
    all_detections = []
    projectors = {cam_id: CameraProjector(cameras[cam_id]) for cam_id in cam_ids if cam_id in cameras}
    
    for frame_idx in tqdm(range(total_frames), desc="Detecting dynamic objects"):
        frame_dets = []
        
        for cam_id in cam_ids:
            if cam_id not in video_caps:
                continue
            
            cap = video_caps[cam_id]
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Get depth for this frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = depth_estimator.estimate_depth(frame_rgb)
            
            # Detect objects
            results = yolo.predict(frame, conf=MIN_CONFIDENCE, iou=0.5, verbose=False,
                                  classes=list(VALID_CLASSES.keys()))
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = VALID_CLASSES.get(cls_id, 'unknown')
                
                # Skip if this matches a static object
                if cls_name in VEHICLE_CLASSES:
                    if is_detection_static(bbox, cam_id, static_by_camera):
                        continue
                
                # Get 3D position using depth
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                cy_bottom = bbox[3]
                
                dx, dy = int(cx), int(cy)
                dx = max(0, min(depth_map.shape[1] - 1, dx))
                dy = max(0, min(depth_map.shape[0] - 1, dy))
                depth_val = depth_map[dy, dx]
                
                pos_3d = projectors[cam_id].to_3d_with_depth(cx, cy_bottom, depth_val)
                
                # Bounds check
                if abs(pos_3d[0]) > 40 or abs(pos_3d[1]) > 40:
                    continue
                
                frame_dets.append({
                    'frame': frame_idx,
                    'camera': cam_id,
                    'class': cls_name,
                    'conf': conf,
                    'bbox': bbox.tolist(),
                    'pos_3d': pos_3d.tolist()
                })
        
        all_detections.extend(frame_dets)
    
    for cap in video_caps.values():
        cap.release()
    
    return all_detections, total_frames, fps

def track_dynamic_objects(detections: List[dict], total_frames: int, fps: float) -> List[DynamicTrack]:
    """Track dynamic objects with occlusion handling."""
    
    print("\n=== TRACKING DYNAMIC OBJECTS ===")
    
    from sklearn.cluster import DBSCAN
    
    # Group by frame and cluster
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d['frame']].append(d)
    
    # Cluster per frame
    observations = {}  # frame -> list of {pos, cls, cameras, conf}
    
    for frame_idx in range(total_frames):
        frame_dets = by_frame.get(frame_idx, [])
        if not frame_dets:
            continue
        
        # Group by class
        by_class = defaultdict(list)
        for d in frame_dets:
            by_class[d['class']].append(d)
        
        frame_obs = []
        for cls, class_dets in by_class.items():
            if len(class_dets) == 1:
                d = class_dets[0]
                frame_obs.append({
                    'pos': np.array(d['pos_3d']),
                    'cls': cls,
                    'cameras': [d['camera']],
                    'conf': d['conf']
                })
            else:
                # Cluster
                points = np.array([d['pos_3d'][:2] for d in class_dets])
                db = DBSCAN(eps=CLUSTER_DIST, min_samples=1).fit(points)
                
                for label in set(db.labels_):
                    if label == -1:
                        continue
                    indices = np.where(db.labels_ == label)[0]
                    cluster_dets = [class_dets[i] for i in indices]
                    
                    avg_pos = np.median([d['pos_3d'] for d in cluster_dets], axis=0)
                    cameras = list(set(d['camera'] for d in cluster_dets))
                    avg_conf = np.mean([d['conf'] for d in cluster_dets])
                    
                    frame_obs.append({
                        'pos': avg_pos,
                        'cls': cls,
                        'cameras': cameras,
                        'conf': avg_conf
                    })
        
        observations[frame_idx] = frame_obs
    
    # Track over time
    tracks = []
    next_id = 1
    active: List[DynamicTrack] = []
    
    for frame_idx in tqdm(range(total_frames), desc="Tracking"):
        current_obs = observations.get(frame_idx, [])
        
        # Age out old tracks
        still_active = []
        for track in active:
            last_frame = max(track.frames.keys())
            if frame_idx - last_frame <= MAX_GAP_FRAMES:
                still_active.append(track)
            else:
                tracks.append(track)
        active = still_active
        
        if not current_obs:
            continue
        
        if active:
            cost = np.zeros((len(current_obs), len(active)))
            
            for oi, obs in enumerate(current_obs):
                for ti, track in enumerate(active):
                    if track.cls != obs['cls']:
                        if not ({track.cls, obs['cls']} <= {'car', 'truck'}):
                            cost[oi, ti] = 1000
                            continue
                    
                    # Use last known position
                    last_frame = max(track.frames.keys())
                    last_pos = np.array(track.frames[last_frame]['pos'])
                    
                    # Simple velocity prediction
                    if len(track.frames) >= 2:
                        frames_list = sorted(track.frames.keys())
                        if len(frames_list) >= 2:
                            p1 = np.array(track.frames[frames_list[-2]]['pos'])
                            p2 = np.array(track.frames[frames_list[-1]]['pos'])
                            dt_hist = (frames_list[-1] - frames_list[-2]) / fps
                            if dt_hist > 0:
                                vel = (p2 - p1) / dt_hist
                                dt_pred = (frame_idx - last_frame) / fps
                                last_pos = p2 + vel * dt_pred
                    
                    dist = np.linalg.norm(obs['pos'][:2] - last_pos[:2])
                    cost[oi, ti] = dist
            
            rows, cols = linear_sum_assignment(cost)
            
            matched_obs = set()
            for oi, ti in zip(rows, cols):
                if cost[oi, ti] < 8.0:
                    track = active[ti]
                    track.frames[frame_idx] = {
                        'pos': current_obs[oi]['pos'].tolist(),
                        'cameras': current_obs[oi]['cameras'],
                        'conf': current_obs[oi]['conf']
                    }
                    matched_obs.add(oi)
            
            for oi, obs in enumerate(current_obs):
                if oi not in matched_obs:
                    track = DynamicTrack(track_id=next_id, cls=obs['cls'])
                    next_id += 1
                    track.frames[frame_idx] = {
                        'pos': obs['pos'].tolist(),
                        'cameras': obs['cameras'],
                        'conf': obs['conf']
                    }
                    active.append(track)
        else:
            for obs in current_obs:
                track = DynamicTrack(track_id=next_id, cls=obs['cls'])
                next_id += 1
                track.frames[frame_idx] = {
                    'pos': obs['pos'].tolist(),
                    'cameras': obs['cameras'],
                    'conf': obs['conf']
                }
                active.append(track)
    
    tracks.extend(active)
    
    # Filter short tracks
    tracks = [t for t in tracks if t.length >= MIN_TRACK_FRAMES]
    
    print(f"  Final dynamic tracks: {len(tracks)}")
    
    return tracks

def generate_scene(static_objects: List[StaticObject], 
                   dynamic_tracks: List[DynamicTrack],
                   total_frames: int, fps: float) -> dict:
    """Generate final scene with static and dynamic objects."""
    
    print("\n=== GENERATING SCENE ===")
    
    DIMS = {
        'car': [4.5, 1.8, 1.5],
        'truck': [7.0, 2.4, 2.8],
        'bus': [10.0, 2.5, 3.0],
        'motorcycle': [2.0, 0.8, 1.3],
        'bicycle': [1.8, 0.5, 1.4],
        'person': [0.5, 0.5, 1.7]
    }
    
    COLORS = {
        'car': [0, 0, 255],       # Red
        'truck': [255, 0, 200],   # Magenta  
        'bus': [0, 200, 255],     # Yellow
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
    
    # Add static objects (present in ALL frames)
    for static_obj in static_objects:
        tid = f"S{static_obj.obj_id}"  # S prefix for static
        
        scene['objects'][tid] = {
            'class': static_obj.cls,
            'dims': DIMS.get(static_obj.cls, DIMS['car']),
            'color': [128, 128, 128],  # Gray for parked
            'is_stationary': True
        }
        
        # Fixed orientation (estimate from camera direction)
        fixed_yaw = 0.0  # Could be improved
        quat = R.from_euler('z', fixed_yaw).as_quat().tolist()
        
        for frame_idx in range(total_frames):
            frame_key = str(frame_idx)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': static_obj.pos_3d.tolist(),
                'rot': quat,
                'conf': static_obj.conf
            })
    
    # Add dynamic tracks
    for track in dynamic_tracks:
        tid = f"D{track.track_id}"  # D prefix for dynamic
        
        scene['objects'][tid] = {
            'class': track.cls,
            'dims': DIMS.get(track.cls, DIMS['car']),
            'color': COLORS.get(track.cls, [255, 255, 255]),
            'is_stationary': False
        }
        
        # Compute orientations from velocity
        frames_list = sorted(track.frames.keys())
        positions = np.array([track.frames[f]['pos'][:2] for f in frames_list])
        
        velocities = np.diff(positions, axis=0)
        velocities = np.vstack([velocities, velocities[-1] if len(velocities) > 0 else [[1, 0]]])
        yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
        yaws = uniform_filter1d(np.unwrap(yaws), size=5, mode='nearest')
        
        for i, frame_idx in enumerate(frames_list):
            frame_key = str(frame_idx)
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            quat = R.from_euler('z', float(yaws[i])).as_quat().tolist()
            
            scene['frames'][frame_key].append({
                'id': tid,
                'pos': track.frames[frame_idx]['pos'],
                'rot': quat,
                'conf': track.frames[frame_idx]['conf']
            })
    
    print(f"  Static objects: {len(static_objects)}")
    print(f"  Dynamic tracks: {len(dynamic_tracks)}")
    
    return scene

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DEPTH-AWARE 4D TRACKING WITH STATIC/DYNAMIC SEPARATION")
    print("=" * 70)
    
    # Load camera params
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    bg_dir = base_dir / "data" / "processed" / "static_backgrounds"
    video_dir = base_dir / "StreetAware-sample"
    
    # Initialize models
    print("\nInitializing models...")
    yolo = YOLO('yolov8x.pt')
    depth_estimator = DepthEstimator()
    
    # Stage 1: Detect static objects
    static_objects = detect_static_objects(bg_dir, cameras, yolo, depth_estimator)
    
    # Stage 2: Process videos for dynamic objects
    detections, total_frames, fps = process_videos(
        video_dir, cameras, static_objects, yolo, depth_estimator
    )
    print(f"\n  Total dynamic detections: {len(detections)}")
    
    # Stage 3: Track dynamic objects
    dynamic_tracks = track_dynamic_objects(detections, total_frames, fps)
    
    # Stage 4: Generate scene
    scene = generate_scene(static_objects['all'], dynamic_tracks, total_frames, fps)
    
    # Save
    out_path = work_dir / "scene_4d.json"
    print(f"\nSaving to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
