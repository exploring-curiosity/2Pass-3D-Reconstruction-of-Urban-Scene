#!/usr/bin/env python3
"""
Clean Vehicle Tracker - Complete rewrite for accurate 3D tracking.

Key improvements:
1. Use BOTTOM CENTER of bbox for ground projection (not center)
2. Stricter track association to avoid fragmentation
3. Smooth trajectories with Kalman filtering
4. Only output tracks with consistent movement
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

# NumPy 2.x compatibility
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import torch
from ultralytics import YOLO
from tqdm import tqdm


@dataclass
class Detection:
    """Single frame detection."""
    frame_idx: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def bottom_center(self) -> np.ndarray:
        """Bottom center of bbox - where the object touches ground."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,  # center x
            self.bbox[3]  # bottom y
        ])
    
    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])
    
    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class Track:
    """A tracked object across frames."""
    track_id: int
    class_name: str
    detections: List[Detection] = field(default_factory=list)
    positions_3d: Dict[int, np.ndarray] = field(default_factory=dict)  # frame -> [x, y, z]
    
    # Kalman filter state for smoothing
    kf_state: Optional[np.ndarray] = None  # [x, y, vx, vy]
    kf_cov: Optional[np.ndarray] = None
    
    @property
    def category(self) -> str:
        if self.class_name in ['car', 'truck', 'bus', 'motorcycle']:
            return 'vehicle'
        elif self.class_name in ['person', 'bicycle']:
            return 'person'
        return 'other'
    
    @property
    def num_frames(self) -> int:
        return len(self.detections)
    
    @property
    def frame_range(self) -> Tuple[int, int]:
        if not self.detections:
            return (0, 0)
        frames = [d.frame_idx for d in self.detections]
        return (min(frames), max(frames))
    
    def last_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None
    
    def total_movement_3d(self) -> float:
        """Total 3D movement in meters."""
        if len(self.positions_3d) < 2:
            return 0.0
        
        frames = sorted(self.positions_3d.keys())
        first_pos = self.positions_3d[frames[0]][:2]  # XY only
        last_pos = self.positions_3d[frames[-1]][:2]
        return float(np.linalg.norm(last_pos - first_pos))
    
    def is_moving(self, threshold_m: float = 2.0) -> bool:
        """Check if track has significant movement."""
        return self.total_movement_3d() > threshold_m


class KalmanFilter2D:
    """Simple 2D Kalman filter for position smoothing."""
    
    def __init__(self, initial_pos: np.ndarray, dt: float = 1/15):
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.state = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float64)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Observation matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # Process noise
        q = 0.5  # Process noise magnitude
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float64) * q
        
        # Measurement noise
        self.R = np.eye(2) * 1.0  # 1 meter measurement noise
        
        # Initial covariance
        self.P = np.eye(4) * 10.0
    
    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with measurement."""
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]


class CleanTracker:
    """Clean multi-object tracker with 3D projection."""
    
    def __init__(
        self,
        camera_params: Dict,
        ground_z: float = 0.0,
        image_size: Tuple[int, int] = (2592, 1944),
        max_age: int = 30,  # Max frames to keep track without detection
        min_hits: int = 5,  # Min detections to confirm track
        iou_threshold: float = 0.3
    ):
        self.camera_params = camera_params
        self.ground_z = ground_z
        self.image_width, self.image_height = image_size
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Parse camera parameters
        self.K = np.array(camera_params['K']).reshape(3, 3)
        self.R_c2w = np.array(camera_params['R'])
        self.t_c2w = np.array(camera_params['t'])
        
        # Scale intrinsics if needed
        calib_w = camera_params.get('width', self.image_width)
        calib_h = camera_params.get('height', self.image_height)
        if calib_w != self.image_width:
            scale_x = self.image_width / calib_w
            scale_y = self.image_height / calib_h
            self.K[0, :] *= scale_x
            self.K[1, :] *= scale_y
        
        # Active tracks
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
    
    def project_to_ground(self, pixel: np.ndarray) -> Optional[np.ndarray]:
        """Project pixel to ground plane in world coordinates."""
        u, v = pixel
        
        # Pixel to normalized camera coordinates
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        # Ray in camera frame
        ray_cam = np.array([x_cam, y_cam, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world frame
        ray_world = self.R_c2w @ ray_cam
        
        # Ray-ground intersection
        if abs(ray_world[2]) < 1e-6:
            return None
        
        t = (self.ground_z - self.t_c2w[2]) / ray_world[2]
        if t < 0:
            return None
        
        intersection = self.t_c2w + t * ray_world
        return intersection
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections."""
        self.frame_count += 1
        
        if not detections:
            # Age out tracks
            for track in list(self.tracks.values()):
                last_det = track.last_detection()
                if last_det and self.frame_count - last_det.frame_idx > self.max_age:
                    del self.tracks[track.track_id]
            return list(self.tracks.values())
        
        # Match detections to existing tracks using IoU
        matched_tracks = set()
        matched_dets = set()
        
        # Sort tracks by number of detections (prioritize established tracks)
        sorted_tracks = sorted(
            self.tracks.values(),
            key=lambda t: len(t.detections),
            reverse=True
        )
        
        for track in sorted_tracks:
            last_det = track.last_detection()
            if last_det is None:
                continue
            
            # Find best matching detection
            best_iou = self.iou_threshold
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                if det.class_name != track.class_name:
                    continue
                
                iou = self.compute_iou(last_det.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx >= 0:
                matched_tracks.add(track.track_id)
                matched_dets.add(best_det_idx)
                
                det = detections[best_det_idx]
                track.detections.append(det)
                
                # Project to 3D
                pos_3d = self.project_to_ground(det.bottom_center)
                if pos_3d is not None:
                    # Apply Kalman filter
                    if track.kf_state is None:
                        track.kf_state = KalmanFilter2D(pos_3d[:2])
                    else:
                        track.kf_state.predict()
                        smoothed = track.kf_state.update(pos_3d[:2])
                        pos_3d[:2] = smoothed
                    
                    track.positions_3d[det.frame_idx] = pos_3d
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            
            # Only create tracks for vehicles
            if det.class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
            
            track = Track(
                track_id=self.next_track_id,
                class_name=det.class_name
            )
            track.detections.append(det)
            
            # Project to 3D
            pos_3d = self.project_to_ground(det.bottom_center)
            if pos_3d is not None:
                track.kf_state = KalmanFilter2D(pos_3d[:2])
                track.positions_3d[det.frame_idx] = pos_3d
            
            self.tracks[track.track_id] = track
            self.next_track_id += 1
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            last_det = track.last_detection()
            if last_det and self.frame_count - last_det.frame_idx > self.max_age:
                del self.tracks[track_id]
        
        return list(self.tracks.values())
    
    def get_confirmed_tracks(self, min_frames: int = 30, min_movement: float = 2.0) -> List[Track]:
        """Get tracks that meet quality criteria."""
        confirmed = []
        for track in self.tracks.values():
            if track.num_frames >= min_frames and track.is_moving(min_movement):
                confirmed.append(track)
        return confirmed


def run_detection(video_path: Path, model: YOLO, conf_threshold: float = 0.5) -> Dict[int, List[Detection]]:
    """Run YOLO detection on video, return detections per frame."""
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detections_by_frame: Dict[int, List[Detection]] = {}
    
    for frame_idx in tqdm(range(total_frames), desc=f"Detecting {video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO
        results = model(frame, verbose=False, conf=conf_threshold)
        
        frame_dets = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = model.names[cls_id]
                
                # Only keep vehicles
                if cls_name in ['car', 'truck', 'bus', 'motorcycle']:
                    frame_dets.append(Detection(
                        frame_idx=frame_idx,
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    ))
        
        detections_by_frame[frame_idx] = frame_dets
    
    cap.release()
    return detections_by_frame


def process_video(
    video_path: Path,
    camera_params: Dict,
    output_path: Path,
    ground_z: float = 0.0
) -> List[Track]:
    """Process a single video and return tracks."""
    
    # Load YOLO model
    model = YOLO('yolov8x.pt')
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Processing {video_path.name}: {width}x{height} @ {fps:.1f} fps")
    
    # Run detection
    detections_by_frame = run_detection(video_path, model)
    
    # Initialize tracker
    tracker = CleanTracker(
        camera_params=camera_params,
        ground_z=ground_z,
        image_size=(width, height),
        max_age=15,  # 1 second at 15fps
        min_hits=5,
        iou_threshold=0.25
    )
    
    # Process frames
    total_frames = max(detections_by_frame.keys()) + 1 if detections_by_frame else 0
    
    for frame_idx in range(total_frames):
        dets = detections_by_frame.get(frame_idx, [])
        tracker.update(dets)
    
    # Get confirmed tracks
    confirmed = tracker.get_confirmed_tracks(min_frames=30, min_movement=2.0)
    
    print(f"  Found {len(confirmed)} confirmed moving tracks")
    
    # Save to JSON
    output_data = {
        'camera_id': video_path.stem,
        'video_path': str(video_path),
        'fps': fps,
        'width': width,
        'height': height,
        'num_tracks': len(confirmed),
        'trajectories': []
    }
    
    for track in confirmed:
        traj = {
            'track_id': track.track_id,
            'class_name': track.class_name,
            'category': track.category,
            'is_stationary': not track.is_moving(),
            'num_frames': track.num_frames,
            'total_movement_m': track.total_movement_3d(),
            'frames': []
        }
        
        for det in track.detections:
            frame_data = {
                'frame_idx': det.frame_idx,
                'bbox': det.bbox.tolist(),
                'confidence': det.confidence
            }
            
            if det.frame_idx in track.positions_3d:
                pos = track.positions_3d[det.frame_idx]
                frame_data['position_3d'] = pos.tolist()
            
            traj['frames'].append(frame_data)
        
        output_data['trajectories'].append(traj)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved to {output_path}")
    
    return confirmed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean Vehicle Tracker")
    parser.add_argument('--video', type=str, help='Single video to process')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "StreetAware-sample"
    output_dir = base_dir / "outputs" / "pass2_dynamic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera parameters
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        all_cameras = json.load(f)
    
    # Ground Z (should be ~0 after correction)
    ground_z = 0.0
    
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            video_path = video_dir / args.video
        
        cam_name = video_path.stem
        if cam_name not in all_cameras:
            print(f"Error: No camera params for {cam_name}")
            return
        
        output_path = output_dir / f"{cam_name}_trajectories_clean.json"
        process_video(video_path, all_cameras[cam_name], output_path, ground_z)
    
    elif args.all:
        for video_path in sorted(video_dir.glob("*.mp4")):
            cam_name = video_path.stem
            if cam_name not in all_cameras:
                print(f"Skipping {cam_name}: no camera params")
                continue
            
            output_path = output_dir / f"{cam_name}_trajectories_clean.json"
            process_video(video_path, all_cameras[cam_name], output_path, ground_z)
    
    else:
        print("Specify --video <path> or --all")


if __name__ == "__main__":
    main()
