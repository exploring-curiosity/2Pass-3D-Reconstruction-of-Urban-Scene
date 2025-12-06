#!/usr/bin/env python3
"""
Generate annotated videos showing what the tracker sees.
Uses two-pass approach:
1. Collect all detections
2. Build tracks with IoU matching, lock class at peak confidence
3. Render with locked classes (no class oscillation)
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def generate_annotated_videos(
    video_dir: Path,
    output_dir: Path,
    camera_params_path: Path
):
    """Generate annotated videos with locked classes."""
    
    print("Loading YOLO model...")
    model = YOLO('yolov8x.pt')
    
    video_paths = sorted(video_dir.glob("*.mp4"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Colors for different classes
    CLASS_COLORS = {
        'car': (0, 0, 255),       # Red
        'truck': (0, 128, 255),   # Orange
        'bus': (0, 255, 255),     # Yellow
        'motorcycle': (255, 0, 255),  # Magenta
        'bicycle': (255, 255, 0),  # Cyan
        'person': (0, 255, 0),     # Green
    }
    
    for video_path in video_paths:
        cam_id = video_path.stem
        print(f"\nProcessing {cam_id}...")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # =====================
        # Pass 1: Collect all detections
        # =====================
        print(f"  Pass 1: Collecting detections...")
        all_detections = {}
        all_frames = {}
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"    Detecting")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            all_frames[frame_idx] = frame.copy()
            
            results = model.predict(
                frame,
                conf=0.3,
                iou=0.5,
                verbose=False,
                classes=[0, 1, 2, 3, 5, 7]
            )
            
            frame_dets = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.int().cpu().numpy()
                
                for i in range(len(xyxys)):
                    bbox = xyxys[i]
                    cls_name = model.names[cls_ids[i]]
                    conf = confs[i]
                    frame_dets.append((bbox.copy(), cls_name, conf))
            
            all_detections[frame_idx] = frame_dets
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # =====================
        # Pass 2: Build tracks with IoU matching
        # =====================
        print(f"  Pass 2: Building tracks...")
        
        tracks: Dict[int, Dict[int, Tuple]] = {}
        next_track_id = 1
        active_tracks: Dict[int, Tuple[np.ndarray, int]] = {}
        track_velocities: Dict[int, np.ndarray] = {}
        
        for frame_idx in range(len(all_detections)):
            dets = all_detections[frame_idx]
            
            if not dets:
                continue
            
            matched_dets = set()
            
            if active_tracks:
                track_ids = list(active_tracks.keys())
                costs = np.zeros((len(track_ids), len(dets)))
                
                for ti, track_id in enumerate(track_ids):
                    last_bbox, last_frame = active_tracks[track_id]
                    frames_gap = frame_idx - last_frame
                    
                    # Predict position
                    velocity = track_velocities.get(track_id)
                    if velocity is not None:
                        predicted = last_bbox.copy()
                        predicted[0] += velocity[0] * frames_gap
                        predicted[1] += velocity[1] * frames_gap
                        predicted[2] += velocity[0] * frames_gap
                        predicted[3] += velocity[1] * frames_gap
                    else:
                        predicted = last_bbox
                    
                    for di, (det_bbox, _, _) in enumerate(dets):
                        iou = compute_iou(predicted, det_bbox)
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
                    track_id = track_ids[ti]
                    det_bbox, det_cls, det_conf = dets[di]
                    
                    tracks[track_id][frame_idx] = (det_bbox, det_cls, det_conf)
                    
                    # Update velocity
                    last_bbox, last_frame = active_tracks[track_id]
                    if frame_idx > last_frame:
                        dx = (det_bbox[0] + det_bbox[2]) / 2 - (last_bbox[0] + last_bbox[2]) / 2
                        dy = (det_bbox[1] + det_bbox[3]) / 2 - (last_bbox[1] + last_bbox[3]) / 2
                        dt = frame_idx - last_frame
                        new_vel = np.array([dx / dt, dy / dt])
                        if track_id in track_velocities:
                            track_velocities[track_id] = 0.7 * track_velocities[track_id] + 0.3 * new_vel
                        else:
                            track_velocities[track_id] = new_vel
                    
                    active_tracks[track_id] = (det_bbox, frame_idx)
                    matched_dets.add(di)
                    
                    costs[ti, :] = np.inf
                    costs[:, di] = np.inf
            
            # Create new tracks
            for di, (det_bbox, det_cls, det_conf) in enumerate(dets):
                if di not in matched_dets:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {frame_idx: (det_bbox, det_cls, det_conf)}
                    active_tracks[track_id] = (det_bbox, frame_idx)
            
            # Remove stale
            stale = [tid for tid, (_, last_f) in active_tracks.items() 
                     if frame_idx - last_f > 30]
            for tid in stale:
                del active_tracks[tid]
                if tid in track_velocities:
                    del track_velocities[tid]
        
        # =====================
        # Pass 3: Lock class at peak confidence
        # =====================
        print(f"  Pass 3: Locking classes...")
        
        locked_tracks = {}  # track_id -> {frame_idx: (bbox, locked_class, conf)}
        
        for track_id, track_frames in tracks.items():
            if len(track_frames) < 3:
                continue
            
            # Collect class votes weighted by confidence
            class_scores = defaultdict(float)
            for f, (bbox, cls_name, conf) in track_frames.items():
                class_scores[cls_name] += conf
            
            # Determine locked class
            locked_class = max(class_scores.keys(), key=lambda c: class_scores[c])
            
            # Special handling for bicycle+person
            if 'bicycle' in class_scores and 'person' in class_scores:
                if class_scores['bicycle'] > class_scores['person'] * 0.3:
                    locked_class = 'bicycle'
            
            locked_tracks[track_id] = {
                'class': locked_class,
                'frames': {f: (bbox, conf) for f, (bbox, _, conf) in track_frames.items()}
            }
        
        print(f"    Built {len(locked_tracks)} tracks with locked classes")
        
        # =====================
        # Pass 4: Render annotated video
        # =====================
        print(f"  Pass 4: Rendering video...")
        
        output_path = output_dir / f"{cam_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        pbar = tqdm(total=len(all_frames), desc=f"    Rendering")
        
        for frame_idx in range(len(all_frames)):
            frame = all_frames[frame_idx]
            
            # Draw all tracks visible in this frame
            for track_id, track_data in locked_tracks.items():
                if frame_idx not in track_data['frames']:
                    continue
                
                bbox, conf = track_data['frames'][frame_idx]
                cls_name = track_data['class']
                
                bbox = bbox.astype(int)
                color = CLASS_COLORS.get(cls_name, (128, 128, 128))
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw track ID and LOCKED class
                label = f"ID:{track_id} {cls_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                cv2.rectangle(frame, 
                              (bbox[0], bbox[1] - label_size[1] - 10),
                              (bbox[0] + label_size[0], bbox[1]),
                              color, -1)
                
                cv2.putText(frame, label,
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw bottom center
                bottom_center = (int((bbox[0] + bbox[2]) / 2), bbox[3])
                cv2.circle(frame, bottom_center, 5, (0, 255, 255), -1)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_idx} | LOCKED CLASSES", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, cam_id, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            out.write(frame)
            pbar.update(1)
        
        pbar.close()
        out.release()
        all_frames.clear()  # Free memory
        
        print(f"  Saved: {output_path}")
    
    print(f"\nAll annotated videos saved to {output_dir}")


def main():
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir / "StreetAware-sample"
    output_dir = base_dir / "outputs" / "pass2_dynamic" / "annotated_videos"
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    
    generate_annotated_videos(video_dir, output_dir, cameras_path)


if __name__ == "__main__":
    main()
