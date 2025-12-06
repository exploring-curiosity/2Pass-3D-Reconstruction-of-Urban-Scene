#!/usr/bin/env python3
"""
Two-Pass 3D Cross-Camera Tracker

Approach:
1. PASS 1: Collect ALL detections from all 8 cameras, project to 3D
2. PASS 2: 
   - Cluster detections by 3D position across ALL frames
   - For each cluster (object), find the frame with HIGHEST confidence
   - Lock the class at that peak confidence frame
   - Build track forward and backward from peak using 3D position matching

This avoids:
- Class oscillation (class is locked at peak confidence)
- Too many tracks (spatial clustering merges same object across cameras)
- Prediction drift (no Kalman, just direct 3D matching)

Only 4 classes: car, truck, bicycle, person
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import colorsys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


# Only these 4 classes
VALID_CLASSES = {'car', 'truck', 'bicycle', 'person'}
CLASS_IDS = [0, 1, 2, 7]  # person=0, bicycle=1, car=2, truck=7


@dataclass
class Detection:
    """A single detection."""
    cam_id: str
    frame_idx: int
    bbox: np.ndarray
    pos_3d: np.ndarray
    cls_name: str
    confidence: float


class CameraProjector:
    """Projects 2D image points to 3D ground plane."""
    
    def __init__(self, camera_params: dict):
        self.K = np.array(camera_params['K']).reshape(3, 3)
        pose_c2w = np.array(camera_params['pose_c2w'])
        self.R_c2w = pose_c2w[:3, :3]
        self.cam_pos = pose_c2w[:3, 3]
        
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
        return self.cam_pos + s * ray_world


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def run_two_pass_tracking(
    video_dir: Path,
    output_dir: Path,
    camera_params_path: Path,
    spatial_threshold: float = 2.5,  # meters - same object if within this distance
    min_detections: int = 10,  # minimum detections to be a valid track
):
    print("=" * 70)
    print("Two-Pass 3D Cross-Camera Tracker")
    print("=" * 70)
    
    with open(camera_params_path) as f:
        camera_params = json.load(f)
    
    print("\nLoading YOLO model...")
    model = YOLO('yolov8x.pt')
    
    cam_order = ['s1-left', 's1-right', 's2-left', 's2-right', 
                 's3-left', 's3-right', 's4-left', 's4-right']
    
    video_paths = {cam_id: video_dir / f"{cam_id}.mp4" 
                   for cam_id in cam_order if (video_dir / f"{cam_id}.mp4").exists()}
    
    first_cap = cv2.VideoCapture(str(list(video_paths.values())[0]))
    total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()
    
    print(f"Processing {len(video_paths)} cameras, {total_frames} frames")
    
    projectors = {cam_id: CameraProjector(camera_params[cam_id]) for cam_id in video_paths}
    
    # ========================================
    # PASS 1: Collect ALL detections
    # ========================================
    print("\n" + "=" * 70)
    print("PASS 1: Collecting all detections")
    print("=" * 70)
    
    all_detections: List[Detection] = []
    
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        for cam_id in cam_order:
            if cam_id not in caps:
                continue
            
            ret, frame = caps[cam_id].read()
            if not ret:
                continue
            
            results = model.predict(
                frame, conf=0.4, iou=0.5, verbose=False,
                classes=CLASS_IDS
            )
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            boxes = results[0].boxes
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()
            
            for i in range(len(xyxys)):
                cls_name = model.names[cls_ids[i]]
                if cls_name not in VALID_CLASSES:
                    continue
                
                bbox = xyxys[i]
                bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                pos_3d = projectors[cam_id].project_to_ground(bottom_center)
                
                if pos_3d is None or abs(pos_3d[0]) > 30 or abs(pos_3d[1]) > 30:
                    continue
                
                all_detections.append(Detection(
                    cam_id=cam_id,
                    frame_idx=frame_idx,
                    bbox=bbox,
                    pos_3d=pos_3d,
                    cls_name=cls_name,
                    confidence=confs[i]
                ))
    
    for cap in caps.values():
        cap.release()
    
    print(f"\nTotal detections: {len(all_detections)}")
    
    # ========================================
    # PASS 2: Cluster by 3D position, lock class at peak
    # ========================================
    print("\n" + "=" * 70)
    print("PASS 2: Clustering and class locking")
    print("=" * 70)
    
    # Sort detections by confidence (highest first)
    all_detections.sort(key=lambda d: -d.confidence)
    
    # Group detections by frame first
    dets_by_frame = defaultdict(list)
    for i, det in enumerate(all_detections):
        dets_by_frame[det.frame_idx].append((i, det))
    
    # For each frame, cluster detections by 3D position
    # Detections within spatial_threshold in same frame = same object
    frame_clusters = {}  # frame_idx -> list of clusters, each cluster = list of det indices
    
    for frame_idx in sorted(dets_by_frame.keys()):
        frame_dets = dets_by_frame[frame_idx]
        used_in_frame = set()
        clusters_in_frame = []
        
        for i, (idx, det) in enumerate(frame_dets):
            if i in used_in_frame:
                continue
            
            cluster = [(idx, det)]
            used_in_frame.add(i)
            
            for j, (idx2, det2) in enumerate(frame_dets):
                if j in used_in_frame:
                    continue
                # Same camera = different objects
                if det.cam_id == det2.cam_id:
                    continue
                
                dist = np.linalg.norm(det.pos_3d[:2] - det2.pos_3d[:2])
                if dist < spatial_threshold:
                    cluster.append((idx2, det2))
                    used_in_frame.add(j)
            
            clusters_in_frame.append(cluster)
        
        frame_clusters[frame_idx] = clusters_in_frame
    
    # Now link clusters across frames
    # Each cluster gets a centroid, link if centroid is close in consecutive frames
    
    # Assign temporary IDs to frame clusters
    cluster_id = 0
    frame_cluster_ids = {}  # frame_idx -> list of (cluster_id, centroid, dets)
    
    for frame_idx, clusters_in_frame in frame_clusters.items():
        frame_cluster_ids[frame_idx] = []
        for cluster in clusters_in_frame:
            centroid = np.mean([det.pos_3d[:2] for _, det in cluster], axis=0)
            frame_cluster_ids[frame_idx].append((cluster_id, centroid, cluster))
            cluster_id += 1
    
    # Link clusters across frames using position continuity
    # Use Union-Find to merge clusters that are the same object
    parent = list(range(cluster_id))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    print("  Linking clusters across frames...")
    sorted_frames = sorted(frame_cluster_ids.keys())
    
    # Get dominant class for each cluster
    def get_cluster_class(cluster):
        class_counts = defaultdict(int)
        for _, det in cluster:
            class_counts[det.cls_name] += 1
        return max(class_counts.keys(), key=lambda c: class_counts[c])
    
    cluster_classes = {}
    for frame_idx, clusters_info in frame_cluster_ids.items():
        for cid, centroid, dets in clusters_info:
            cluster_classes[cid] = get_cluster_class(dets)
    
    for i in range(len(sorted_frames) - 1):
        f1, f2 = sorted_frames[i], sorted_frames[i + 1]
        
        # Only link consecutive frames
        if f2 - f1 > 2:
            continue
        
        for cid1, cent1, _ in frame_cluster_ids[f1]:
            best_match = None
            best_dist = float('inf')
            
            for cid2, cent2, _ in frame_cluster_ids[f2]:
                # Must be same class category
                cls1, cls2 = cluster_classes[cid1], cluster_classes[cid2]
                # Allow car<->truck confusion
                if cls1 != cls2:
                    if not (cls1 in ['car', 'truck'] and cls2 in ['car', 'truck']):
                        continue
                
                dist = np.linalg.norm(cent1 - cent2)
                # Strict: max 2m movement per frame
                if dist < 2.0 * (f2 - f1) and dist < best_dist:
                    best_match = cid2
                    best_dist = dist
            
            if best_match is not None:
                union(cid1, best_match)
    
    # Group by root parent
    cluster_groups = defaultdict(list)
    for frame_idx, clusters_info in frame_cluster_ids.items():
        for cid, centroid, dets in clusters_info:
            root = find(cid)
            cluster_groups[root].extend([det for _, det in dets])
    
    # Filter by minimum detections
    raw_clusters = [dets for dets in cluster_groups.values() if len(dets) >= min_detections]
    
    print(f"  Raw clusters after linking: {len(raw_clusters)}")
    
    # ========================================
    # Merge stationary objects
    # ========================================
    # Stationary objects have low position variance across frames
    # Merge clusters that overlap in space (same stationary object detected in different time windows)
    
    print("  Merging stationary objects...")
    
    def get_cluster_stats(cluster):
        """Get centroid and position variance of cluster."""
        positions = np.array([det.pos_3d[:2] for det in cluster])
        centroid = np.mean(positions, axis=0)
        variance = np.var(positions, axis=0).sum()
        frames = [det.frame_idx for det in cluster]
        return centroid, variance, min(frames), max(frames)
    
    cluster_stats = [get_cluster_stats(c) for c in raw_clusters]
    
    # Merge clusters with similar centroids (stationary objects)
    merged_parent = list(range(len(raw_clusters)))
    
    def find_merged(x):
        if merged_parent[x] != x:
            merged_parent[x] = find_merged(merged_parent[x])
        return merged_parent[x]
    
    def union_merged(x, y):
        px, py = find_merged(x), find_merged(y)
        if px != py:
            merged_parent[px] = py
    
    for i in range(len(raw_clusters)):
        cent_i, var_i, _, _ = cluster_stats[i]
        cls_i = max(set(d.cls_name for d in raw_clusters[i]), key=lambda c: sum(1 for d in raw_clusters[i] if d.cls_name == c))
        
        for j in range(i + 1, len(raw_clusters)):
            cent_j, var_j, _, _ = cluster_stats[j]
            cls_j = max(set(d.cls_name for d in raw_clusters[j]), key=lambda c: sum(1 for d in raw_clusters[j] if d.cls_name == c))
            
            # Must be same class (or car/truck)
            if cls_i != cls_j:
                if not (cls_i in ['car', 'truck'] and cls_j in ['car', 'truck']):
                    continue
            
            # Check if centroids are close (stationary object)
            dist = np.linalg.norm(cent_i - cent_j)
            if dist < 2.5:  # Within 2.5m = same stationary object
                union_merged(i, j)
    
    # Group merged clusters
    final_groups = defaultdict(list)
    for i, cluster in enumerate(raw_clusters):
        root = find_merged(i)
        final_groups[root].extend(cluster)
    
    clusters = list(final_groups.values())
    
    # Sort by number of detections (largest first)
    clusters.sort(key=lambda c: -len(c))
    
    print(f"\nFound {len(clusters)} object clusters")
    
    # ========================================
    # Build tracks with locked classes
    # ========================================
    print("\n" + "=" * 70)
    print("Building tracks with locked classes")
    print("=" * 70)
    
    tracks = {}  # track_id -> {class, frames: {frame_idx: {cam_id: (bbox, pos_3d, conf)}}}
    
    for track_id, cluster in enumerate(clusters, 1):
        # Find peak confidence detection - this locks the class
        peak_det = max(cluster, key=lambda d: d.confidence)
        locked_class = peak_det.cls_name
        
        # Build frame data
        frames_data = defaultdict(dict)
        for det in cluster:
            frames_data[det.frame_idx][det.cam_id] = (det.bbox, det.pos_3d, det.confidence)
        
        # Compute average position per frame
        frame_positions = {}
        for f, cam_data in frames_data.items():
            positions = [pos for _, pos, _ in cam_data.values()]
            frame_positions[f] = np.mean(positions, axis=0)
        
        # Count cameras
        all_cams = set()
        for cam_data in frames_data.values():
            all_cams.update(cam_data.keys())
        
        tracks[track_id] = {
            'class': locked_class,
            'peak_confidence': peak_det.confidence,
            'peak_frame': peak_det.frame_idx,
            'frames': dict(frames_data),
            'positions': frame_positions,
            'cameras': all_cams,
            'total_detections': len(cluster)
        }
        
        print(f"  Track {track_id}: {locked_class}, {len(cluster)} dets, "
              f"{len(all_cams)} cams, frames {min(frames_data.keys())}-{max(frames_data.keys())}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(tracks)} tracks")
    print(f"{'='*70}")
    
    class_counts = defaultdict(int)
    cam_counts = defaultdict(int)
    for t in tracks.values():
        class_counts[t['class']] += 1
        cam_counts[len(t['cameras'])] += 1
    
    print("\nBy class:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")
    
    print("\nBy camera coverage:")
    for n in sorted(cam_counts.keys(), reverse=True):
        print(f"  {n} cameras: {cam_counts[n]} tracks")
    
    # ========================================
    # Generate verification video
    # ========================================
    print("\n" + "=" * 70)
    print("Generating verification video")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = generate_colors(len(tracks) + 5)
    track_colors = {tid: colors[i] for i, tid in enumerate(tracks.keys())}
    
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    grid_w, grid_h = 640, 360
    output_w, output_h = grid_w * 2, grid_h * 4
    
    output_path = output_dir / "cross_camera_verification.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = cv2.resize(frame, (grid_w, grid_h))
            else:
                frames[cam_id] = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Draw tracks
        for tid, track in tracks.items():
            if frame_idx not in track['frames']:
                continue
            
            color = track_colors[tid]
            
            for cam_id, (bbox, pos_3d, conf) in track['frames'][frame_idx].items():
                if cam_id not in frames:
                    continue
                
                frame = frames[cam_id]
                
                scale_x, scale_y = grid_w / width, grid_h / height
                bbox_s = [int(bbox[0]*scale_x), int(bbox[1]*scale_y),
                          int(bbox[2]*scale_x), int(bbox[3]*scale_y)]
                
                cv2.rectangle(frame, (bbox_s[0], bbox_s[1]), (bbox_s[2], bbox_s[3]), color, 2)
                
                label = f"T{tid} {track['class']}"
                cv2.putText(frame, label, (bbox_s[0], bbox_s[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Camera labels
        for cam_id in cam_order:
            if cam_id in frames:
                cv2.putText(frames[cam_id], cam_id, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Arrange grid
        grid = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        for i, cam_id in enumerate(cam_order):
            if cam_id in frames:
                row, col = i // 2, i % 2
                grid[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w] = frames[cam_id]
        
        cv2.putText(grid, f"Frame {frame_idx}/{total_frames}", (10, output_h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        out.write(grid)
    
    for cap in caps.values():
        cap.release()
    out.release()
    
    print(f"\nSaved: {output_path}")
    
    # Save track data
    track_json = {
        'total_tracks': len(tracks),
        'tracks': {
            str(tid): {
                'class': t['class'],
                'cameras': list(t['cameras']),
                'total_detections': t['total_detections'],
                'frame_range': [min(t['frames'].keys()), max(t['frames'].keys())]
            }
            for tid, t in tracks.items()
        }
    }
    
    with open(output_dir / "tracks.json", 'w') as f:
        json.dump(track_json, f, indent=2)
    
    # ========================================
    # Generate 4D trajectory data for viewer
    # ========================================
    print("\n" + "=" * 70)
    print("Generating 4D trajectory data")
    print("=" * 70)
    
    # Determine if track is stationary (low movement variance)
    def is_stationary(positions_dict):
        if len(positions_dict) < 5:
            return False
        positions = np.array(list(positions_dict.values()))[:, :2]
        variance = np.var(positions, axis=0).sum()
        return variance < 2.0  # Less than 2m^2 total variance
    
    trajectory_data = {
        'fps': fps,
        'max_frame': total_frames - 1,
        'tracks': []
    }
    
    for tid, track in tracks.items():
        positions = track['positions']
        frames = sorted(positions.keys())
        
        if len(frames) < 3:
            continue
        
        # Determine category for viewer
        cls = track['class']
        if cls in ['car', 'truck']:
            category = 'vehicle'
        elif cls == 'bicycle':
            category = 'bicycle'
        else:
            category = 'person'
        
        stationary = is_stationary(positions)
        
        # Build frame-wise position data
        # Format: {frame_idx: [x, y, z]} where z is height (0 for ground)
        frame_positions = {}
        
        # Smooth positions using simple moving average
        sorted_frames = sorted(positions.keys())
        smoothed = {}
        
        for i, f in enumerate(sorted_frames):
            # Get nearby positions for smoothing
            nearby = []
            for j in range(max(0, i-2), min(len(sorted_frames), i+3)):
                nearby.append(positions[sorted_frames[j]][:2])
            smoothed[f] = np.mean(nearby, axis=0)
        
        # Compute heading for each frame
        headings = {}
        for i, f in enumerate(sorted_frames):
            if i < len(sorted_frames) - 1:
                next_f = sorted_frames[i + 1]
                delta = smoothed[next_f] - smoothed[f]
                if np.linalg.norm(delta) > 0.1:
                    headings[f] = float(np.arctan2(delta[1], delta[0]))
                else:
                    headings[f] = headings.get(sorted_frames[i-1], 0) if i > 0 else 0
            else:
                headings[f] = headings.get(sorted_frames[i-1], 0) if i > 0 else 0
        
        # Build final frame data
        for f in sorted_frames:
            pos = smoothed[f]
            # Convert to Three.js coordinates: X=X, Y=Z(height), Z=-Y
            frame_positions[str(f)] = [float(pos[0]), 0.0, float(-pos[1])]
        
        track_entry = {
            'track_id': int(tid),
            'class': cls,
            'category': category,
            'is_stationary': bool(stationary),
            'frames': frame_positions,
            'headings': {str(f): h for f, h in headings.items()}
        }
        
        trajectory_data['tracks'].append(track_entry)
    
    # Save trajectory data
    traj_path = output_dir / "trajectories.json"
    with open(traj_path, 'w') as f:
        json.dump(trajectory_data, f)
    print(f"Saved: {traj_path}")
    
    # Also save to viewer data directory
    viewer_data_dir = output_dir.parent.parent / "viewer" / "data"
    viewer_data_dir.mkdir(parents=True, exist_ok=True)
    with open(viewer_data_dir / "trajectories.json", 'w') as f:
        json.dump(trajectory_data, f)
    print(f"Saved: {viewer_data_dir / 'trajectories.json'}")
    
    print(f"\n4D Reconstruction: {len(trajectory_data['tracks'])} tracks")
    print(f"  Stationary: {sum(1 for t in trajectory_data['tracks'] if t['is_stationary'])}")
    print(f"  Moving: {sum(1 for t in trajectory_data['tracks'] if not t['is_stationary'])}")
    
    return tracks


def main():
    base_dir = Path(__file__).parent.parent
    run_two_pass_tracking(
        video_dir=base_dir / "StreetAware-sample",
        output_dir=base_dir / "outputs" / "pass2_dynamic",
        camera_params_path=base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json",
        spatial_threshold=2.5,
        min_detections=30  # Need at least 30 detections to be a valid track
    )


if __name__ == "__main__":
    main()
