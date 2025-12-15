#!/usr/bin/env python3
"""
Stage 3: Canonical Object Fitter
================================
1. Load 4D tracks.
2. Assign canonical dimensions based on class.
3. Compute heading/orientation from velocity (smoothed).
4. Lock stationary objects (no rotation drift).
5. Output scene descriptor for visualization.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
import sys

# Canonical dimensions (Length, Width, Height) in meters
CANONICAL_DIMS = {
    'car': [4.5, 1.8, 1.5],
    'truck': [8.0, 2.5, 3.0],
    'bus': [12.0, 2.5, 3.2],
    'motorcycle': [2.2, 0.8, 1.5],
    'bicycle': [1.8, 0.6, 1.5],
    'person': [0.5, 0.5, 1.7]  # Cylinder approximated as box
}

# Class colors (BGR for OpenCV)
CLASS_COLORS = {
    'car': [0, 0, 255],      # Red
    'truck': [255, 0, 255],  # Magenta
    'bus': [0, 255, 255],    # Yellow
    'motorcycle': [255, 128, 0],  # Cyan-ish
    'bicycle': [0, 255, 128],     # Green-ish
    'person': [0, 255, 0]    # Green
}

def compute_heading_from_trajectory(positions: np.ndarray, is_stationary: bool) -> np.ndarray:
    """
    Compute heading angles (yaw) from trajectory positions.
    Returns array of yaw angles in radians.
    """
    n = len(positions)
    if n < 2:
        return np.zeros(n)
    
    # Compute velocities (forward differences)
    velocities = np.zeros((n, 2))
    for i in range(n - 1):
        velocities[i] = positions[i + 1, :2] - positions[i, :2]
    velocities[-1] = velocities[-2]  # Copy last
    
    # If stationary, use average heading
    if is_stationary:
        avg_vel = np.mean(velocities, axis=0)
        if np.linalg.norm(avg_vel) > 0.01:
            yaw = np.arctan2(avg_vel[1], avg_vel[0])
        else:
            yaw = 0.0
        return np.full(n, yaw)
    
    # Compute yaw from velocity
    yaws = np.zeros(n)
    for i in range(n):
        v = velocities[i]
        speed = np.linalg.norm(v)
        if speed > 0.1:  # Moving
            yaws[i] = np.arctan2(v[1], v[0])
        else:
            # Slow/stopped: use previous heading
            yaws[i] = yaws[i - 1] if i > 0 else 0.0
    
    # Smooth headings (unwrap first)
    yaws_unwrapped = np.unwrap(yaws)
    yaws_smooth = uniform_filter1d(yaws_unwrapped, size=5, mode='nearest')
    
    return yaws_smooth

def yaw_to_quaternion(yaw: float) -> list:
    """Convert yaw angle to quaternion [x, y, z, w]."""
    r = R.from_euler('z', yaw)
    q = r.as_quat()  # [x, y, z, w]
    return q.tolist()

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    
    tracks_path = work_dir / "tracks_4d.json"
    if not tracks_path.exists():
        print("ERROR: Run physics_tracker.py first")
        sys.exit(1)
    
    print("Loading tracks...")
    with open(tracks_path) as f:
        data = json.load(f)
    
    tracks = data['tracks']
    total_frames = data['total_frames']
    fps = data['fps']
    
    print(f"Loaded {len(tracks)} tracks, {total_frames} frames @ {fps:.1f} FPS")
    
    # Build scene descriptor
    scene = {
        'total_frames': total_frames,
        'fps': fps,
        'objects': {},  # id -> {class, dims, color, is_stationary}
        'frames': {}    # frame -> [{id, pos, rot}]
    }
    
    for track in tracks:
        track_id = track['id']
        cls = track['class']
        is_stationary = track['is_stationary']
        trajectory = track['trajectory']
        
        # Object definition
        dims = CANONICAL_DIMS.get(cls, CANONICAL_DIMS['car'])
        color = CLASS_COLORS.get(cls, [255, 255, 255])
        
        scene['objects'][track_id] = {
            'class': cls,
            'dims': dims,
            'color': color,
            'is_stationary': is_stationary
        }
        
        # Extract positions
        positions = np.array([node['pos'] for node in trajectory])
        frames = [node['frame'] for node in trajectory]
        
        # Compute headings
        headings = compute_heading_from_trajectory(positions, is_stationary)
        
        # Add to frames
        for i, node in enumerate(trajectory):
            frame_key = str(node['frame'])
            if frame_key not in scene['frames']:
                scene['frames'][frame_key] = []
            
            quat = yaw_to_quaternion(headings[i])
            
            scene['frames'][frame_key].append({
                'id': track_id,
                'pos': node['pos'],
                'rot': quat,
                'conf': node['conf']
            })
    
    # Save
    out_path = work_dir / "scene_4d.json"
    print(f"Saving scene to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"  {len(scene['objects'])} objects")
    print(f"  {len(scene['frames'])} frames with data")

if __name__ == "__main__":
    main()
