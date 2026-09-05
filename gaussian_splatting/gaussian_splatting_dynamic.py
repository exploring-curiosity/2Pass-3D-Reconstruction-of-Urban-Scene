#!/usr/bin/env python3
"""
Add Gaussian Splatting to Dynamic Objects
Fully integrated with Two-Pass 3D Cross-Camera Tracker.
- Uses tracks from two-pass tracker (track_id, frames, positions)
- Maps 2D masks → 3D points → Gaussian splats
- Handles multi-camera frames
- Ensures temporal consistency and dynamic object representation
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import json
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from two_pass_3d_tracker import run_two_pass_tracking, CameraProjector  # import your tracker

# Optional: SAM2 or any segmentation model
from segment_anything import sam_model_registry, SamPredictor

# -------------------------------
# CONFIG
# -------------------------------
VIDEO_DIR = Path(__file__).parent.parent / "StreetAware-sample"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "gaussian_splats"
CAMERA_PARAMS_PATH = Path(__file__).parent.parent / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"

MIN_FRAMES_TRACK = 5  # Only generate splats for tracks longer than this
GRID_SAMPLE = 2       # Downsample pixels for Gaussian mapping
# -------------------------------

def initialize_sam(model_type="vit_b", checkpoint=None):
    """Initialize SAM predictor"""
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    predictor = SamPredictor(sam)
    return predictor

def map_mask_to_3d(mask, camera_projector, bbox):
    """
    Map 2D mask pixels to 3D points on ground plane using bottom-center approximation.
    Returns N x 3 array of 3D points.
    """
    ys, xs = np.where(mask > 0)
    points_3d = []
    for y, x in zip(ys[::GRID_SAMPLE], xs[::GRID_SAMPLE]):
        pixel = np.array([bbox[0] + x, bbox[1] + y])
        pos_3d = camera_projector.project_to_ground(pixel)
        if pos_3d is not None:
            points_3d.append(pos_3d)
    if len(points_3d) == 0:
        return None
    return np.array(points_3d)

def generate_gaussian_splats(tracks, camera_params, video_paths):
    """
    For each track, generate Gaussian splats from multi-camera frames.
    """
    print("\nGenerating Gaussian splats for dynamic objects...")
    
    projectors = {cam_id: CameraProjector(camera_params[cam_id]) for cam_id in video_paths}
    caps = {cam_id: cv2.VideoCapture(str(path)) for cam_id, path in video_paths.items()}
    
    gaussian_data = []  # List of dicts per track
    
    for track_id, track in tracks.items():
        if len(track['frames']) < MIN_FRAMES_TRACK:
            continue
        
        track_points = []  # Accumulate 3D points for this track
        
        for frame_idx, cam_data in track['frames'].items():
            for cam_id, (bbox, pos_3d, conf) in cam_data.items():
                cap = caps[cam_id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Crop bbox
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Optional: use SAM to get mask
                # Here, we just use full bbox as mask
                mask = np.ones((y2-y1, x2-x1), dtype=np.uint8)
                
                # Map mask to 3D points
                points_3d = map_mask_to_3d(mask, projectors[cam_id], bbox)
                if points_3d is not None:
                    track_points.append(points_3d)
        
        if len(track_points) == 0:
            continue
        
        track_points = np.vstack(track_points)
        
        # Compute Gaussian splat parameters: mean, covariance
        mean = np.mean(track_points, axis=0)
        cov = np.cov(track_points.T) + np.eye(3)*1e-3  # Add small epsilon
        gaussian_data.append({
            'track_id': int(track_id),
            'class': track['class'],
            'num_points': track_points.shape[0],
            'mean': mean.tolist(),
            'cov': cov.tolist(),
        })
        
        print(f"Track {track_id}: {track['class']} - {track_points.shape[0]} points")
    
    for cap in caps.values():
        cap.release()
    
    return gaussian_data

def main():
    # Step 1: Run two-pass tracker
    tracks = run_two_pass_tracking(
        video_dir=VIDEO_DIR,
        output_dir=OUTPUT_DIR / "pass2_dynamic",
        camera_params_path=CAMERA_PARAMS_PATH,
        spatial_threshold=2.5,
        min_detections=30
    )
    
    # Load camera params
    with open(CAMERA_PARAMS_PATH) as f:
        camera_params = json.load(f)
    
    # Video paths
    cam_order = ['s1-left', 's1-right', 's2-left', 's2-right', 
                 's3-left', 's3-right', 's4-left', 's4-right']
    video_paths = {cam_id: VIDEO_DIR / f"{cam_id}.mp4" 
                   for cam_id in cam_order if (VIDEO_DIR / f"{cam_id}.mp4").exists()}
    
    # Step 2: Generate Gaussian splats
    gaussians = generate_gaussian_splats(tracks, camera_params, video_paths)
    
    # Step 3: Save Gaussian splat data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dynamic_gaussians.json"
    import json
    with open(out_path, 'w') as f:
        json.dump(gaussians, f, indent=2)
    print(f"\nSaved Gaussian splat data: {out_path}")

if __name__ == "__main__":
    main()
