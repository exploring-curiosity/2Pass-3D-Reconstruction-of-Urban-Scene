#!/usr/bin/env python3
"""
Prepare data for nerfstudio Gaussian Splatting training.

Converts our Pi3 output to nerfstudio's expected format.
Then run:
    ns-train splatfacto --data outputs/pass1_static/nerfstudio_data
"""

import sys
from pathlib import Path
import numpy as np
import json
import shutil
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_transforms_json(cameras_path: Path, images_dir: Path, output_path: Path):
    """Create transforms.json in nerfstudio format."""
    
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    frames = []
    
    for cam_name, cam_info in cameras.items():
        img_path = images_dir / f"{cam_name}_bg.png"
        if not img_path.exists():
            print(f"  Skipping {cam_name}: image not found")
            continue
        
        # Get camera-to-world pose
        pose_c2w = np.array(cam_info['pose_c2w'], dtype=np.float64)
        
        # Nerfstudio expects OpenGL convention (Y-up, -Z forward)
        # Our convention is Z-up, so we need to convert
        # OpenGL from OpenCV: rotate 180 around X
        R_cv_to_gl = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        pose_gl = pose_c2w @ R_cv_to_gl
        
        # Get intrinsics
        K = np.array(cam_info['K'], dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = cam_info['width'], cam_info['height']
        
        frame = {
            "file_path": f"images/{cam_name}_bg.png",
            "transform_matrix": pose_gl.tolist(),
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
        }
        frames.append(frame)
    
    # Create transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "frames": frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"  Created {output_path} with {len(frames)} frames")
    return len(frames)


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs" / "pass1_static"
    images_dir = base_dir / "data" / "processed" / "static_backgrounds"
    
    # Create nerfstudio data directory
    ns_data_dir = output_dir / "nerfstudio_data"
    ns_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory and copy images
    ns_images_dir = ns_data_dir / "images"
    ns_images_dir.mkdir(exist_ok=True)
    
    print("Copying images...")
    cameras_path = output_dir / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    for cam_name in cameras.keys():
        src = images_dir / f"{cam_name}_bg.png"
        dst = ns_images_dir / f"{cam_name}_bg.png"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"  Copied {cam_name}_bg.png")
    
    # Create transforms.json
    print("Creating transforms.json...")
    n_frames = create_transforms_json(
        cameras_path,
        images_dir,
        ns_data_dir / "transforms.json"
    )
    
    if n_frames == 0:
        print("Error: No valid frames found!")
        return
    
    print(f"\n=== Data prepared for nerfstudio ===")
    print(f"Data directory: {ns_data_dir}")
    print(f"\nTo train Gaussian Splatting:")
    print(f"  mamba activate gsplat")
    print(f"  ns-train splatfacto --data {ns_data_dir}")
    print(f"\nOr for faster training with fewer iterations:")
    print(f"  ns-train splatfacto --data {ns_data_dir} --max-num-iterations 5000")


if __name__ == "__main__":
    main()
