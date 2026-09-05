#!/usr/bin/env python3
"""
Test π³ (Pi3) on the 8-camera static backgrounds.

This script compares Pi3 with DUSt3R for multi-view 3D reconstruction.

NOTE: This test requires GPU, background images, and the Pi3 model from
HuggingFace. It is skipped in CI/local environments where these are unavailable.
"""

import sys
import pytest
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "Pi3"))

from utils import load_config, setup_logger, save_ply

# Skip all tests in this module if GPU, images, or Pi3 model are unavailable
pytestmark = pytest.mark.skipif(
    True,
    reason="Requires GPU, background images, and Pi3 model from HuggingFace (unavailable locally)"
)


def load_background_images(config) -> dict:
    """Load all static background images."""
    bg_dir = Path(config["data"]["processed_dir"]) / "static_backgrounds"
    cameras = config["data"]["cameras"]
    
    images = {}
    for cam_name in cameras:
        img_path = bg_dir / f"{cam_name}_bg.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[cam_name] = img
            print(f"Loaded {cam_name}: {img.shape}")
    
    return images


def run_pi3_reconstruction(images: dict, output_dir: Path, device: str = "cuda"):
    """Run Pi3 reconstruction on the images."""
    from pi3.models.pi3 import Pi3
    
    print("\n=== Running π³ (Pi3) Reconstruction ===")
    
    # Load model
    print("Loading Pi3 model from HuggingFace...")
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    
    # Prepare images
    cam_names = list(images.keys())
    img_list = [images[name] for name in cam_names]
    
    # Convert to tensor: [N, 3, H, W] with values in [0, 1]
    # Pi3 expects resolution divisible by patch size (14)
    # Use 518x392 (closest to 512x384 that's divisible by 14)
    target_h, target_w = 392, 518
    
    imgs_resized = []
    for img in img_list:
        img_resized = cv2.resize(img, (target_w, target_h))
        imgs_resized.append(img_resized)
    
    # Stack and convert to tensor
    imgs_np = np.stack(imgs_resized, axis=0)  # [N, H, W, 3]
    imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0  # [N, 3, H, W]
    imgs_tensor = imgs_tensor.to(device)
    
    print(f"Input tensor shape: {imgs_tensor.shape}")
    
    # Run inference
    print("Running Pi3 inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Add batch dimension -> (1, N, 3, H, W)
            results = model(imgs_tensor[None])
    
    print("Reconstruction complete!")
    
    # Extract outputs
    points = results['points'][0].cpu().numpy()  # [N, H, W, 3]
    local_points = results['local_points'][0].cpu().numpy()  # [N, H, W, 3]
    conf = torch.sigmoid(results['conf'][0]).cpu().numpy()  # [N, H, W, 1]
    camera_poses = results['camera_poses'][0].cpu().numpy()  # [N, 4, 4]
    
    print(f"Points shape: {points.shape}")
    print(f"Confidence shape: {conf.shape}")
    print(f"Camera poses shape: {camera_poses.shape}")
    
    # Print camera poses
    print("\nCamera poses (camera-to-world):")
    for i, cam_name in enumerate(cam_names):
        pose = camera_poses[i]
        pos = pose[:3, 3]
        print(f"  {cam_name}: position = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Collect all points with colors
    all_points = []
    all_colors = []
    all_conf = []
    
    for i, cam_name in enumerate(cam_names):
        pts = points[i].reshape(-1, 3)  # [H*W, 3]
        c = conf[i].reshape(-1)  # [H*W]
        
        # Get colors from resized image
        colors = imgs_resized[i].reshape(-1, 3)  # [H*W, 3]
        
        all_points.append(pts)
        all_colors.append(colors)
        all_conf.append(c)
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    all_conf = np.concatenate(all_conf)
    
    print(f"\nTotal points: {len(all_points)}")
    print(f"Confidence range: [{all_conf.min():.3f}, {all_conf.max():.3f}]")
    
    # Filter by confidence
    conf_threshold = np.percentile(all_conf, 20)  # Keep top 80%
    mask = all_conf > conf_threshold
    
    filtered_points = all_points[mask]
    filtered_colors = all_colors[mask]
    
    print(f"After confidence filter (>{conf_threshold:.3f}): {len(filtered_points)} points")
    
    # Filter outliers
    center = filtered_points.mean(axis=0)
    distances = np.linalg.norm(filtered_points - center, axis=1)
    max_dist = np.percentile(distances, 95)
    
    mask2 = distances < max_dist
    filtered_points = filtered_points[mask2]
    filtered_colors = filtered_colors[mask2]
    
    print(f"After outlier filter: {len(filtered_points)} points")
    
    # Save point cloud
    output_file = output_dir / "pi3_pointcloud.ply"
    save_ply(str(output_file), filtered_points, filtered_colors)
    print(f"\n✓ Saved Pi3 point cloud to {output_file}")
    
    # Save camera poses
    import json
    cameras_out = {}
    for i, cam_name in enumerate(cam_names):
        pose = camera_poses[i]
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Intrinsics (approximate from image size)
        fx = fy = target_w * 0.6  # Rough estimate
        cx, cy = target_w / 2, target_h / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        cameras_out[cam_name] = {
            "K": K.tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
            "pose_c2w": pose.tolist(),
            "width": target_w,
            "height": target_h,
            "camera_id": cam_name
        }
    
    cameras_file = output_dir / "pi3_cameras.json"
    with open(cameras_file, "w") as f:
        json.dump(cameras_out, f, indent=2)
    print(f"✓ Saved Pi3 cameras to {cameras_file}")
    
    return filtered_points, filtered_colors, cameras_out


def main():
    config = load_config()
    logger = setup_logger(
        name="TestPi3",
        log_dir=config["data"]["log_dir"],
        level="INFO",
        save_to_file=False
    )
    
    output_dir = Path(config["data"]["output_dir"]) / "pass1_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    images = load_background_images(config)
    
    if len(images) == 0:
        print("No background images found!")
        return
    
    # Run Pi3
    device = config["hardware"]["device"]
    run_pi3_reconstruction(images, output_dir, device)
    
    print("\n=== Done! ===")
    print("Compare outputs:")
    print(f"  DUSt3R: {output_dir / 'dust3r_pointcloud.ply'}")
    print(f"  Pi3:    {output_dir / 'pi3_pointcloud.ply'}")


if __name__ == "__main__":
    main()
