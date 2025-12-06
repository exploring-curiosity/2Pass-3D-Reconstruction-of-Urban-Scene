#!/usr/bin/env python3
"""
Fix Pi3 point cloud orientation and prepare for Gaussian Splatting.

The Pi3 output is in an arbitrary coordinate frame. We need to:
1. Rotate so ground plane is horizontal (Z-up)
2. Center the scene
3. Scale to reasonable metric units
4. Export in COLMAP format for Gaussian Splatting
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_ply, save_ply


def estimate_ground_plane(points: np.ndarray, percentile: float = 10) -> tuple:
    """Estimate ground plane from lowest points."""
    # Get the lowest points (by Y in Pi3 frame, which seems to be "down")
    # Actually, looking at the image, the scene is tilted
    # We need to find the dominant plane
    
    from sklearn.decomposition import PCA
    
    # Use lowest 10% of points by each axis to find ground
    z_threshold = np.percentile(points[:, 2], percentile)
    ground_candidates = points[points[:, 2] < z_threshold]
    
    if len(ground_candidates) < 100:
        # Fallback: use PCA on all points
        ground_candidates = points
    
    # Fit plane using PCA
    pca = PCA(n_components=3)
    pca.fit(ground_candidates)
    
    # The normal to the ground plane is the component with smallest variance
    normal = pca.components_[2]  # Smallest eigenvalue direction
    
    # Make sure normal points "up" (positive Z in final frame)
    if normal[2] < 0:
        normal = -normal
    
    return normal, ground_candidates.mean(axis=0)


def compute_rotation_to_align_ground(normal: np.ndarray) -> np.ndarray:
    """Compute rotation matrix to align ground normal with Z-axis."""
    # Target: Z-up (0, 0, 1)
    target = np.array([0, 0, 1])
    
    # Rotation axis is cross product
    axis = np.cross(normal, target)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # Already aligned
        return np.eye(3)
    
    axis = axis / axis_norm
    
    # Rotation angle
    angle = np.arccos(np.clip(np.dot(normal, target), -1, 1))
    
    # Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R


def fix_orientation(input_ply: Path, output_ply: Path, cameras_in: Path, cameras_out: Path):
    """Fix point cloud orientation and update cameras."""
    
    print("Loading Pi3 point cloud...")
    points, colors, _ = load_ply(str(input_ply))
    print(f"  Loaded {len(points)} points")
    print(f"  Original extent: {points.min(axis=0)} to {points.max(axis=0)}")
    
    # Load cameras
    with open(cameras_in) as f:
        cameras = json.load(f)
    
    # Step 1: Center the point cloud
    center = points.mean(axis=0)
    points_centered = points - center
    print(f"  Centered at: {center}")
    
    # Step 2: Estimate ground plane and compute alignment rotation
    print("Estimating ground plane...")
    
    # For traffic intersection, ground should be relatively flat
    # Looking at camera positions, they form a rough square around the intersection
    # The cameras are at positions like (0.3, 0.36, -0.65), (1.08, 0.05, 0.03), etc.
    # This suggests the scene is roughly in XY plane with Z varying
    
    # Let's try a different approach: use the camera positions to estimate "up"
    cam_positions = []
    for cam_name, cam_data in cameras.items():
        pos = np.array(cam_data['t'])
        cam_positions.append(pos)
    cam_positions = np.array(cam_positions)
    
    print(f"  Camera positions:\n{cam_positions}")
    
    # The cameras should be roughly at the same height above ground
    # Find the principal axes of camera positions
    cam_center = cam_positions.mean(axis=0)
    cam_centered = cam_positions - cam_center
    
    # SVD to find principal directions
    U, S, Vt = np.linalg.svd(cam_centered)
    
    # The direction with smallest variance is likely the "up" direction
    # (cameras are spread horizontally, not vertically)
    up_direction = Vt[2]  # Smallest singular value direction
    
    # Make sure it points in a consistent direction
    if up_direction[1] < 0:  # Assuming Y should be positive for "up" initially
        up_direction = -up_direction
    
    print(f"  Estimated up direction: {up_direction}")
    
    # Compute rotation to make this the Z-axis
    R_align = compute_rotation_to_align_ground(up_direction)
    
    # Apply rotation to points
    points_rotated = (R_align @ points_centered.T).T
    
    # Step 3: Additional rotations to align with expected view
    # First flip upside down (rotate 180 around X axis)
    R_flip = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    points_flipped = (R_flip @ points_rotated.T).T
    
    # Then rotate around Z to align roads with X/Y axes
    R_z = Rotation.from_euler('z', -45, degrees=True).as_matrix()
    points_final = (R_z @ points_flipped.T).T
    
    # Combined rotation
    R_total = R_z @ R_flip @ R_align
    
    # Step 4: Scale to approximate metric units
    # Assuming the intersection is roughly 30-40 meters across
    current_extent = points_final.max(axis=0) - points_final.min(axis=0)
    target_extent = 40.0  # meters
    scale = target_extent / current_extent.max()
    
    points_scaled = points_final * scale
    
    print(f"  Applied scale: {scale:.2f}")
    print(f"  Final extent: {points_scaled.min(axis=0)} to {points_scaled.max(axis=0)}")
    
    # Step 5: Shift so ground is at Z=0
    z_min = np.percentile(points_scaled[:, 2], 5)
    points_scaled[:, 2] -= z_min
    
    print(f"  Ground level adjusted, Z range: [{points_scaled[:, 2].min():.2f}, {points_scaled[:, 2].max():.2f}]")
    
    # Save corrected point cloud
    save_ply(str(output_ply), points_scaled, colors)
    print(f"  Saved corrected point cloud to {output_ply}")
    
    # Step 6: Update camera parameters
    cameras_corrected = {}
    for cam_name, cam_data in cameras.items():
        # Original pose (camera-to-world)
        pose_c2w = np.array(cam_data['pose_c2w'])
        R_orig = pose_c2w[:3, :3]
        t_orig = pose_c2w[:3, 3]
        
        # Apply transformations to camera position
        t_centered = t_orig - center
        t_rotated = R_total @ t_centered
        t_scaled = t_rotated * scale
        t_scaled[2] -= z_min
        
        # Apply rotation to camera orientation
        R_new = R_total @ R_orig
        
        # Scale intrinsics to full resolution (2592x1944 from 518x392)
        K_orig = np.array(cam_data['K'])
        scale_x = 2592 / 518
        scale_y = 1944 / 392
        
        K_scaled = np.array([
            [K_orig[0][0] * scale_x, 0, K_orig[0][2] * scale_x],
            [0, K_orig[1][1] * scale_y, K_orig[1][2] * scale_y],
            [0, 0, 1]
        ])
        
        # Build new pose
        pose_new = np.eye(4)
        pose_new[:3, :3] = R_new
        pose_new[:3, 3] = t_scaled
        
        cameras_corrected[cam_name] = {
            "K": K_scaled.tolist(),
            "R": R_new.tolist(),
            "t": t_scaled.tolist(),
            "pose_c2w": pose_new.tolist(),
            "width": 2592,
            "height": 1944,
            "camera_id": cam_name
        }
        
        print(f"  {cam_name}: pos = [{t_scaled[0]:.2f}, {t_scaled[1]:.2f}, {t_scaled[2]:.2f}]")
    
    # Save corrected cameras
    with open(cameras_out, 'w') as f:
        json.dump(cameras_corrected, f, indent=2)
    print(f"  Saved corrected cameras to {cameras_out}")
    
    return points_scaled, colors, cameras_corrected, {
        'center': center.tolist(),
        'R_total': R_total.tolist(),
        'scale': scale,
        'z_offset': z_min
    }


def export_colmap_format(points: np.ndarray, colors: np.ndarray, 
                         cameras: dict, output_dir: Path):
    """Export in COLMAP format for Gaussian Splatting."""
    
    colmap_dir = output_dir / "colmap_sparse"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    # cameras.txt
    cameras_txt = colmap_dir / "cameras.txt"
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (cam_name, cam_data) in enumerate(cameras.items(), 1):
            K = np.array(cam_data['K'])
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            w, h = cam_data['width'], cam_data['height']
            # PINHOLE model: fx, fy, cx, cy
            f.write(f"{i} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    # images.txt
    images_txt = colmap_dir / "images.txt"
    with open(images_txt, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (cam_name, cam_data) in enumerate(cameras.items(), 1):
            # COLMAP uses world-to-camera, so invert the pose
            pose_c2w = np.array(cam_data['pose_c2w'])
            R_c2w = pose_c2w[:3, :3]
            t_c2w = pose_c2w[:3, 3]
            
            # World-to-camera
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            
            # Convert to quaternion (COLMAP uses qw, qx, qy, qz)
            rot = Rotation.from_matrix(R_w2c)
            quat = rot.as_quat()  # Returns [qx, qy, qz, qw]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{i} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i} {cam_name}_bg.png\n")
            f.write("\n")  # Empty line for POINTS2D
    
    # points3D.txt (sparse points)
    points3d_txt = colmap_dir / "points3D.txt"
    
    # Subsample points for sparse representation
    n_sparse = min(50000, len(points))
    indices = np.random.choice(len(points), n_sparse, replace=False)
    sparse_points = points[indices]
    sparse_colors = colors[indices]
    
    with open(points3d_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (pt, col) in enumerate(zip(sparse_points, sparse_colors), 1):
            f.write(f"{i} {pt[0]} {pt[1]} {pt[2]} {int(col[0])} {int(col[1])} {int(col[2])} 0.0\n")
    
    print(f"  Exported COLMAP format to {colmap_dir}")
    print(f"  - cameras.txt: {len(cameras)} cameras")
    print(f"  - images.txt: {len(cameras)} images")
    print(f"  - points3D.txt: {n_sparse} sparse points")
    
    return colmap_dir


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs" / "pass1_static"
    
    input_ply = output_dir / "pi3_pointcloud.ply"
    output_ply = output_dir / "pi3_pointcloud_corrected.ply"
    cameras_in = output_dir / "pi3_cameras.json"
    cameras_out = output_dir / "pi3_cameras_corrected.json"
    
    if not input_ply.exists():
        print(f"Error: {input_ply} not found. Run test_pi3.py first.")
        return
    
    print("=== Fixing Pi3 Orientation ===\n")
    
    points, colors, cameras, transform = fix_orientation(
        input_ply, output_ply, cameras_in, cameras_out
    )
    
    # Save transform for later use
    transform_file = output_dir / "pi3_transform.json"
    with open(transform_file, 'w') as f:
        json.dump(transform, f, indent=2)
    print(f"  Saved transform to {transform_file}")
    
    print("\n=== Exporting COLMAP Format ===\n")
    export_colmap_format(points, colors, cameras, output_dir)
    
    print("\n=== Done! ===")
    print(f"Corrected point cloud: {output_ply}")
    print(f"Corrected cameras: {cameras_out}")
    print(f"COLMAP sparse: {output_dir / 'colmap_sparse'}")


if __name__ == "__main__":
    main()
