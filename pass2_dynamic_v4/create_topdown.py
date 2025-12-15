#!/usr/bin/env python3
"""
Create proper top-down visualization from point cloud
matching the user's reference image orientation.
"""

import open3d as o3d
import numpy as np
import cv2
from pathlib import Path

def main():
    base = Path(__file__).parent.parent
    out_dir = base / "outputs" / "pass2_dynamic_v4"
    
    # Load point cloud
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(base / 'outputs/pass1_static/pi3_pointcloud_corrected.ply'))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"Points: {len(points)}")
    print(f"X: {points[:, 0].min():.1f} to {points[:, 0].max():.1f}")
    print(f"Y: {points[:, 1].min():.1f} to {points[:, 1].max():.1f}")
    print(f"Z: {points[:, 2].min():.1f} to {points[:, 2].max():.1f}")
    
    # Create top-down view (X-Y plane, Z is up)
    # Looking from above (like user's image)
    size = 900
    scale = 18  # pixels per meter
    cx, cy = size // 2, size // 2
    
    # Create image with black background
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Sample points (every 3rd for speed)
    sample_idx = np.arange(0, len(points), 3)
    sampled_pts = points[sample_idx]
    sampled_colors = colors[sample_idx]
    
    print(f"Drawing {len(sampled_pts)} points...")
    
    # Draw points colored by their actual RGB
    for i, (pt, col) in enumerate(zip(sampled_pts, sampled_colors)):
        # X goes right, Y goes up in world coords
        # In image: X goes right, Y goes down
        px = int(cx + pt[0] * scale)
        py = int(cy - pt[1] * scale)  # Flip Y for image coordinates
        
        if 0 <= px < size and 0 <= py < size:
            # Convert RGB (0-1) to BGR (0-255)
            bgr = (int(col[2] * 255), int(col[1] * 255), int(col[0] * 255))
            img[py, px] = bgr
    
    # Draw coordinate markers
    cv2.putText(img, "Top-Down View from Point Cloud", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Scale: {scale} px/m", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    # Draw center cross
    cv2.line(img, (cx-10, cy), (cx+10, cy), (100, 100, 100), 1)
    cv2.line(img, (cx, cy-10), (cx, cy+10), (100, 100, 100), 1)
    
    # Save
    out_path = out_dir / "pointcloud_topdown.png"
    cv2.imwrite(str(out_path), img)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
