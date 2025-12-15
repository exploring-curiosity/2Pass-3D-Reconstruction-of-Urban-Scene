#!/usr/bin/env python3
"""
Stage 4: 8-Camera Visualization + Bird's Eye View
=================================================
1. Load scene_4d.json (objects + per-frame poses).
2. Load static point cloud.
3. Render:
   - 4x2 camera grid with bounding boxes + track IDs.
   - Bird's eye view (BEV) panel showing top-down 3D.
4. Output MP4 video.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import colorsys
import sys

# If Open3D is unavailable, fallback to OpenCV-only BEV
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    
class CameraProjector:
    """Projects 3D world points to 2D image."""
    
    def __init__(self, cam_params):
        self.K = np.array(cam_params['K'])
        pose = np.array(cam_params['pose_c2w'])
        self.R_c2w = pose[:3, :3]
        self.t_c2w = pose[:3, 3]
        
        # World to camera
        self.R_w2c = self.R_c2w.T
        self.t_w2c = -self.R_w2c @ self.t_c2w
    
    def project(self, point_3d):
        """Project 3D world point to 2D pixel."""
        p_cam = self.R_w2c @ point_3d + self.t_w2c
        if p_cam[2] <= 0:
            return None
        p_img = self.K @ p_cam
        return p_img[:2] / p_cam[2]

def get_3d_box_corners(pos, dims, rot_quat):
    """Get 8 corners of a 3D box given center, dimensions, and rotation."""
    from scipy.spatial.transform import Rotation as R
    
    L, W, H = dims
    
    # Box corners in local frame (center at origin, Z-up)
    corners_local = np.array([
        [-L/2, -W/2, 0],
        [L/2, -W/2, 0],
        [L/2, W/2, 0],
        [-L/2, W/2, 0],
        [-L/2, -W/2, H],
        [L/2, -W/2, H],
        [L/2, W/2, H],
        [-L/2, W/2, H],
    ])
    
    # Apply rotation
    r = R.from_quat(rot_quat)  # [x, y, z, w]
    corners_rotated = r.apply(corners_local)
    
    # Translate to position
    corners_world = corners_rotated + np.array(pos)
    
    return corners_world

def draw_3d_box_projection(img, corners_2d, color, thickness=2):
    """Draw projected 3D box on image."""
    if corners_2d is None or len(corners_2d) != 8:
        return
    
    # Check if all corners are valid
    if any(c is None for c in corners_2d):
        return
    
    corners_2d = np.array(corners_2d).astype(int)
    
    # Draw bottom face
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    
    # Draw top face
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(img, tuple(corners_2d[i+4]), tuple(corners_2d[j+4]), color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)

def draw_bev(frame_data, objects, bev_size=(400, 400), world_range=40):
    """Draw Bird's Eye View of current frame."""
    bev = np.zeros((bev_size[1], bev_size[0], 3), dtype=np.uint8)
    
    # Gray background
    bev[:] = (40, 40, 40)
    
    # Draw grid
    scale = bev_size[0] / (2 * world_range)
    center = (bev_size[0] // 2, bev_size[1] // 2)
    
    # Grid lines every 10m
    for i in range(-4, 5):
        x = int(center[0] + i * 10 * scale)
        cv2.line(bev, (x, 0), (x, bev_size[1]), (60, 60, 60), 1)
    for i in range(-4, 5):
        y = int(center[1] + i * 10 * scale)
        cv2.line(bev, (0, y), (bev_size[0], y), (60, 60, 60), 1)
    
    # Draw objects
    for obj in frame_data:
        oid = obj['id']
        pos = obj['pos']
        rot = obj['rot']
        
        if oid not in objects:
            continue
        
        obj_info = objects[oid]
        dims = obj_info['dims']
        color = tuple(obj_info['color'])
        
        # Convert to BEV coordinates
        bev_x = int(center[0] + pos[0] * scale)
        bev_y = int(center[1] - pos[1] * scale)  # Flip Y
        
        # Draw oriented rectangle
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(rot)
        yaw = r.as_euler('xyz')[2]
        
        L, W = dims[0], dims[1]
        corners = np.array([
            [-L/2, -W/2],
            [L/2, -W/2],
            [L/2, W/2],
            [-L/2, W/2]
        ])
        
        # Rotate
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_rot = (rot_mat @ corners.T).T
        
        # Scale and translate
        corners_bev = corners_rot * scale
        corners_bev[:, 0] += bev_x
        corners_bev[:, 1] = bev_y - corners_bev[:, 1]  # Flip Y
        
        corners_bev = corners_bev.astype(int)
        
        # Draw filled polygon
        cv2.fillPoly(bev, [corners_bev], color)
        
        # Draw ID
        cv2.putText(bev, str(oid), (bev_x - 5, bev_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return bev

def main():
    base_dir = Path(__file__).parent.parent
    work_dir = base_dir / "pass2_dynamic_new"
    out_dir = base_dir / "outputs" / "pass2_dynamic_new"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scene
    scene_path = work_dir / "scene_4d.json"
    if not scene_path.exists():
        print("ERROR: Run canonical_fitter.py first")
        sys.exit(1)
    
    print("Loading scene...")
    with open(scene_path) as f:
        scene = json.load(f)
    
    # Load camera params
    cameras_path = base_dir / "outputs" / "pass1_static" / "pi3_cameras_corrected.json"
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    objects = scene['objects']
    frames_data = scene['frames']
    total_frames = scene['total_frames']
    fps = scene['fps']
    
    print(f"Scene: {len(objects)} objects, {total_frames} frames")
    
    # Camera order
    cam_ids = ['s1-left', 's1-right', 's2-left', 's2-right',
               's3-left', 's3-right', 's4-left', 's4-right']
    
    # Load video captures
    video_dir = base_dir / "StreetAware-sample"
    caps = {}
    for cam_id in cam_ids:
        vpath = video_dir / f"{cam_id}.mp4"
        if vpath.exists():
            caps[cam_id] = cv2.VideoCapture(str(vpath))
    
    # Get frame size
    test_cap = list(caps.values())[0]
    frame_w = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Grid layout: 4 rows x 2 cols for cameras, plus BEV panel
    grid_w, grid_h = 480, 360  # Each camera cell
    bev_size = (grid_w * 2, grid_h)  # BEV same width as 2 cameras
    
    output_w = grid_w * 2  # 2 columns
    output_h = grid_h * 4 + bev_size[1]  # 4 camera rows + BEV
    
    # Video writer
    out_video = str(out_dir / "reconstruction_4d.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (output_w, output_h))
    
    # Create projectors
    projectors = {cam_id: CameraProjector(cameras[cam_id]) for cam_id in cam_ids if cam_id in cameras}
    
    # Object IDs may be strings (S1, D1, etc.) or integers
    objects_dict = {str(k): v for k, v in objects.items()}
    
    print(f"Rendering {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        # Read all camera frames
        cam_frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                # Resize
                cam_frames[cam_id] = cv2.resize(frame, (grid_w, grid_h))
            else:
                cam_frames[cam_id] = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Get frame data
        frame_key = str(frame_idx)
        current_objects = frames_data.get(frame_key, [])
        
        # Draw on each camera
        for cam_id in cam_ids:
            if cam_id not in cam_frames or cam_id not in projectors:
                continue
            
            frame = cam_frames[cam_id]
            proj = projectors[cam_id]
            
            scale_x = grid_w / frame_w
            scale_y = grid_h / frame_h
            
            for obj in current_objects:
                oid = obj['id']
                if oid not in objects_dict:
                    continue
                
                obj_info = objects_dict[oid]
                pos = np.array(obj['pos'])
                rot = obj['rot']
                dims = obj_info['dims']
                color = tuple(obj_info['color'])
                
                # Get 3D box corners
                corners_3d = get_3d_box_corners(pos, dims, rot)
                
                # Project to image
                corners_2d = []
                valid = True
                for corner in corners_3d:
                    p2d = proj.project(corner)
                    if p2d is None:
                        valid = False
                        break
                    # Scale to grid size
                    p2d_scaled = [p2d[0] * scale_x, p2d[1] * scale_y]
                    corners_2d.append(p2d_scaled)
                
                if valid:
                    draw_3d_box_projection(frame, corners_2d, color, 2)
                    
                    # Draw label at center
                    center_2d = proj.project(pos)
                    if center_2d is not None:
                        cx = int(center_2d[0] * scale_x)
                        cy = int(center_2d[1] * scale_y)
                        label = f"T{oid}"
                        cv2.putText(frame, label, (cx - 15, cy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Camera label
            cv2.putText(frame, cam_id, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Arrange cameras in grid (4 rows, 2 cols)
        grid = np.zeros((grid_h * 4, grid_w * 2, 3), dtype=np.uint8)
        for i, cam_id in enumerate(cam_ids):
            if cam_id in cam_frames:
                row = i // 2
                col = i % 2
                y1, y2 = row * grid_h, (row + 1) * grid_h
                x1, x2 = col * grid_w, (col + 1) * grid_w
                grid[y1:y2, x1:x2] = cam_frames[cam_id]
        
        # Draw BEV
        bev = draw_bev(current_objects, objects_dict, bev_size)
        cv2.putText(bev, f"Frame: {frame_idx}/{total_frames}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Combine grid + BEV
        output_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        output_frame[:grid_h * 4, :] = grid
        output_frame[grid_h * 4:, :] = bev
        
        writer.write(output_frame)
    
    # Cleanup
    writer.release()
    for cap in caps.values():
        cap.release()
    
    print(f"\nSaved: {out_video}")

if __name__ == "__main__":
    main()
