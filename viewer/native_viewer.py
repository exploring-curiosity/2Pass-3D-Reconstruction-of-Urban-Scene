#!/usr/bin/env python3
"""
Native 4D Scene Viewer using Open3D.

Features:
- Full resolution point cloud (no downsampling)
- Dynamic object boxes with trajectory animation
- Keyboard controls for playback
- Mouse controls for camera orbit/pan/zoom

Controls:
- SPACE: Play/Pause
- LEFT/RIGHT: Step frame backward/forward
- R: Reset to frame 0
- +/-: Speed up/slow down
- Q: Quit
"""

import sys
from pathlib import Path
import numpy as np
import json
import time
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    print("Open3D not installed. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "open3d"])
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

from utils import load_ply


# Box sizes in meters
VEHICLE_SIZE = np.array([4.5, 1.8, 1.5])  # length, width, height
PERSON_SIZE = np.array([0.5, 0.5, 1.7])

# Colors (RGB 0-1)
COLORS = {
    'moving_vehicle': [1.0, 0.2, 0.2],    # Red
    'stationary_vehicle': [1.0, 0.6, 0.0], # Orange
    'moving_person': [0.2, 1.0, 0.2],      # Green
    'stationary_person': [0.2, 1.0, 1.0],  # Cyan
}


class Track:
    """A tracked object with 3D positions over time."""
    
    def __init__(self, track_id: int, class_name: str, category: str, is_stationary: bool):
        self.track_id = track_id
        self.class_name = class_name
        self.category = category
        self.is_stationary = is_stationary
        self.frames: Dict[int, np.ndarray] = {}  # frame_idx -> [x, y, z, heading]
    
    def get_position(self, frame: int) -> Optional[np.ndarray]:
        return self.frames.get(frame)
    
    def get_color(self) -> List[float]:
        if self.category == 'vehicle':
            return COLORS['stationary_vehicle'] if self.is_stationary else COLORS['moving_vehicle']
        return COLORS['stationary_person'] if self.is_stationary else COLORS['moving_person']
    
    def get_size(self) -> np.ndarray:
        return VEHICLE_SIZE if self.category == 'vehicle' else PERSON_SIZE


def create_box_mesh(size: np.ndarray, color: List[float]) -> o3d.geometry.TriangleMesh:
    """Create a box mesh with given size and color."""
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0], height=size[2], depth=size[1]
    )
    # Center the box
    box.translate([-size[0]/2, 0, -size[1]/2])
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    return box


def create_wireframe_box(size: np.ndarray, color: List[float]) -> o3d.geometry.LineSet:
    """Create a wireframe box."""
    points = [
        [-size[0]/2, 0, -size[1]/2],
        [size[0]/2, 0, -size[1]/2],
        [size[0]/2, 0, size[1]/2],
        [-size[0]/2, 0, size[1]/2],
        [-size[0]/2, size[2], -size[1]/2],
        [size[0]/2, size[2], -size[1]/2],
        [size[0]/2, size[2], size[1]/2],
        [-size[0]/2, size[2], size[1]/2],
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Sides
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


class Viewer4D:
    """Interactive 4D scene viewer."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.pass1_dir = output_dir / "pass1_static"
        self.pass2_dir = output_dir / "pass2_dynamic"
        
        # State
        self.current_frame = 0
        self.min_frame = 0
        self.max_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.fps = 15.0
        self.last_update_time = 0
        
        # Data
        self.point_cloud = None
        self.tracks: List[Track] = []
        self.box_geometries: Dict[int, o3d.geometry.LineSet] = {}
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load point cloud and trajectories."""
        
        # Load point cloud (FULL resolution)
        print("Loading point cloud (full resolution)...")
        ply_path = self.pass1_dir / "pi3_pointcloud_corrected.ply"
        points, colors, _ = load_ply(str(ply_path))
        print(f"  Loaded {len(points)} points")
        
        # Create Open3D point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Load trajectories with filtering
        print("Loading trajectories...")
        self.load_trajectories()
        print(f"  Loaded {len(self.tracks)} tracks")
        
        # Compute frame range
        all_frames = set()
        for track in self.tracks:
            all_frames.update(track.frames.keys())
        
        if all_frames:
            self.min_frame = min(all_frames)
            self.max_frame = max(all_frames)
        
        print(f"  Frame range: {self.min_frame} - {self.max_frame}")
    
    def load_trajectories(self):
        """Load unified multi-camera trajectories."""
        
        # Use multi-camera trajectories
        multicam_path = self.pass2_dir / "multicam_trajectories.json"
        
        if not multicam_path.exists():
            print("  WARNING: No multicam trajectories. Run multicam_tracker.py first.")
            return
        
        with open(multicam_path) as f:
            data = json.load(f)
        
        print(f"  Found {data['num_tracks']} unified tracks")
        
        for traj in data['trajectories']:
            # Only moving vehicles (>5m)
            if traj['total_movement_m'] < 5.0:
                continue
            
            track = Track(
                track_id=traj['track_id'],
                class_name=traj['class_name'],
                category='vehicle',
                is_stationary=False
            )
            
            for frame in traj['frames']:
                pos = frame['position_3d']
                heading = frame.get('heading', 0)
                if abs(pos[0]) < 30 and abs(pos[1]) < 30:
                    track.frames[frame['frame_idx']] = np.array([pos[0], pos[1], pos[2], heading])
            
            if len(track.frames) >= 15:
                self.tracks.append(track)
        
        print(f"  Loaded {len(self.tracks)} moving vehicle tracks")
    
    def create_box_at_position(self, track: Track, pos: np.ndarray) -> o3d.geometry.LineSet:
        """Create a wireframe box at the given position with heading."""
        size = track.get_size()
        color = track.get_color()
        
        box = create_wireframe_box(size, color)
        
        # Apply heading rotation (around Z axis since we're Z-up)
        if len(pos) > 3:
            heading = pos[3]
            R = box.get_rotation_matrix_from_xyz([0, 0, heading])
            box.rotate(R, center=[0, 0, 0])
        
        # Translate to position
        box.translate([pos[0], pos[1], pos[2]])
        
        return box
    
    def run(self):
        """Run the viewer."""
        
        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("4D Scene Viewer", width=1600, height=900)
        
        # Add point cloud
        vis.add_geometry(self.point_cloud)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        vis.add_geometry(coord_frame)
        
        # Add ground grid
        grid_lines = []
        grid_size = 40
        grid_step = 2
        for i in range(-grid_size//2, grid_size//2 + 1, grid_step):
            grid_lines.append([[i, -grid_size//2, 0], [i, grid_size//2, 0]])
            grid_lines.append([[-grid_size//2, i, 0], [grid_size//2, i, 0]])
        
        grid = o3d.geometry.LineSet()
        points = []
        lines = []
        for i, line in enumerate(grid_lines):
            points.extend(line)
            lines.append([i*2, i*2+1])
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3]] * len(lines))
        vis.add_geometry(grid)
        
        # Initial box geometries
        for track in self.tracks:
            pos = track.get_position(self.current_frame)
            if pos is not None:
                box = self.create_box_at_position(track, pos)
                self.box_geometries[track.track_id] = box
                vis.add_geometry(box)
        
        # Set render options
        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.1, 0.1, 0.15])
        
        # Set up camera
        ctr = vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0.5, -0.5, 0.7])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # Key callbacks
        def toggle_play(vis):
            self.is_playing = not self.is_playing
            print(f"{'Playing' if self.is_playing else 'Paused'} at frame {self.current_frame}")
            return False
        
        def next_frame(vis):
            self.current_frame = min(self.current_frame + 1, self.max_frame)
            self.update_boxes(vis)
            print(f"Frame: {self.current_frame}/{self.max_frame}")
            return False
        
        def prev_frame(vis):
            self.current_frame = max(self.current_frame - 1, self.min_frame)
            self.update_boxes(vis)
            print(f"Frame: {self.current_frame}/{self.max_frame}")
            return False
        
        def reset_frame(vis):
            self.current_frame = self.min_frame
            self.update_boxes(vis)
            print(f"Reset to frame {self.current_frame}")
            return False
        
        def speed_up(vis):
            self.playback_speed = min(self.playback_speed * 1.5, 5.0)
            print(f"Speed: {self.playback_speed:.1f}x")
            return False
        
        def slow_down(vis):
            self.playback_speed = max(self.playback_speed / 1.5, 0.1)
            print(f"Speed: {self.playback_speed:.1f}x")
            return False
        
        vis.register_key_callback(ord(" "), toggle_play)
        vis.register_key_callback(262, next_frame)  # Right arrow
        vis.register_key_callback(263, prev_frame)  # Left arrow
        vis.register_key_callback(ord("R"), reset_frame)
        vis.register_key_callback(ord("="), speed_up)
        vis.register_key_callback(ord("-"), slow_down)
        
        print("\n=== 4D Scene Viewer ===")
        print("Controls:")
        print("  SPACE: Play/Pause")
        print("  LEFT/RIGHT: Step frame")
        print("  R: Reset to frame 0")
        print("  +/-: Speed up/slow down")
        print("  Mouse: Rotate/Pan/Zoom")
        print(f"\nLoaded {len(self.tracks)} tracks, {len(self.point_cloud.points)} points")
        print(f"Frame range: {self.min_frame} - {self.max_frame}")
        print()
        
        # Main loop
        self.last_update_time = time.time()
        
        while True:
            # Update animation
            if self.is_playing:
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                if dt >= 1.0 / (self.fps * self.playback_speed):
                    self.current_frame += 1
                    if self.current_frame > self.max_frame:
                        self.current_frame = self.min_frame
                    
                    self.update_boxes(vis)
                    self.last_update_time = current_time
            
            # Poll events
            if not vis.poll_events():
                break
            vis.update_renderer()
        
        vis.destroy_window()
    
    def update_boxes(self, vis):
        """Update box positions for current frame."""
        
        for track in self.tracks:
            pos = track.get_position(self.current_frame)
            
            if track.track_id in self.box_geometries:
                # Remove old geometry
                vis.remove_geometry(self.box_geometries[track.track_id], reset_bounding_box=False)
            
            if pos is not None:
                # Create new box at new position
                box = self.create_box_at_position(track, pos)
                self.box_geometries[track.track_id] = box
                vis.add_geometry(box, reset_bounding_box=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Native 4D Scene Viewer")
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory path')
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "outputs"
    
    viewer = Viewer4D(output_dir)
    viewer.run()


if __name__ == "__main__":
    main()
