#!/usr/bin/env python3
"""
4D Scene Viewer with synchronized video playback.

Layout:
- Left: 3D scene view (Open3D)
- Right: 4x2 grid of camera videos (OpenCV)

Controls:
- SPACE: Play/Pause
- LEFT/RIGHT: Step frame
- R: Reset
- Q: Quit
"""

import sys
from pathlib import Path
import numpy as np
import json
import cv2
import threading
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import open3d as o3d
except ImportError:
    print("Installing Open3D...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "open3d"])
    import open3d as o3d

from utils import load_ply


# Box sizes in meters [length, width, height]
VEHICLE_SIZE = np.array([4.5, 1.8, 1.5])
PERSON_SIZE = np.array([0.5, 0.5, 1.7])
BICYCLE_SIZE = np.array([1.8, 0.6, 1.2])

def get_box_size(class_name: str) -> np.ndarray:
    """Get box size based on object class."""
    if class_name in ['person']:
        return PERSON_SIZE
    elif class_name in ['bicycle', 'motorcycle']:
        return BICYCLE_SIZE
    else:
        return VEHICLE_SIZE

def get_box_color(class_name: str) -> List[float]:
    """Get box color based on object class."""
    if class_name == 'person':
        return [0.3, 1.0, 0.3]  # Green
    elif class_name in ['bicycle', 'motorcycle']:
        return [0.3, 0.8, 1.0]  # Cyan
    else:
        return [1.0, 0.3, 0.3]  # Red


def create_wireframe_box(size: np.ndarray, color: List[float]) -> o3d.geometry.LineSet:
    """Create a wireframe box centered at origin, bottom at z=0."""
    l, w, h = size
    points = [
        [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],  # Bottom
        [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h],  # Top
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


class Track:
    """Tracked object."""
    def __init__(self, track_id: int, class_name: str, frames: Dict[int, np.ndarray]):
        self.track_id = track_id
        self.class_name = class_name
        self.frames = frames  # frame_idx -> [x, y, z, heading]
    
    def get_position(self, frame: int) -> Optional[np.ndarray]:
        return self.frames.get(frame)


class VideoPlayer:
    """Manages synchronized video playback."""
    
    def __init__(self, video_paths: List[Path]):
        self.video_paths = video_paths
        self.caps = []
        self.frames_cache: Dict[int, List[np.ndarray]] = {}
        
        # Open all videos
        for path in video_paths:
            cap = cv2.VideoCapture(str(path))
            self.caps.append(cap)
        
        # Get video info from first video
        if self.caps:
            self.fps = self.caps[0].get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.fps = 15
            self.total_frames = 0
            self.width = 640
            self.height = 480
    
    def get_frames(self, frame_idx: int) -> List[np.ndarray]:
        """Get frames from all videos at given index."""
        if frame_idx in self.frames_cache:
            return self.frames_cache[frame_idx]
        
        frames = []
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize for display
                frame = cv2.resize(frame, (320, 240))
                frames.append(frame)
            else:
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
        
        # Cache a few frames
        if len(self.frames_cache) > 100:
            # Remove oldest
            oldest = min(self.frames_cache.keys())
            del self.frames_cache[oldest]
        self.frames_cache[frame_idx] = frames
        
        return frames
    
    def create_grid(self, frame_idx: int, current_frame: int, is_playing: bool) -> np.ndarray:
        """Create 4x2 grid of video frames."""
        frames = self.get_frames(frame_idx)
        
        # Pad to 8 if needed
        while len(frames) < 8:
            frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
        
        # Create 4 rows x 2 cols grid
        rows = []
        for i in range(4):
            row = np.hstack([frames[i*2], frames[i*2 + 1]])
            rows.append(row)
        
        grid = np.vstack(rows)
        
        # Add info overlay
        status = "PLAYING" if is_playing else "PAUSED"
        cv2.putText(grid, f"Frame: {current_frame}/{self.total_frames} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(grid, "SPACE: Play/Pause | LEFT/RIGHT: Step | R: Reset | Q: Quit",
                    (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add camera labels
        labels = ['s1-left', 's1-right', 's2-left', 's2-right', 
                  's3-left', 's3-right', 's4-left', 's4-right']
        for i, label in enumerate(labels[:len(self.video_paths)]):
            row = i // 2
            col = i % 2
            x = col * 320 + 10
            y = row * 240 + 20
            cv2.putText(grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return grid
    
    def release(self):
        for cap in self.caps:
            cap.release()


class Viewer4DWithVideos:
    """4D viewer with synchronized video playback."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.output_dir = base_dir / "outputs"
        
        # State
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.should_quit = False
        
        # Data
        self.point_cloud = None
        self.tracks: List[Track] = []
        self.box_geometries: Dict[int, o3d.geometry.LineSet] = {}
        
        # Load data
        self.load_point_cloud()
        self.load_tracks()
        
        # Setup video player - prefer annotated videos if available
        annotated_dir = self.output_dir / "pass2_dynamic" / "annotated_videos"
        video_dir = base_dir / "StreetAware-sample"
        
        if annotated_dir.exists() and list(annotated_dir.glob("*_annotated.mp4")):
            print("Using annotated videos (showing tracker detections)")
            video_paths = sorted(annotated_dir.glob("*_annotated.mp4"))
        else:
            print("Using original videos (run generate_annotated_videos.py for tracker view)")
            video_paths = sorted(video_dir.glob("*.mp4"))
        
        self.video_player = VideoPlayer(video_paths)
        
        self.max_frame = self.video_player.total_frames - 1
        self.fps = self.video_player.fps
        
        print(f"Loaded {len(self.tracks)} tracks")
        print(f"Frame range: 0 - {self.max_frame}")
    
    def load_point_cloud(self):
        """Load full resolution point cloud."""
        print("Loading point cloud...")
        ply_path = self.output_dir / "pass1_static" / "pi3_pointcloud_corrected.ply"
        points, colors, _ = load_ply(str(ply_path))
        print(f"  Loaded {len(points)} points")
        
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    def load_tracks(self):
        """Load trajectory data from multi-camera tracker."""
        print("Loading tracks...")
        
        # Use multi-camera trajectories (unified across all cameras)
        multicam_path = self.output_dir / "pass2_dynamic" / "multicam_trajectories.json"
        
        if not multicam_path.exists():
            print("  WARNING: No multicam trajectories found. Run multicam_tracker.py first.")
            return
        
        with open(multicam_path) as f:
            data = json.load(f)
        
        print(f"  Found {data['num_tracks']} unified tracks")
        
        for traj in data['trajectories']:
            # Only include moving vehicles (>5m movement)
            if traj['total_movement_m'] < 5.0:
                continue
            
            track = Track(
                track_id=traj['track_id'],
                class_name=traj['class_name'],
                frames={}
            )
            
            for frame in traj['frames']:
                pos = frame['position_3d']
                heading = frame.get('heading', 0)
                # Check bounds
                if abs(pos[0]) < 30 and abs(pos[1]) < 30:
                    track.frames[frame['frame_idx']] = np.array([pos[0], pos[1], pos[2], heading])
            
            if len(track.frames) >= 15:
                self.tracks.append(track)
        
        print(f"  Loaded {len(self.tracks)} moving vehicle tracks")
    
    def create_box_at_position(self, pos: np.ndarray, class_name: str) -> o3d.geometry.LineSet:
        """Create box at position with heading, sized by class."""
        size = get_box_size(class_name)
        color = get_box_color(class_name)
        box = create_wireframe_box(size, color)
        
        # Apply heading rotation (around Z axis)
        if len(pos) > 3:
            R = box.get_rotation_matrix_from_xyz([0, 0, pos[3]])
            box.rotate(R, center=[0, 0, 0])
        
        # Translate to position
        box.translate([pos[0], pos[1], pos[2]])
        
        return box
    
    def update_boxes(self, vis):
        """Update box positions for current frame."""
        for track in self.tracks:
            pos = track.get_position(self.current_frame)
            
            if track.track_id in self.box_geometries:
                vis.remove_geometry(self.box_geometries[track.track_id], reset_bounding_box=False)
            
            if pos is not None:
                box = self.create_box_at_position(pos, track.class_name)
                self.box_geometries[track.track_id] = box
                vis.add_geometry(box, reset_bounding_box=False)
    
    def run_video_window(self):
        """Run video display in separate thread."""
        cv2.namedWindow("Camera Views", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Views", 640, 960)
        
        last_frame_time = time.time()
        
        while not self.should_quit:
            # Get current frame grid
            grid = self.video_player.create_grid(
                self.current_frame, 
                self.current_frame, 
                self.is_playing
            )
            
            cv2.imshow("Camera Views", grid)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.should_quit = True
            elif key == ord(' '):
                self.is_playing = not self.is_playing
            elif key == 83 or key == ord('d'):  # Right arrow
                self.current_frame = min(self.current_frame + 1, self.max_frame)
            elif key == 81 or key == ord('a'):  # Left arrow
                self.current_frame = max(self.current_frame - 1, 0)
            elif key == ord('r'):
                self.current_frame = 0
            
            # Auto-advance if playing
            if self.is_playing:
                current_time = time.time()
                if current_time - last_frame_time >= 1.0 / (self.fps * self.playback_speed):
                    self.current_frame += 1
                    if self.current_frame > self.max_frame:
                        self.current_frame = 0
                    last_frame_time = current_time
            
            time.sleep(0.01)
        
        cv2.destroyAllWindows()
        self.video_player.release()
    
    def run(self):
        """Run the viewer."""
        
        # Start video window in separate thread
        video_thread = threading.Thread(target=self.run_video_window)
        video_thread.start()
        
        # Create 3D visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("4D Scene", width=800, height=900)
        
        # Add point cloud
        vis.add_geometry(self.point_cloud)
        
        # Set point size (small for dense point cloud)
        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.1, 0.1, 0.15])
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        vis.add_geometry(coord_frame)
        
        # Add ground grid
        grid_lines = []
        for i in range(-20, 21, 2):
            grid_lines.append([[i, -20, 0], [i, 20, 0]])
            grid_lines.append([[-20, i, 0], [20, i, 0]])
        
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
        
        # Initial boxes
        for track in self.tracks:
            pos = track.get_position(self.current_frame)
            if pos is not None:
                box = self.create_box_at_position(pos, track.class_name)
                self.box_geometries[track.track_id] = box
                vis.add_geometry(box)
        
        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.4)
        ctr.set_front([0.5, -0.5, 0.7])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # Key callbacks
        def toggle_play(vis):
            self.is_playing = not self.is_playing
            return False
        
        def next_frame(vis):
            self.current_frame = min(self.current_frame + 1, self.max_frame)
            self.update_boxes(vis)
            return False
        
        def prev_frame(vis):
            self.current_frame = max(self.current_frame - 1, 0)
            self.update_boxes(vis)
            return False
        
        def reset_frame(vis):
            self.current_frame = 0
            self.update_boxes(vis)
            return False
        
        def quit_viewer(vis):
            self.should_quit = True
            return False
        
        vis.register_key_callback(ord(" "), toggle_play)
        vis.register_key_callback(262, next_frame)  # Right
        vis.register_key_callback(263, prev_frame)  # Left
        vis.register_key_callback(ord("R"), reset_frame)
        vis.register_key_callback(ord("Q"), quit_viewer)
        
        print("\n=== 4D Scene Viewer with Videos ===")
        print("Controls (both windows):")
        print("  SPACE: Play/Pause")
        print("  LEFT/RIGHT or A/D: Step frame")
        print("  R: Reset to frame 0")
        print("  Q: Quit")
        print()
        
        # Main loop
        last_update = time.time()
        last_frame = -1
        
        while not self.should_quit:
            # Update boxes if frame changed
            if self.current_frame != last_frame:
                self.update_boxes(vis)
                last_frame = self.current_frame
            
            if not vis.poll_events():
                self.should_quit = True
                break
            vis.update_renderer()
            
            time.sleep(0.01)
        
        vis.destroy_window()
        video_thread.join()


def main():
    base_dir = Path(__file__).parent.parent
    viewer = Viewer4DWithVideos(base_dir)
    viewer.run()


if __name__ == "__main__":
    main()
