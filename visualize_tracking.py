#!/usr/bin/env python3
"""
Interactive Tracking and 3D Reconstruction Visualizer

Visualizes tracked objects in 2D video with bounding boxes and their 3D reconstruction.

Left Panel: Video playback with bounding boxes
Right Panel: 3D reconstruction and trajectory visualization

Usage:
    python visualize_tracking.py
"""

import sys
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import queue

# GUI
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# 3D Visualization
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(str(Path(__file__).parent))
from utils import load_config


class TrackingVisualizer:
    """Interactive tracking and 3D reconstruction visualizer"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize visualizer"""
        self.config = load_config(config_path)
        
        # Load data
        self.output_dir = Path(self.config['data']['output_dir']) / "pass2_dynamic"
        self.video_dir = Path(self.config['data']['video_dir'])
        
        self.trajectories = self._load_trajectories()
        self.objects_3d = self._load_3d_objects()
        
        # Current state
        self.current_camera = None
        self.current_frame = 0
        self.video_cap = None
        self.current_tracks = []
        self.selected_track_id = None
        self.playing = False
        self.fps = 30.0
        
        # GUI
        self.root = None
        self.video_canvas = None
        self.track_list = None
        self.fig_3d = None
        self.ax_3d = None
        self.canvas_3d = None
        
    def _load_trajectories(self) -> Dict:
        """Load trajectory data"""
        traj_file = self.output_dir / "trajectories.json"
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
        
        with open(traj_file, 'r') as f:
            return json.load(f)
    
    def _load_3d_objects(self) -> Dict:
        """Load 3D reconstruction data"""
        obj_file = self.output_dir / "objects_3d" / "objects_3d.json"
        if not obj_file.exists():
            raise FileNotFoundError(f"3D objects file not found: {obj_file}")
        
        with open(obj_file, 'r') as f:
            return json.load(f)
    
    def _get_camera_list(self) -> List[str]:
        """Get list of available cameras"""
        return sorted(self.trajectories['cameras'].keys())
    
    def _load_camera_video(self, camera_id: str) -> bool:
        """Load video for camera"""
        video_file = self.video_dir / f"{camera_id}.mp4"
        
        if not video_file.exists():
            messagebox.showerror("Error", f"Video not found: {video_file}")
            return False
        
        if self.video_cap:
            self.video_cap.release()
        
        self.video_cap = cv2.VideoCapture(str(video_file))
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video: {video_file}")
            return False
        
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.current_camera = camera_id
        
        # Load tracks for this camera
        self.current_tracks = self.trajectories['cameras'][camera_id]['tracks']
        
        return True
    
    def _get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get specific frame from video"""
        if not self.video_cap:
            return None
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def _draw_bboxes(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw bounding boxes on frame"""
        frame = frame.copy()
        
        for track in self.current_tracks:
            # Find trajectory point at this frame
            traj_point = None
            for tp in track['trajectory']:
                if tp['frame'] == frame_idx:
                    traj_point = tp
                    break
            
            if not traj_point:
                continue
            
            # Get bbox
            bbox = traj_point['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on class
            color = self._get_class_color(track['class_name'])
            
            # Highlight if selected
            thickness = 3 if track['track_id'] == self.selected_track_id else 2
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{track['class_name']} #{track['track_id']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            center = traj_point['center']
            cx, cy = map(int, center)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Draw velocity arrow if available
            if traj_point['velocity'] is not None:
                vel = np.array(traj_point['velocity'])
                # Scale velocity for visualization
                vel_scaled = vel * 0.1
                end_x = int(cx + vel_scaled[0])
                end_y = int(cy + vel_scaled[1])
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
        
        # Draw frame info
        info = f"Frame: {frame_idx}/{self.total_frames} | Time: {frame_idx/self.fps:.2f}s"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return frame
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for object class"""
        colors = {
            'person': (255, 0, 0),      # Red
            'car': (0, 255, 0),          # Green
            'bus': (0, 0, 255),          # Blue
            'truck': (255, 255, 0),      # Yellow
            'motorcycle': (255, 0, 255), # Magenta
            'bicycle': (0, 255, 255),    # Cyan
        }
        return colors.get(class_name, (128, 128, 128))
    
    def _find_3d_object(self, track_id: int, camera_id: str) -> Optional[Dict]:
        """Find 3D object corresponding to track"""
        # This is a simple lookup - might need improvement for better matching
        for obj in self.objects_3d['objects']:
            # Check if this object has instances from our camera
            for instance in obj['instances']:
                if camera_id in instance['camera_ids']:
                    # Found a match - return the object
                    return obj
        return None
    
    def _visualize_3d(self, track_id: int):
        """Visualize 3D reconstruction and trajectory"""
        if not self.ax_3d:
            return
        
        # Clear previous plot
        self.ax_3d.clear()
        
        # Get the selected track
        track = None
        for t in self.current_tracks:
            if t['track_id'] == track_id:
                track = t
                break
        
        if not track:
            return
        
        # Find corresponding 3D object
        obj_3d = self._find_3d_object(track_id, self.current_camera)
        
        if obj_3d:
            # Plot 3D positions
            positions = np.array([inst['position_3d'] for inst in obj_3d['instances']])
            
            if len(positions) > 0:
                self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                               'o-', linewidth=2, markersize=4, 
                               label=f"{obj_3d['class_name']} #{obj_3d['object_id']}")
                
                # Highlight start and end
                self.ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                                  c='green', s=100, marker='o', label='Start')
                self.ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                                  c='red', s=100, marker='o', label='End')
                
                # Draw bounding boxes at key frames
                for i in [0, len(obj_3d['instances'])//2, -1]:
                    inst = obj_3d['instances'][i]
                    if inst['bbox_3d']:
                        bbox_min = inst['bbox_3d']['min']
                        bbox_max = inst['bbox_3d']['max']
                        self._draw_3d_bbox(bbox_min, bbox_max, alpha=0.2)
        
        # Plot 2D trajectory in image plane (as reference)
        trajectory = np.array([tp['center'] for tp in track['trajectory']])
        if len(trajectory) > 0:
            # Project to XY plane for visualization
            traj_scaled = trajectory * 0.01  # Scale down
            self.ax_3d.plot(traj_scaled[:, 0], traj_scaled[:, 1], 
                           np.zeros(len(traj_scaled)), 
                           'k--', alpha=0.3, linewidth=1, label='2D Trajectory')
        
        # Set labels and title
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title(f"{track['class_name']} #{track_id}\n"
                            f"Duration: {track['duration']:.2f}s | "
                            f"Avg Velocity: {track['avg_velocity']:.1f} px/s")
        self.ax_3d.legend()
        self.ax_3d.grid(True)
        
        # Auto-scale
        if obj_3d and len(positions) > 0:
            margin = 2.0
            self.ax_3d.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
            self.ax_3d.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
            self.ax_3d.set_zlim(positions[:, 2].min() - margin, positions[:, 2].max() + margin)
        
        self.canvas_3d.draw()
    
    def _draw_3d_bbox(self, bbox_min: List[float], bbox_max: List[float], alpha: float = 0.2):
        """Draw 3D bounding box"""
        min_x, min_y, min_z = bbox_min
        max_x, max_y, max_z = bbox_max
        
        # Define 8 corners
        corners = [
            [min_x, min_y, min_z], [max_x, min_y, min_z],
            [max_x, max_y, min_z], [min_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z],
            [max_x, max_y, max_z], [min_x, max_y, max_z]
        ]
        
        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
        ]
        
        # Draw edges
        for edge in edges:
            points = [corners[edge[0]], corners[edge[1]]]
            points = np.array(points)
            self.ax_3d.plot(points[:, 0], points[:, 1], points[:, 2], 
                           'b-', alpha=alpha, linewidth=1)
    
    def create_gui(self):
        """Create GUI window"""
        self.root = tk.Tk()
        self.root.title("Tracking and 3D Reconstruction Visualizer")
        self.root.geometry("1600x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Video and controls
        left_frame = ttk.Frame(main_frame, width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Camera selection
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(control_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_combo = ttk.Combobox(control_frame, values=self._get_camera_list(), 
                                         state='readonly', width=15)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo.bind('<<ComboboxSelected>>', self._on_camera_selected)
        
        # Playback controls
        self.play_button = ttk.Button(control_frame, text="▶ Play", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="⏮ Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⏭ Next", command=self._next_frame).pack(side=tk.LEFT, padx=2)
        
        # Frame slider
        slider_frame = ttk.Frame(left_frame)
        slider_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      command=self._on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # Video canvas
        self.video_canvas = tk.Canvas(left_frame, bg='black', width=800, height=600)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind('<Button-1>', self._on_video_click)
        
        # Track list below video
        list_frame = ttk.LabelFrame(left_frame, text="Tracked Objects", height=150)
        list_frame.pack(fill=tk.BOTH, pady=(5, 0))
        
        # Scrollbar for list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.track_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=6)
        self.track_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.track_list.yview)
        self.track_list.bind('<<ListboxSelect>>', self._on_track_selected)
        
        # Right panel: 3D visualization
        right_frame = ttk.LabelFrame(main_frame, text="3D Reconstruction & Trajectory", width=700)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create 3D plot
        self.fig_3d = Figure(figsize=(7, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=right_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Select a camera to begin", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set default camera
        if self.camera_combo['values']:
            self.camera_combo.current(0)
            self._on_camera_selected(None)
    
    def _on_camera_selected(self, event):
        """Handle camera selection"""
        camera_id = self.camera_combo.get()
        if self._load_camera_video(camera_id):
            self.frame_slider.config(to=self.total_frames - 1)
            self._update_track_list()
            self._update_frame()
            self.status_label.config(text=f"Loaded {camera_id}: {self.total_frames} frames @ {self.fps:.1f} FPS")
    
    def _update_track_list(self):
        """Update track list"""
        self.track_list.delete(0, tk.END)
        
        for track in self.current_tracks:
            label = f"#{track['track_id']:03d} - {track['class_name']:12s} | " \
                   f"{track['duration']:.2f}s | {track['num_detections']} det"
            self.track_list.insert(tk.END, label)
    
    def _on_track_selected(self, event):
        """Handle track selection from list"""
        selection = self.track_list.curselection()
        if selection:
            idx = selection[0]
            track = self.current_tracks[idx]
            self.selected_track_id = track['track_id']
            
            # Jump to first frame of track
            first_frame = track['start_frame']
            self.current_frame = first_frame
            self.frame_slider.set(first_frame)
            
            # Update displays
            self._update_frame()
            self._visualize_3d(track['track_id'])
            
            self.status_label.config(text=f"Selected: {track['class_name']} #{track['track_id']}")
    
    def _on_video_click(self, event):
        """Handle click on video to select object"""
        if not self.video_cap:
            return
        
        # Get click position
        x, y = event.x, event.y
        
        # Find track at this position
        for track in self.current_tracks:
            for tp in track['trajectory']:
                if tp['frame'] == self.current_frame:
                    bbox = tp['bbox']
                    # Scale bbox to canvas size
                    # TODO: Add proper scaling
                    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                        self.selected_track_id = track['track_id']
                        self._visualize_3d(track['track_id'])
                        self.status_label.config(text=f"Selected: {track['class_name']} #{track['track_id']}")
                        self._update_frame()
                        return
    
    def _toggle_play(self):
        """Toggle playback"""
        self.playing = not self.playing
        self.play_button.config(text="⏸ Pause" if self.playing else "▶ Play")
        
        if self.playing:
            self._play_video()
    
    def _play_video(self):
        """Play video"""
        if not self.playing:
            return
        
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set(self.current_frame)
            self._update_frame()
            
            # Schedule next frame
            delay = int(1000 / self.fps)
            self.root.after(delay, self._play_video)
        else:
            self.playing = False
            self.play_button.config(text="▶ Play")
    
    def _prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set(self.current_frame)
            self._update_frame()
    
    def _next_frame(self):
        """Go to next frame"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set(self.current_frame)
            self._update_frame()
    
    def _on_slider_change(self, value):
        """Handle slider change"""
        self.current_frame = int(float(value))
        self._update_frame()
    
    def _update_frame(self):
        """Update video frame display"""
        frame = self._get_frame(self.current_frame)
        
        if frame is not None:
            # Draw bounding boxes
            frame = self._draw_bboxes(frame, self.current_frame)
            
            # Resize to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = frame.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                frame_resized = cv2.resize(frame, (new_w, new_h))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(canvas_width//2, canvas_height//2, 
                                              image=photo, anchor=tk.CENTER)
                self.video_canvas.image = photo  # Keep reference
            
            # Update frame label
            self.frame_label.config(text=f"{self.current_frame}/{self.total_frames}")
    
    def run(self):
        """Run the visualizer"""
        self.create_gui()
        self.root.mainloop()
        
        # Cleanup
        if self.video_cap:
            self.video_cap.release()


def main():
    """Main entry point"""
    try:
        visualizer = TrackingVisualizer()
        visualizer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the tracking pipeline first:")
        print("  1. python pass2_dynamic/track_objects.py")
        print("  2. python pass2_dynamic/reconstruct_objects.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
