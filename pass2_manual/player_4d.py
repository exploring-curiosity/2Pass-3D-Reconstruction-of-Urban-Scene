#!/usr/bin/env python3
"""
Interactive 4D Scene Player

- Play/pause animation
- Scrub through time
- Export any frame to PLY
- Solid mesh objects
"""

import sys
from pathlib import Path
import numpy as np
import json
import open3d as o3d
import time
import threading

from scene_generator import SceneGenerator, create_solid_mesh
from scene_objects import ObjectType


class Player4D:
    """Interactive 4D scene player."""
    
    def __init__(self, static_scene_path: Path, config_path: Path):
        print("Loading scene...")
        self.generator = SceneGenerator.load(static_scene_path, config_path)
        self.static_pcd = self.generator.static_pcd
        self.duration = self.generator.duration
        self.ground_z = self.generator.ground_z
        
        print(f"  Duration: {self.duration:.1f}s")
        print(f"  Objects: {len(self.generator.objects)}")
        
        # State
        self.current_time = 0.0
        self.playing = False
        self.speed = 1.0
        self.fps = 10.0
        
        # Output directory
        self.output_dir = Path(__file__).parent / "exports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualizer
        self.vis = None
        self.dynamic_meshes = []
    
    def _print_controls(self):
        """Print keyboard controls."""
        print("\n" + "="*60)
        print("4D SCENE PLAYER")
        print("="*60)
        print("\nKEYBOARD CONTROLS:")
        print("  SPACE     - Play/Pause")
        print("  LEFT/RIGHT- Step backward/forward 1 second")
        print("  UP/DOWN   - Speed up/slow down")
        print("  HOME      - Go to start")
        print("  END       - Go to end")
        print("  E         - Export current frame to PLY")
        print("  R         - Reset view")
        print("  Q         - Quit")
        print("="*60)
    
    def _update_scene(self):
        """Update scene for current time."""
        # Remove old dynamic meshes
        for mesh in self.dynamic_meshes:
            self.vis.remove_geometry(mesh, reset_bounding_box=False)
        self.dynamic_meshes.clear()
        
        # Add new meshes
        meshes = self.generator.get_meshes_at_time(self.current_time)
        for mesh in meshes:
            self.vis.add_geometry(mesh, reset_bounding_box=False)
            self.dynamic_meshes.append(mesh)
        
        self.vis.update_renderer()
    
    def _export_frame(self):
        """Export current frame to PLY."""
        # Combine static scene with dynamic objects as point cloud
        all_points = [self.generator.static_points.copy()]
        all_colors = [self.generator.static_colors.copy()]
        
        # Sample points from meshes
        meshes = self.generator.get_meshes_at_time(self.current_time)
        for mesh in meshes:
            # Sample points from mesh surface
            pcd = mesh.sample_points_uniformly(number_of_points=2000)
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            all_points.append(pts)
            all_colors.append(cols)
        
        # Create combined point cloud
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        # Save
        filename = f"frame_t{self.current_time:.2f}s.ply"
        filepath = self.output_dir / filename
        o3d.io.write_point_cloud(str(filepath), combined)
        
        # Also save state
        state = {
            'time': self.current_time,
            'num_points': len(combined.points),
            'objects': []
        }
        for obj in self.generator.objects:
            pos = obj.get_position_at_time(self.current_time)
            if pos:
                state['objects'].append({
                    'id': obj.id,
                    'type': obj.obj_type.value,
                    'position': [pos[0], pos[1], self.ground_z],
                    'heading': obj.get_heading_at_time(self.current_time)
                })
        
        json_path = self.output_dir / f"frame_t{self.current_time:.2f}s.json"
        with open(json_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"\nExported: {filepath.name} ({len(combined.points):,} points)")
    
    def run(self):
        """Run the player."""
        self._print_controls()
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="4D Scene Player", width=1600, height=1000)
        
        # Add static scene
        self.vis.add_geometry(self.static_pcd)
        
        # Initial dynamic objects
        self._update_scene()
        
        # Render options
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.15, 0.15, 0.15])
        opt.mesh_show_back_face = True
        
        # Set initial view
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.25)
        
        # Key callbacks
        def toggle_play(vis):
            self.playing = not self.playing
            print(f"{'Playing' if self.playing else 'Paused'} at t={self.current_time:.1f}s")
            return False
        
        def step_forward(vis):
            self.current_time = min(self.duration, self.current_time + 1.0)
            self._update_scene()
            print(f"t={self.current_time:.1f}s")
            return False
        
        def step_backward(vis):
            self.current_time = max(0, self.current_time - 1.0)
            self._update_scene()
            print(f"t={self.current_time:.1f}s")
            return False
        
        def speed_up(vis):
            self.speed = min(4.0, self.speed * 1.5)
            print(f"Speed: {self.speed:.1f}x")
            return False
        
        def slow_down(vis):
            self.speed = max(0.25, self.speed / 1.5)
            print(f"Speed: {self.speed:.1f}x")
            return False
        
        def go_start(vis):
            self.current_time = 0
            self._update_scene()
            print(f"t={self.current_time:.1f}s")
            return False
        
        def go_end(vis):
            self.current_time = self.duration
            self._update_scene()
            print(f"t={self.current_time:.1f}s")
            return False
        
        def export_frame(vis):
            self._export_frame()
            return False
        
        def reset_view(vis):
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_lookat([0, 0, 0])
            ctr.set_zoom(0.25)
            return False
        
        def quit_player(vis):
            vis.close()
            return False
        
        # Register callbacks
        self.vis.register_key_callback(ord(' '), toggle_play)
        self.vis.register_key_callback(262, step_forward)   # RIGHT arrow
        self.vis.register_key_callback(263, step_backward)  # LEFT arrow
        self.vis.register_key_callback(265, speed_up)       # UP arrow
        self.vis.register_key_callback(264, slow_down)      # DOWN arrow
        self.vis.register_key_callback(268, go_start)       # HOME
        self.vis.register_key_callback(269, go_end)         # END
        self.vis.register_key_callback(ord('E'), export_frame)
        self.vis.register_key_callback(ord('R'), reset_view)
        self.vis.register_key_callback(ord('Q'), quit_player)
        
        # Animation loop
        last_update = time.time()
        frame_duration = 1.0 / self.fps
        
        print(f"\nStarting at t=0.0s (press SPACE to play)")
        
        while True:
            if not self.vis.poll_events():
                break
            
            if self.playing:
                now = time.time()
                if now - last_update >= frame_duration:
                    dt = (now - last_update) * self.speed
                    self.current_time += dt
                    
                    if self.current_time >= self.duration:
                        self.current_time = 0  # Loop
                    
                    self._update_scene()
                    last_update = now
            
            self.vis.update_renderer()
            time.sleep(0.01)
        
        self.vis.destroy_window()
        print("\nPlayer closed.")


def main():
    base_dir = Path(__file__).parent.parent
    static_scene = base_dir / "outputs" / "pass1_static" / "pi3_pointcloud_corrected.ply"
    config_path = Path(__file__).parent / "generated_scene.json"
    
    if not config_path.exists():
        print(f"No scene config at {config_path}")
        print("Run scene_generator.py first")
        return
    
    player = Player4D(static_scene, config_path)
    player.run()


if __name__ == "__main__":
    main()
