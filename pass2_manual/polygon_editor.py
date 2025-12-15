#!/usr/bin/env python3
"""
Polygon Annotation Editor for Scene Layout

Draw polygons to annotate:
- Roads (bidirectional, NYC style)
- Crosswalks
- Sidewalks
- Parking areas

Uses matplotlib for interactive polygon drawing on top of the static scene.
"""

import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, RadioButtons
import open3d as o3d
from PIL import Image


class PolygonEditor:
    """Interactive polygon editor using matplotlib."""
    
    def __init__(self, static_scene_path: Path, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Load static scene and create top-view image
        print("Loading static scene...")
        self.pcd = o3d.io.read_point_cloud(str(static_scene_path))
        self.points = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors)
        
        # Scene bounds
        self.bounds_min = self.points.min(axis=0)
        self.bounds_max = self.points.max(axis=0)
        
        # Extend bounds for drawing outside scene (add 50% margin)
        margin = 20  # meters
        self.view_min = np.array([self.bounds_min[0] - margin, self.bounds_min[1] - margin])
        self.view_max = np.array([self.bounds_max[0] + margin, self.bounds_max[1] + margin])
        
        print(f"  Scene bounds: X=[{self.bounds_min[0]:.1f}, {self.bounds_max[0]:.1f}], "
              f"Y=[{self.bounds_min[1]:.1f}, {self.bounds_max[1]:.1f}]")
        print(f"  View bounds:  X=[{self.view_min[0]:.1f}, {self.view_max[0]:.1f}], "
              f"Y=[{self.view_min[1]:.1f}, {self.view_max[1]:.1f}]")
        
        # Create top-view rasterized image
        self._create_top_view_image()
        
        # Annotation data
        self.annotations = {
            'roads': [],        # Bidirectional roads
            'crosswalks': [],   # Pedestrian crossings
            'sidewalks': [],    # Walking areas
            'parking': []       # Parking zones
        }
        
        # Current drawing state
        self.current_category = 'roads'
        self.current_polygon = []
        self.temp_line = None
        
        # Colors for each category
        self.category_colors = {
            'roads': (0.3, 0.3, 0.3, 0.5),      # Dark gray
            'crosswalks': (1.0, 1.0, 0.0, 0.6), # Yellow
            'sidewalks': (0.6, 0.6, 0.6, 0.5),  # Light gray
            'parking': (0.0, 0.5, 1.0, 0.5)     # Blue
        }
        
        # Figure and axes
        self.fig = None
        self.ax = None
        self.polygon_patches = {}
    
    def _create_top_view_image(self):
        """Create a rasterized top-view image of the point cloud."""
        print("Creating top-view image...")
        
        # Resolution: pixels per meter
        self.resolution = 50  # 50 pixels per meter
        
        width = int((self.view_max[0] - self.view_min[0]) * self.resolution)
        height = int((self.view_max[1] - self.view_min[1]) * self.resolution)
        
        print(f"  Image size: {width} x {height} pixels")
        
        # Create grey background (road-like color)
        self.top_view = np.ones((height, width, 3), dtype=np.float32) * 0.35
        
        # Project points to image
        x_idx = ((self.points[:, 0] - self.view_min[0]) * self.resolution).astype(int)
        y_idx = ((self.view_max[1] - self.points[:, 1]) * self.resolution).astype(int)  # Flip Y
        
        # Filter valid indices
        valid = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        colors_valid = self.colors[valid]
        
        # Draw points
        self.top_view[y_idx, x_idx] = colors_valid
        
        print("  Done")
    
    def _world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates."""
        px = (x - self.view_min[0]) * self.resolution
        py = (self.view_max[1] - y) * self.resolution
        return px, py
    
    def _pixel_to_world(self, px, py):
        """Convert pixel coordinates to world coordinates."""
        x = px / self.resolution + self.view_min[0]
        y = self.view_max[1] - py / self.resolution
        return x, y
    
    def _setup_ui(self):
        """Setup the matplotlib UI."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 14))
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.98, top=0.95)
        
        # Show top view image
        self.ax.imshow(self.top_view, extent=[self.view_min[0], self.view_max[0], 
                                               self.view_min[1], self.view_max[1]],
                       origin='lower')
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Polygon Editor - Click to draw, Right-click to finish polygon')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Add scene boundary rectangle
        from matplotlib.patches import Rectangle
        scene_rect = Rectangle((self.bounds_min[0], self.bounds_min[1]),
                               self.bounds_max[0] - self.bounds_min[0],
                               self.bounds_max[1] - self.bounds_min[1],
                               fill=False, edgecolor='white', linestyle='--', linewidth=1)
        self.ax.add_patch(scene_rect)
        
        # Category selection radio buttons
        ax_radio = plt.axes([0.02, 0.7, 0.1, 0.15])
        self.radio = RadioButtons(ax_radio, ('roads', 'crosswalks', 'sidewalks', 'parking'))
        self.radio.on_clicked(self._on_category_change)
        
        # Buttons
        ax_undo = plt.axes([0.02, 0.6, 0.1, 0.04])
        self.btn_undo = Button(ax_undo, 'Undo Point')
        self.btn_undo.on_clicked(self._on_undo)
        
        ax_clear = plt.axes([0.02, 0.55, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear Current')
        self.btn_clear.on_clicked(self._on_clear_current)
        
        ax_delete = plt.axes([0.02, 0.5, 0.1, 0.04])
        self.btn_delete = Button(ax_delete, 'Delete Last')
        self.btn_delete.on_clicked(self._on_delete_last)
        
        ax_save = plt.axes([0.02, 0.4, 0.1, 0.04])
        self.btn_save = Button(ax_save, 'SAVE')
        self.btn_save.on_clicked(self._on_save)
        
        ax_load = plt.axes([0.02, 0.35, 0.1, 0.04])
        self.btn_load = Button(ax_load, 'Load')
        self.btn_load.on_clicked(self._on_load)
        
        # Status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=10, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                        color='white')
        self._update_status()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        
        # Initialize polygon patch collections
        for cat in self.annotations:
            self.polygon_patches[cat] = []
    
    def _update_status(self):
        """Update status text."""
        counts = {cat: len(polys) for cat, polys in self.annotations.items()}
        status = f"Category: {self.current_category.upper()}\n"
        status += f"Current points: {len(self.current_polygon)}\n"
        status += f"Roads: {counts['roads']} | Crosswalks: {counts['crosswalks']}\n"
        status += f"Sidewalks: {counts['sidewalks']} | Parking: {counts['parking']}"
        self.status_text.set_text(status)
        self.fig.canvas.draw_idle()
    
    def _on_category_change(self, label):
        """Handle category change."""
        # Finish current polygon if any
        if len(self.current_polygon) >= 3:
            self._finish_polygon()
        self.current_polygon = []
        self.current_category = label
        self._update_status()
    
    def _on_click(self, event):
        """Handle mouse click."""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click - add point
            x, y = event.xdata, event.ydata
            self.current_polygon.append([x, y])
            
            # Draw point
            self.ax.plot(x, y, 'o', color=self.category_colors[self.current_category][:3], 
                        markersize=8)
            
            # Draw line to previous point
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon[-2:])
                self.ax.plot(pts[:, 0], pts[:, 1], '-', 
                            color=self.category_colors[self.current_category][:3], linewidth=2)
            
            self._update_status()
            self.fig.canvas.draw_idle()
        
        elif event.button == 3:  # Right click - finish polygon
            if len(self.current_polygon) >= 3:
                self._finish_polygon()
    
    def _on_motion(self, event):
        """Handle mouse motion for preview line."""
        pass  # Could add preview line here
    
    def _finish_polygon(self):
        """Finish current polygon and add to annotations."""
        if len(self.current_polygon) < 3:
            return
        
        polygon = self.current_polygon.copy()
        self.annotations[self.current_category].append(polygon)
        
        # Draw filled polygon
        poly_patch = MplPolygon(polygon, closed=True, 
                                facecolor=self.category_colors[self.current_category],
                                edgecolor=self.category_colors[self.current_category][:3],
                                linewidth=2)
        self.ax.add_patch(poly_patch)
        self.polygon_patches[self.current_category].append(poly_patch)
        
        print(f"Added {self.current_category} polygon with {len(polygon)} points")
        
        self.current_polygon = []
        self._update_status()
        self.fig.canvas.draw_idle()
    
    def _on_undo(self, event):
        """Undo last point."""
        if self.current_polygon:
            self.current_polygon.pop()
            self._redraw()
    
    def _on_clear_current(self, event):
        """Clear current polygon."""
        self.current_polygon = []
        self._redraw()
    
    def _on_delete_last(self, event):
        """Delete last polygon of current category."""
        if self.annotations[self.current_category]:
            self.annotations[self.current_category].pop()
            if self.polygon_patches[self.current_category]:
                patch = self.polygon_patches[self.current_category].pop()
                patch.remove()
            self._redraw()
    
    def _redraw(self):
        """Redraw all annotations."""
        # Clear and redraw
        self.ax.clear()
        
        # Show top view
        self.ax.imshow(self.top_view, extent=[self.view_min[0], self.view_max[0],
                                               self.view_min[1], self.view_max[1]],
                       origin='lower')
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Polygon Editor - Click to draw, Right-click to finish polygon')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Scene boundary
        from matplotlib.patches import Rectangle
        scene_rect = Rectangle((self.bounds_min[0], self.bounds_min[1]),
                               self.bounds_max[0] - self.bounds_min[0],
                               self.bounds_max[1] - self.bounds_min[1],
                               fill=False, edgecolor='white', linestyle='--', linewidth=1)
        self.ax.add_patch(scene_rect)
        
        # Redraw all polygons
        self.polygon_patches = {cat: [] for cat in self.annotations}
        for cat, polygons in self.annotations.items():
            for polygon in polygons:
                poly_patch = MplPolygon(polygon, closed=True,
                                        facecolor=self.category_colors[cat],
                                        edgecolor=self.category_colors[cat][:3],
                                        linewidth=2)
                self.ax.add_patch(poly_patch)
                self.polygon_patches[cat].append(poly_patch)
        
        # Draw current polygon points
        if self.current_polygon:
            pts = np.array(self.current_polygon)
            self.ax.plot(pts[:, 0], pts[:, 1], 'o-',
                        color=self.category_colors[self.current_category][:3],
                        markersize=8, linewidth=2)
        
        # Re-add status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        fontsize=10, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                        color='white')
        self._update_status()
        self.fig.canvas.draw_idle()
    
    def _on_save(self, event):
        """Save annotations to JSON."""
        save_path = self.output_dir / "scene_annotations.json"
        
        data = {
            'annotations': self.annotations,
            'bounds': {
                'scene_min': self.bounds_min.tolist(),
                'scene_max': self.bounds_max.tolist(),
                'view_min': self.view_min.tolist(),
                'view_max': self.view_max.tolist()
            },
            'notes': {
                'roads': 'Bidirectional, NYC style (right-hand traffic)',
                'crosswalks': 'Pedestrian crossings, traffic stops when pedestrians cross',
                'sidewalks': 'Pedestrian walking areas',
                'parking': 'Parked vehicle areas'
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSaved annotations to {save_path}")
        print(f"  Roads: {len(self.annotations['roads'])}")
        print(f"  Crosswalks: {len(self.annotations['crosswalks'])}")
        print(f"  Sidewalks: {len(self.annotations['sidewalks'])}")
        print(f"  Parking: {len(self.annotations['parking'])}")
    
    def _on_load(self, event):
        """Load annotations from JSON."""
        load_path = self.output_dir / "scene_annotations.json"
        
        if not load_path.exists():
            print(f"No annotations file found at {load_path}")
            return
        
        with open(load_path) as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        print(f"Loaded annotations from {load_path}")
        self._redraw()
    
    def run(self):
        """Run the editor."""
        print("\n" + "="*60)
        print("POLYGON ANNOTATION EDITOR")
        print("="*60)
        print("\nInstructions:")
        print("  1. Select category (roads/crosswalks/sidewalks/parking)")
        print("  2. LEFT CLICK to add polygon points")
        print("  3. RIGHT CLICK to finish polygon")
        print("  4. Use buttons to undo/clear/delete")
        print("  5. Click SAVE when done")
        print("\nWhite dashed rectangle = scene boundary")
        print("You can draw OUTSIDE the scene boundary")
        print("="*60)
        
        self._setup_ui()
        
        # Try to load existing annotations
        load_path = self.output_dir / "scene_annotations.json"
        if load_path.exists():
            print(f"\nFound existing annotations, loading...")
            self._on_load(None)
        
        plt.show()
        
        print("\nEditor closed.")


def main():
    base_dir = Path(__file__).parent.parent
    static_scene = base_dir / "outputs" / "pass1_static" / "pi3_pointcloud_corrected.ply"
    output_dir = Path(__file__).parent
    
    if not static_scene.exists():
        print(f"Error: {static_scene} not found")
        return
    
    editor = PolygonEditor(static_scene, output_dir)
    editor.run()


if __name__ == "__main__":
    main()
