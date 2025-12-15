#!/usr/bin/env python3
"""
Interactive Object Placer with Live Preview

- See canonical mesh preview at cursor position
- Click to place, see it immediately
- Interactive controls for type, rotation, delete
- Place anywhere (not limited to point cloud)
- Export to PLY
"""

import sys
from pathlib import Path
import numpy as np
import json
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from typing import List, Optional
from dataclasses import dataclass

# Object dimensions (length, width, height) in meters
OBJECT_DIMS = {
    'person': (0.5, 0.5, 1.7),
    'bicycle': (1.8, 0.6, 1.1),
    'car': (4.5, 1.8, 1.5),
    'truck': (7.0, 2.5, 3.0),
    'bus': (12.0, 2.5, 3.2),
}

OBJECT_COLORS = {
    'person': (0.2, 0.6, 0.9),
    'bicycle': (0.8, 0.2, 0.8),
    'car': (0.9, 0.2, 0.2),
    'truck': (0.2, 0.7, 0.2),
    'bus': (0.9, 0.6, 0.1),
}

OBJECT_TYPES = ['person', 'bicycle', 'car', 'truck', 'bus']


@dataclass
class PlacedObject:
    id: int
    obj_type: str
    x: float
    y: float
    z: float
    heading: float
    
    def to_dict(self):
        return {'id': self.id, 'type': self.obj_type, 
                'x': self.x, 'y': self.y, 'z': self.z, 'heading': self.heading}
    
    @classmethod
    def from_dict(cls, d):
        return cls(d['id'], d['type'], d['x'], d['y'], d['z'], d['heading'])


def create_mesh(obj_type: str, x: float, y: float, z: float, heading: float) -> o3d.geometry.TriangleMesh:
    """Create canonical mesh for object."""
    dims = OBJECT_DIMS[obj_type]
    color = OBJECT_COLORS[obj_type]
    length, width, height = dims
    
    if obj_type == 'person':
        body = o3d.geometry.TriangleMesh.create_cylinder(radius=width/2, height=height*0.7)
        body.translate([0, 0, height * 0.35])
        head = o3d.geometry.TriangleMesh.create_sphere(radius=width/2 * 0.8)
        head.translate([0, 0, height * 0.85])
        mesh = body + head
    else:
        mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
        mesh.translate([-length/2, -width/2, 0])
    
    mesh.paint_uniform_color(color)
    R = mesh.get_rotation_matrix_from_xyz([0, 0, heading])
    mesh.rotate(R, center=[0, 0, 0])
    mesh.translate([x, y, z])
    mesh.compute_vertex_normals()
    return mesh


class ObjectPlacerApp:
    def __init__(self, static_scene_path: Path):
        # Load scene
        print("Loading static scene...")
        self.static_pcd = o3d.io.read_point_cloud(str(static_scene_path))
        self.points = np.asarray(self.static_pcd.points)
        self.ground_z = np.percentile(self.points[:, 2], 5)
        
        bounds_min = self.points.min(axis=0)
        bounds_max = self.points.max(axis=0)
        self.scene_center = (bounds_min + bounds_max) / 2
        self.scene_size = (bounds_max - bounds_min).max()
        
        print(f"  Points: {len(self.points):,}")
        print(f"  Ground Z: {self.ground_z:.2f}")
        
        # State
        self.objects: List[PlacedObject] = []
        self.next_id = 0
        self.current_type_idx = 2  # car
        self.current_heading = 0.0
        self.preview_mesh = None
        self.output_dir = Path(__file__).parent
        
        # Load existing
        self._load_if_exists()
        
        # Create app
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window("Object Placer", 1600, 1000)
        
        # 3D widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.15, 0.15, 0.15, 1.0])
        
        # Material for point cloud
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = 2.0
        
        # Material for meshes
        self.mesh_mat = rendering.MaterialRecord()
        self.mesh_mat.shader = "defaultLit"
        
        # Preview material (semi-transparent)
        self.preview_mat = rendering.MaterialRecord()
        self.preview_mat.shader = "defaultLitTransparency"
        self.preview_mat.base_color = [0.5, 0.5, 0.5, 0.5]
        
        # Add static scene
        self.scene_widget.scene.add_geometry("static_scene", self.static_pcd, self.pcd_mat)
        
        # Add ground plane for clicking outside point cloud
        ground = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=0.01)
        ground.translate([-50, -50, self.ground_z - 0.01])
        ground.paint_uniform_color([0.3, 0.3, 0.3])
        ground.compute_vertex_normals()
        ground_mat = rendering.MaterialRecord()
        ground_mat.shader = "defaultLit"
        ground_mat.base_color = [0.25, 0.25, 0.25, 1.0]
        self.scene_widget.scene.add_geometry("ground", ground, ground_mat)
        
        # Setup camera
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
        
        # Control panel
        self._setup_panel()
        
        # Layout
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        
        # Mouse handling
        self.scene_widget.set_on_mouse(self._on_mouse)
        
        # Add existing objects
        self._refresh_objects()
        
        print("\nEditor ready. Click on scene to place objects.")
    
    def _setup_panel(self):
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        self.panel.background_color = gui.Color(0.1, 0.1, 0.1, 0.9)
        
        # Title
        title = gui.Label("OBJECT PLACER")
        title.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(title)
        
        self.panel.add_child(gui.Label(""))
        
        # Object type selector
        self.panel.add_child(gui.Label("Object Type:"))
        self.type_combo = gui.Combobox()
        for t in OBJECT_TYPES:
            self.type_combo.add_item(t.upper())
        self.type_combo.selected_index = self.current_type_idx
        self.type_combo.set_on_selection_changed(self._on_type_changed)
        self.panel.add_child(self.type_combo)
        
        self.panel.add_child(gui.Label(""))
        
        # Heading slider
        self.panel.add_child(gui.Label("Heading (degrees):"))
        self.heading_slider = gui.Slider(gui.Slider.DOUBLE)
        self.heading_slider.set_limits(-180, 180)
        self.heading_slider.double_value = 0
        self.heading_slider.set_on_value_changed(self._on_heading_changed)
        self.panel.add_child(self.heading_slider)
        
        self.heading_label = gui.Label("0°")
        self.panel.add_child(self.heading_label)
        
        self.panel.add_child(gui.Label(""))
        
        # Buttons
        undo_btn = gui.Button("Undo Last (U)")
        undo_btn.set_on_clicked(self._on_undo)
        self.panel.add_child(undo_btn)
        
        clear_btn = gui.Button("Clear All")
        clear_btn.set_on_clicked(self._on_clear)
        self.panel.add_child(clear_btn)
        
        self.panel.add_child(gui.Label(""))
        
        save_btn = gui.Button("Save (S)")
        save_btn.set_on_clicked(self._on_save)
        self.panel.add_child(save_btn)
        
        export_btn = gui.Button("Export PLY (E)")
        export_btn.set_on_clicked(self._on_export)
        self.panel.add_child(export_btn)
        
        self.panel.add_child(gui.Label(""))
        
        # Status
        self.status_label = gui.Label(f"Objects: {len(self.objects)}")
        self.panel.add_child(self.status_label)
        
        # Instructions
        self.panel.add_child(gui.Label(""))
        self.panel.add_child(gui.Label("CLICK to place object"))
        self.panel.add_child(gui.Label("Scroll to zoom"))
        self.panel.add_child(gui.Label("Right-drag to rotate"))
    
    def _on_layout(self, ctx):
        r = self.window.content_rect
        panel_width = 200
        self.scene_widget.frame = gui.Rect(0, 0, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.width - panel_width, 0, panel_width, r.height)
    
    def _on_type_changed(self, name, idx):
        self.current_type_idx = idx
        print(f"Selected: {OBJECT_TYPES[idx]}")
    
    def _on_heading_changed(self, value):
        self.current_heading = np.radians(value)
        self.heading_label.text = f"{value:.0f}°"
    
    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            # Raycast to find click position
            def depth_callback(depth_image):
                x = event.x - self.scene_widget.frame.x
                y = event.y - self.scene_widget.frame.y
                
                if x < 0 or y < 0:
                    return
                
                depth = np.asarray(depth_image)[int(y), int(x)]
                
                if depth == 1.0:  # No hit
                    return
                
                # Unproject to world coordinates
                world_pt = self.scene_widget.scene.camera.unproject(
                    x, y, depth,
                    self.scene_widget.frame.width,
                    self.scene_widget.frame.height
                )
                
                self._place_object(world_pt[0], world_pt[1])
            
            self.scene_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _place_object(self, x: float, y: float):
        """Place object at position."""
        obj_type = OBJECT_TYPES[self.current_type_idx]
        
        obj = PlacedObject(
            id=self.next_id,
            obj_type=obj_type,
            x=x, y=y, z=self.ground_z,
            heading=self.current_heading
        )
        self.objects.append(obj)
        self.next_id += 1
        
        # Add mesh to scene
        mesh = create_mesh(obj_type, x, y, self.ground_z, self.current_heading)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        color = OBJECT_COLORS[obj_type]
        mat.base_color = [color[0], color[1], color[2], 1.0]
        
        self.scene_widget.scene.add_geometry(f"obj_{obj.id}", mesh, mat)
        
        self._update_status()
        print(f"Placed {obj_type} #{obj.id} at ({x:.1f}, {y:.1f})")
    
    def _refresh_objects(self):
        """Refresh all object meshes in scene."""
        # Remove existing
        for obj in self.objects:
            self.scene_widget.scene.remove_geometry(f"obj_{obj.id}")
        
        # Re-add
        for obj in self.objects:
            mesh = create_mesh(obj.obj_type, obj.x, obj.y, obj.z, obj.heading)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            color = OBJECT_COLORS[obj.obj_type]
            mat.base_color = [color[0], color[1], color[2], 1.0]
            self.scene_widget.scene.add_geometry(f"obj_{obj.id}", mesh, mat)
        
        self._update_status()
    
    def _update_status(self):
        self.status_label.text = f"Objects: {len(self.objects)}"
    
    def _on_undo(self):
        if self.objects:
            obj = self.objects.pop()
            self.scene_widget.scene.remove_geometry(f"obj_{obj.id}")
            self._update_status()
            print(f"Removed {obj.obj_type} #{obj.id}")
    
    def _on_clear(self):
        for obj in self.objects:
            self.scene_widget.scene.remove_geometry(f"obj_{obj.id}")
        self.objects.clear()
        self._update_status()
        print("Cleared all objects")
    
    def _on_save(self):
        path = self.output_dir / "placements.json"
        data = {
            'ground_z': self.ground_z,
            'objects': [obj.to_dict() for obj in self.objects]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.objects)} objects to {path}")
    
    def _load_if_exists(self):
        path = self.output_dir / "placements.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.objects = [PlacedObject.from_dict(d) for d in data['objects']]
            self.next_id = max((o.id for o in self.objects), default=-1) + 1
            print(f"Loaded {len(self.objects)} existing objects")
    
    def _on_export(self):
        # Combine static + objects
        all_points = [self.points.copy()]
        all_colors = [np.asarray(self.static_pcd.colors).copy()]
        
        for obj in self.objects:
            mesh = create_mesh(obj.obj_type, obj.x, obj.y, obj.z, obj.heading)
            pcd = mesh.sample_points_uniformly(number_of_points=3000)
            all_points.append(np.asarray(pcd.points))
            all_colors.append(np.asarray(pcd.colors))
        
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        path = self.output_dir / "scene_with_objects.ply"
        o3d.io.write_point_cloud(str(path), combined)
        print(f"Exported {len(combined.points):,} points to {path}")
    
    def run(self):
        self.app.run()


def main():
    base_dir = Path(__file__).parent.parent
    static_scene = base_dir / "outputs" / "pass1_static" / "pi3_pointcloud_corrected.ply"
    
    if not static_scene.exists():
        print(f"Error: {static_scene} not found")
        return
    
    app = ObjectPlacerApp(static_scene)
    app.run()


if __name__ == "__main__":
    main()
