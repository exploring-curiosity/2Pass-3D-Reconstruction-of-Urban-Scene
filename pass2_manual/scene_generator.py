#!/usr/bin/env python3
"""
Generate 4D Scene from Polygon Annotations

Fixed version:
- Gray parked vehicles (only in parking areas)
- Colored moving vehicles by type (only on roads)
- Single color pedestrians (sidewalks + crosswalks perpendicular crossing)
- Proper collision avoidance
- Correct orientations
"""

import sys
from pathlib import Path
import numpy as np
import json
import open3d as o3d
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from scene_objects import ObjectType, OBJECT_DIMENSIONS


# Color scheme
PARKED_COLOR = (0.4, 0.4, 0.4)  # Gray for all parked vehicles
PEDESTRIAN_COLOR = (0.2, 0.6, 0.9)  # Blue for all pedestrians

# Moving vehicle colors by type
MOVING_COLORS = {
    ObjectType.CAR: (0.9, 0.2, 0.2),      # Red
    ObjectType.TRUCK: (0.2, 0.7, 0.2),    # Green  
    ObjectType.BUS: (0.9, 0.6, 0.1),      # Orange
    ObjectType.BICYCLE: (0.8, 0.2, 0.8),  # Purple
}


@dataclass
class DynamicObject:
    """A dynamic object in the scene."""
    id: int
    obj_type: ObjectType
    color: Tuple[float, float, float]
    trajectory: List[Tuple[float, float, float]]  # (x, y, time)
    start_time: float
    end_time: float
    heading: float = 0.0  # Fixed heading for parked vehicles
    
    def get_position_at_time(self, t: float) -> Optional[Tuple[float, float]]:
        """Get position at time t."""
        if t < self.start_time or t > self.end_time:
            return None
        
        if len(self.trajectory) == 1:
            return (self.trajectory[0][0], self.trajectory[0][1])
        
        # Find segment
        for i in range(len(self.trajectory) - 1):
            t0 = self.trajectory[i][2]
            t1 = self.trajectory[i + 1][2]
            
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
                x = self.trajectory[i][0] + alpha * (self.trajectory[i + 1][0] - self.trajectory[i][0])
                y = self.trajectory[i][1] + alpha * (self.trajectory[i + 1][1] - self.trajectory[i][1])
                return (x, y)
        
        return (self.trajectory[-1][0], self.trajectory[-1][1])
    
    def get_heading_at_time(self, t: float) -> float:
        """Get heading angle at time t."""
        # For parked vehicles, use fixed heading
        if len(self.trajectory) < 2:
            return self.heading
        
        for i in range(len(self.trajectory) - 1):
            t0 = self.trajectory[i][2]
            t1 = self.trajectory[i + 1][2]
            
            if t0 <= t <= t1:
                dx = self.trajectory[i + 1][0] - self.trajectory[i][0]
                dy = self.trajectory[i + 1][1] - self.trajectory[i][1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    return np.arctan2(dy, dx)
                return self.heading
        
        return self.heading


def create_solid_mesh(obj_type: ObjectType, color: Tuple[float, float, float],
                      position: Tuple[float, float], heading: float, 
                      ground_z: float) -> o3d.geometry.TriangleMesh:
    """Create a solid mesh for an object."""
    dims = OBJECT_DIMENSIONS[obj_type]
    length, width, height = dims[0], dims[1], dims[2]
    
    if obj_type == ObjectType.PERSON:
        # Cylinder + sphere for person
        body = o3d.geometry.TriangleMesh.create_cylinder(radius=width/2, height=height*0.7)
        body.translate([0, 0, height * 0.35])
        
        head = o3d.geometry.TriangleMesh.create_sphere(radius=width/2 * 0.8)
        head.translate([0, 0, height * 0.85])
        
        mesh = body + head
        
    elif obj_type == ObjectType.BICYCLE:
        # Thin box for bicycle
        mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
        mesh.translate([-length/2, -width/2, 0])
        
    else:
        # Box for vehicles
        mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
        mesh.translate([-length/2, -width/2, 0])
    
    # Apply color
    mesh.paint_uniform_color(color)
    
    # Rotate by heading (around Z axis)
    R = mesh.get_rotation_matrix_from_xyz([0, 0, heading])
    mesh.rotate(R, center=[0, 0, 0])
    
    # Translate to position
    mesh.translate([position[0], position[1], ground_z])
    
    mesh.compute_vertex_normals()
    
    return mesh


class SceneGenerator:
    """Generate 4D scene from annotations."""
    
    def __init__(self, static_scene_path: Path, annotations_path: Path):
        # Load static scene
        print("Loading static scene...")
        self.static_pcd = o3d.io.read_point_cloud(str(static_scene_path))
        self.static_points = np.asarray(self.static_pcd.points)
        self.static_colors = np.asarray(self.static_pcd.colors)
        self.ground_z = np.percentile(self.static_points[:, 2], 5)
        print(f"  {len(self.static_points):,} points, ground z={self.ground_z:.2f}")
        
        # Load annotations
        print("Loading annotations...")
        with open(annotations_path) as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        
        # Convert to shapely polygons
        self.road_polygons = [Polygon(p) for p in self.annotations.get('roads', [])]
        self.crosswalk_polygons = [Polygon(p) for p in self.annotations.get('crosswalks', [])]
        self.sidewalk_polygons = [Polygon(p) for p in self.annotations.get('sidewalks', [])]
        self.parking_polygons = [Polygon(p) for p in self.annotations.get('parking', [])]
        
        # Compute road area (excluding crosswalks for vehicle paths)
        if self.road_polygons:
            self.road_area = unary_union(self.road_polygons)
        else:
            self.road_area = None
            
        print(f"  Roads: {len(self.road_polygons)}, Crosswalks: {len(self.crosswalk_polygons)}")
        print(f"  Sidewalks: {len(self.sidewalk_polygons)}, Parking: {len(self.parking_polygons)}")
        
        # Objects
        self.objects: List[DynamicObject] = []
        self.next_id = 0
        
        # Occupied positions for collision checking
        self.occupied_positions: List[Tuple[float, float, float]] = []  # (x, y, radius)
        
        # Duration
        self.duration = 60.0
    
    def _get_polygon_primary_axis(self, polygon: Polygon) -> Tuple[np.ndarray, float]:
        """Get primary axis direction and length of a polygon."""
        coords = np.array(polygon.exterior.coords)[:-1]  # Remove closing point
        
        # Find the two points that are furthest apart
        max_dist = 0
        p1, p2 = coords[0], coords[1]
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist > max_dist:
                    max_dist = dist
                    p1, p2 = coords[i], coords[j]
        
        direction = (p2 - p1)[:2]
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        else:
            direction = np.array([1, 0])
        
        return direction, length
    
    def _get_polygon_width(self, polygon: Polygon, direction: np.ndarray) -> float:
        """Get width of polygon perpendicular to direction."""
        coords = np.array(polygon.exterior.coords)[:-1]
        perp = np.array([-direction[1], direction[0]])
        
        projections = np.dot(coords[:, :2], perp)
        return projections.max() - projections.min()
    
    def _is_position_free(self, x: float, y: float, radius: float) -> bool:
        """Check if position is free of collisions."""
        for ox, oy, oradius in self.occupied_positions:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < radius + oradius + 0.5:  # 0.5m buffer
                return False
        return True
    
    def _mark_position_occupied(self, x: float, y: float, radius: float):
        """Mark a position as occupied."""
        self.occupied_positions.append((x, y, radius))
    
    def generate_parked_vehicles(self, count: int = 12):
        """Generate parked vehicles ONLY in parking areas."""
        print(f"\nGenerating {count} parked vehicles...")
        
        if not self.parking_polygons:
            print("  No parking areas defined!")
            return
        
        placed = 0
        
        # Distribute across parking areas
        cars_per_area = max(1, count // len(self.parking_polygons))
        
        for parking_poly in self.parking_polygons:
            if placed >= count:
                break
                
            # Get parking area orientation
            direction, length = self._get_polygon_primary_axis(parking_poly)
            width = self._get_polygon_width(parking_poly, direction)
            
            # Heading aligned with parking direction
            heading = np.arctan2(direction[1], direction[0])
            
            # Get centroid and bounds
            centroid = np.array([parking_poly.centroid.x, parking_poly.centroid.y])
            
            # Project corners to get extent along direction
            coords = np.array(parking_poly.exterior.coords)[:-1]
            projections = np.dot(coords[:, :2] - centroid, direction)
            min_proj, max_proj = projections.min(), projections.max()
            
            # Car dimensions
            car_length = OBJECT_DIMENSIONS[ObjectType.CAR][0]
            car_width = OBJECT_DIMENSIONS[ObjectType.CAR][1]
            
            # Space cars along the parking area
            spacing = car_length + 1.5  # Gap between cars
            num_spots = int((max_proj - min_proj) / spacing)
            
            for i in range(min(num_spots, cars_per_area)):
                if placed >= count:
                    break
                
                # Position along parking
                t = min_proj + spacing/2 + i * spacing
                pos = centroid + direction * t
                
                # Check if inside parking polygon
                if not parking_poly.contains(Point(pos[0], pos[1])):
                    continue
                
                # Check collision
                radius = max(car_length, car_width) / 2
                if not self._is_position_free(pos[0], pos[1], radius):
                    continue
                
                # Create parked vehicle (all gray)
                obj = DynamicObject(
                    id=self.next_id,
                    obj_type=ObjectType.CAR,
                    color=PARKED_COLOR,
                    trajectory=[(pos[0], pos[1], 0)],
                    start_time=0,
                    end_time=self.duration,
                    heading=heading
                )
                self.objects.append(obj)
                self._mark_position_occupied(pos[0], pos[1], radius)
                self.next_id += 1
                placed += 1
        
        print(f"  Placed {placed} parked vehicles")
    
    def generate_moving_vehicles(self, count: int = 12):
        """Generate moving vehicles ONLY on roads."""
        print(f"\nGenerating {count} moving vehicles...")
        
        if not self.road_area:
            print("  No roads defined!")
            return
        
        # Get road direction and create lanes
        road_poly = self.road_polygons[0]  # Main road
        direction, length = self._get_polygon_primary_axis(road_poly)
        perp = np.array([-direction[1], direction[0]])
        
        centroid = np.array([road_poly.centroid.x, road_poly.centroid.y])
        
        # Project to get road extent
        coords = np.array(road_poly.exterior.coords)[:-1]
        proj_along = np.dot(coords[:, :2] - centroid, direction)
        proj_perp = np.dot(coords[:, :2] - centroid, perp)
        
        road_start = proj_along.min()
        road_end = proj_along.max()
        road_width = proj_perp.max() - proj_perp.min()
        
        # Define lanes (NYC right-hand traffic)
        # Lane 1: direction, offset to right
        # Lane 2: opposite direction, offset to left
        lane_offset = road_width / 4  # Offset from center
        
        lanes = [
            {'direction': direction, 'offset': perp * lane_offset},
            {'direction': -direction, 'offset': -perp * lane_offset}
        ]
        
        # Vehicle types distribution
        vehicle_types = [ObjectType.CAR] * 8 + [ObjectType.TRUCK] * 2 + [ObjectType.BUS] * 1 + [ObjectType.BICYCLE] * 1
        
        placed = 0
        
        for i in range(count):
            # Alternate lanes
            lane = lanes[i % 2]
            lane_dir = lane['direction']
            lane_center = centroid + lane['offset']
            
            # Random start time (stagger vehicles)
            start_time = (i // 2) * 5.0 + np.random.uniform(0, 2)
            if start_time > self.duration - 10:
                start_time = np.random.uniform(0, self.duration - 15)
            
            # Start and end positions
            if np.dot(lane_dir, direction) > 0:
                start_pos = lane_center + direction * road_start
                end_pos = lane_center + direction * road_end
            else:
                start_pos = lane_center + direction * road_end
                end_pos = lane_center + direction * road_start
            
            # Speed
            speed = np.random.uniform(6, 10)  # m/s
            travel_time = np.linalg.norm(end_pos - start_pos) / speed
            end_time = start_time + travel_time
            
            if end_time > self.duration:
                # Adjust to fit
                travel_time = self.duration - start_time - 1
                end_pos = start_pos + lane_dir * (speed * travel_time)
                end_time = start_time + travel_time
            
            # Vehicle type
            vtype = vehicle_types[i % len(vehicle_types)]
            
            obj = DynamicObject(
                id=self.next_id,
                obj_type=vtype,
                color=MOVING_COLORS[vtype],
                trajectory=[
                    (start_pos[0], start_pos[1], start_time),
                    (end_pos[0], end_pos[1], end_time)
                ],
                start_time=start_time,
                end_time=end_time + 1
            )
            self.objects.append(obj)
            self.next_id += 1
            placed += 1
        
        print(f"  Created {placed} moving vehicles")
    
    def generate_pedestrians(self, count: int = 10):
        """Generate pedestrians on sidewalks and crossing crosswalks."""
        print(f"\nGenerating {count} pedestrians...")
        
        placed = 0
        
        # Split: some on sidewalks, some crossing
        sidewalk_count = count // 2
        crosswalk_count = count - sidewalk_count
        
        # Sidewalk pedestrians - walk ALONG the sidewalk
        if self.sidewalk_polygons:
            for i in range(sidewalk_count):
                sidewalk = self.sidewalk_polygons[i % len(self.sidewalk_polygons)]
                
                # Get sidewalk direction (walk along it)
                direction, length = self._get_polygon_primary_axis(sidewalk)
                
                # Random start position inside sidewalk
                centroid = np.array([sidewalk.centroid.x, sidewalk.centroid.y])
                
                # Project to get extent
                coords = np.array(sidewalk.exterior.coords)[:-1]
                projections = np.dot(coords[:, :2] - centroid, direction)
                min_proj, max_proj = projections.min(), projections.max()
                
                # Start near one end
                start_t = np.random.uniform(min_proj + 1, min_proj + (max_proj - min_proj) * 0.3)
                start_pos = centroid + direction * start_t
                
                # Walk along sidewalk
                walk_dist = np.random.uniform(5, min(15, (max_proj - min_proj) * 0.6))
                
                # Random direction along sidewalk
                if np.random.random() < 0.5:
                    end_pos = start_pos + direction * walk_dist
                else:
                    end_pos = start_pos - direction * walk_dist
                
                # Timing
                start_time = np.random.uniform(0, self.duration * 0.6)
                speed = np.random.uniform(1.0, 1.5)
                walk_time = walk_dist / speed
                end_time = start_time + walk_time
                
                if end_time > self.duration:
                    continue
                
                obj = DynamicObject(
                    id=self.next_id,
                    obj_type=ObjectType.PERSON,
                    color=PEDESTRIAN_COLOR,
                    trajectory=[
                        (start_pos[0], start_pos[1], start_time),
                        (end_pos[0], end_pos[1], end_time)
                    ],
                    start_time=start_time,
                    end_time=end_time + 2
                )
                self.objects.append(obj)
                self.next_id += 1
                placed += 1
        
        # Crosswalk pedestrians - walk ACROSS (perpendicular to crosswalk length)
        if self.crosswalk_polygons:
            for i in range(crosswalk_count):
                crosswalk = self.crosswalk_polygons[i % len(self.crosswalk_polygons)]
                
                # Get crosswalk primary axis (this is the LENGTH of crosswalk, parallel to road)
                primary_dir, primary_len = self._get_polygon_primary_axis(crosswalk)
                
                # Crossing direction is PERPENDICULAR to primary axis
                cross_dir = np.array([-primary_dir[1], primary_dir[0]])
                
                # Get crosswalk center and width (crossing distance)
                centroid = np.array([crosswalk.centroid.x, crosswalk.centroid.y])
                cross_width = self._get_polygon_width(crosswalk, primary_dir)
                
                # Start at one edge, end at other edge
                start_pos = centroid - cross_dir * (cross_width / 2 - 0.5)
                end_pos = centroid + cross_dir * (cross_width / 2 - 0.5)
                
                # Random direction
                if np.random.random() < 0.5:
                    start_pos, end_pos = end_pos, start_pos
                
                # Stagger crossing times
                start_time = i * 8.0 + np.random.uniform(0, 3)
                if start_time > self.duration - 15:
                    start_time = np.random.uniform(0, self.duration - 15)
                
                speed = np.random.uniform(1.0, 1.4)
                cross_time = cross_width / speed
                end_time = start_time + cross_time
                
                if end_time > self.duration:
                    continue
                
                obj = DynamicObject(
                    id=self.next_id,
                    obj_type=ObjectType.PERSON,
                    color=PEDESTRIAN_COLOR,
                    trajectory=[
                        (start_pos[0], start_pos[1], start_time),
                        (end_pos[0], end_pos[1], end_time)
                    ],
                    start_time=start_time,
                    end_time=end_time + 2
                )
                self.objects.append(obj)
                self.next_id += 1
                placed += 1
        
        print(f"  Created {placed} pedestrians")
    
    def generate_all(self, parked: int = 12, moving: int = 12, pedestrians: int = 10):
        """Generate complete scene."""
        print("\n" + "="*60)
        print("GENERATING 4D SCENE")
        print("="*60)
        
        # Clear previous
        self.objects = []
        self.occupied_positions = []
        self.next_id = 0
        
        self.generate_parked_vehicles(parked)
        self.generate_moving_vehicles(moving)
        self.generate_pedestrians(pedestrians)
        
        print(f"\nTotal objects: {len(self.objects)}")
        print(f"  Parked (gray): {sum(1 for o in self.objects if len(o.trajectory) == 1)}")
        print(f"  Moving vehicles: {sum(1 for o in self.objects if o.obj_type != ObjectType.PERSON and len(o.trajectory) > 1)}")
        print(f"  Pedestrians (blue): {sum(1 for o in self.objects if o.obj_type == ObjectType.PERSON)}")
    
    def get_meshes_at_time(self, t: float) -> List[o3d.geometry.TriangleMesh]:
        """Get all object meshes at time t."""
        meshes = []
        
        for obj in self.objects:
            pos = obj.get_position_at_time(t)
            if pos is not None:
                heading = obj.get_heading_at_time(t)
                mesh = create_solid_mesh(obj.obj_type, obj.color, pos, heading, self.ground_z)
                meshes.append(mesh)
        
        return meshes
    
    def save(self, path: Path):
        """Save scene configuration."""
        data = {
            'duration': self.duration,
            'ground_z': self.ground_z,
            'objects': []
        }
        
        for obj in self.objects:
            data['objects'].append({
                'id': obj.id,
                'type': obj.obj_type.value,
                'color': list(obj.color),
                'trajectory': obj.trajectory,
                'start_time': obj.start_time,
                'end_time': obj.end_time,
                'heading': obj.heading
            })
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved to {path}")
    
    @classmethod
    def load(cls, static_scene_path: Path, config_path: Path) -> 'SceneGenerator':
        """Load scene from config."""
        gen = cls.__new__(cls)
        
        # Load static scene
        gen.static_pcd = o3d.io.read_point_cloud(str(static_scene_path))
        gen.static_points = np.asarray(gen.static_pcd.points)
        gen.static_colors = np.asarray(gen.static_pcd.colors)
        gen.ground_z = np.percentile(gen.static_points[:, 2], 5)
        
        # Load config
        with open(config_path) as f:
            data = json.load(f)
        
        gen.duration = data['duration']
        gen.objects = []
        gen.next_id = 0
        gen.occupied_positions = []
        
        for obj_data in data['objects']:
            obj = DynamicObject(
                id=obj_data['id'],
                obj_type=ObjectType(obj_data['type']),
                color=tuple(obj_data['color']),
                trajectory=[tuple(t) for t in obj_data['trajectory']],
                start_time=obj_data['start_time'],
                end_time=obj_data['end_time'],
                heading=obj_data.get('heading', 0.0)
            )
            gen.objects.append(obj)
            gen.next_id = max(gen.next_id, obj.id + 1)
        
        return gen


def main():
    base_dir = Path(__file__).parent.parent
    static_scene = base_dir / "outputs" / "pass1_static" / "pi3_pointcloud_corrected.ply"
    annotations_path = Path(__file__).parent / "scene_annotations.json"
    config_path = Path(__file__).parent / "generated_scene.json"
    
    if not annotations_path.exists():
        print(f"No annotations at {annotations_path}")
        print("Run polygon_editor.py first")
        return
    
    gen = SceneGenerator(static_scene, annotations_path)
    gen.generate_all(parked=12, moving=12, pedestrians=10)
    gen.save(config_path)


if __name__ == "__main__":
    main()
