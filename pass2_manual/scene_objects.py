#!/usr/bin/env python3
"""
Canonical object representations for scene editing.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json


class ObjectType(Enum):
    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"


# Canonical dimensions (length, width, height) in meters
OBJECT_DIMENSIONS = {
    ObjectType.PERSON: (0.5, 0.5, 1.7),
    ObjectType.BICYCLE: (1.8, 0.6, 1.1),
    ObjectType.CAR: (4.5, 1.8, 1.5),
    ObjectType.TRUCK: (7.0, 2.5, 3.0),
    ObjectType.BUS: (12.0, 2.5, 3.2),
}

# Default colors
OBJECT_COLORS = {
    ObjectType.PERSON: (0.2, 0.4, 0.8),      # Blue
    ObjectType.BICYCLE: (0.1, 0.7, 0.3),     # Green
    ObjectType.CAR: (0.8, 0.2, 0.2),         # Red
    ObjectType.TRUCK: (0.6, 0.4, 0.2),       # Brown
    ObjectType.BUS: (0.9, 0.7, 0.1),         # Yellow
}


@dataclass
class TrajectoryPoint:
    """A point on a trajectory with timing."""
    x: float
    y: float
    time: float  # Time in seconds when object reaches this point
    
    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'time': self.time}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrajectoryPoint':
        return cls(x=d['x'], y=d['y'], time=d['time'])


@dataclass
class SceneObject:
    """An object in the scene with trajectory."""
    id: int
    object_type: ObjectType
    color: Tuple[float, float, float]
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    
    # Timing
    start_time: float = 0.0  # When object appears
    end_time: float = 10.0   # When object disappears
    
    # Optional stops (list of (stop_time, resume_time) tuples)
    stops: List[Tuple[float, float]] = field(default_factory=list)
    
    def get_dimensions(self) -> Tuple[float, float, float]:
        return OBJECT_DIMENSIONS[self.object_type]
    
    def get_position_at_time(self, t: float) -> Optional[Tuple[float, float, float]]:
        """Get object position at time t. Returns None if not visible."""
        if t < self.start_time or t > self.end_time:
            return None
        
        if len(self.trajectory) == 0:
            return None
        
        if len(self.trajectory) == 1:
            return (self.trajectory[0].x, self.trajectory[0].y, 0.0)
        
        # Check if stopped
        for stop_time, resume_time in self.stops:
            if stop_time <= t < resume_time:
                # Find position at stop time
                return self._interpolate_position(stop_time)
        
        return self._interpolate_position(t)
    
    def _interpolate_position(self, t: float) -> Tuple[float, float, float]:
        """Interpolate position along trajectory at time t."""
        # Find segment
        for i in range(len(self.trajectory) - 1):
            p1 = self.trajectory[i]
            p2 = self.trajectory[i + 1]
            
            if p1.time <= t <= p2.time:
                # Linear interpolation
                if p2.time == p1.time:
                    alpha = 0
                else:
                    alpha = (t - p1.time) / (p2.time - p1.time)
                
                x = p1.x + alpha * (p2.x - p1.x)
                y = p1.y + alpha * (p2.y - p1.y)
                return (x, y, 0.0)
        
        # Before first point
        if t <= self.trajectory[0].time:
            return (self.trajectory[0].x, self.trajectory[0].y, 0.0)
        
        # After last point
        return (self.trajectory[-1].x, self.trajectory[-1].y, 0.0)
    
    def get_heading_at_time(self, t: float) -> float:
        """Get object heading (radians) at time t."""
        if len(self.trajectory) < 2:
            return 0.0
        
        # Find current segment
        for i in range(len(self.trajectory) - 1):
            p1 = self.trajectory[i]
            p2 = self.trajectory[i + 1]
            
            if p1.time <= t <= p2.time:
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                return np.arctan2(dy, dx)
        
        # Use last segment direction
        p1 = self.trajectory[-2]
        p2 = self.trajectory[-1]
        return np.arctan2(p2.y - p1.y, p2.x - p1.x)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'object_type': self.object_type.value,
            'color': list(self.color),
            'trajectory': [p.to_dict() for p in self.trajectory],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'stops': self.stops
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SceneObject':
        obj = cls(
            id=d['id'],
            object_type=ObjectType(d['object_type']),
            color=tuple(d['color']),
            start_time=d['start_time'],
            end_time=d['end_time'],
            stops=[tuple(s) for s in d.get('stops', [])]
        )
        obj.trajectory = [TrajectoryPoint.from_dict(p) for p in d['trajectory']]
        return obj


def create_object_points(obj: SceneObject, position: Tuple[float, float, float], 
                         heading: float, density: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create point cloud for an object at given position and heading."""
    points = []
    colors = []
    
    l, w, h = obj.get_dimensions()
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    
    R = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    pos = np.array(position)
    
    if obj.object_type == ObjectType.PERSON:
        # Cylinder body + sphere head
        body_h = h * 0.7
        body_r = w / 2
        
        # Body
        for z in np.arange(0, body_h, density):
            for angle in np.arange(0, 2*np.pi, 0.5):
                x = body_r * np.cos(angle)
                y = body_r * np.sin(angle)
                local = np.array([x, y, z])
                world = R @ local + pos
                points.append(world)
                colors.append(obj.color)
        
        # Head
        head_z = body_h + 0.12
        head_r = 0.12
        for phi in np.arange(0, np.pi, 0.5):
            for theta in np.arange(0, 2*np.pi, 0.5):
                x = head_r * np.sin(phi) * np.cos(theta)
                y = head_r * np.sin(phi) * np.sin(theta)
                z = head_r * np.cos(phi) + head_z
                local = np.array([x, y, z])
                world = R @ local + pos
                points.append(world)
                colors.append((0.9, 0.7, 0.5))  # Skin
    
    elif obj.object_type == ObjectType.BICYCLE:
        # Simple bicycle shape
        # Wheels
        wheel_r = 0.35
        for wheel_x in [-l/3, l/3]:
            for angle in np.arange(0, 2*np.pi, 0.3):
                x = wheel_x
                y = 0
                z = wheel_r + wheel_r * np.sin(angle)
                local = np.array([x, y, z])
                world = R @ local + pos
                points.append(world)
                colors.append((0.2, 0.2, 0.2))
        
        # Frame
        for x in np.arange(-l/3, l/3, density):
            local = np.array([x, 0, wheel_r + 0.3])
            world = R @ local + pos
            points.append(world)
            colors.append(obj.color)
        
        # Rider (simplified)
        rider_pos = pos + R @ np.array([0, 0, wheel_r + 0.5])
        for z in np.arange(0, 0.6, density):
            for angle in np.arange(0, 2*np.pi, 0.6):
                x = 0.15 * np.cos(angle)
                y = 0.15 * np.sin(angle)
                local = np.array([x, y, z])
                world = R @ local + rider_pos
                points.append(world)
                colors.append((0.3, 0.4, 0.6))
    
    else:
        # Box-shaped vehicles (car, truck, bus)
        # Bottom
        for x in np.arange(-l/2, l/2, density):
            for y in np.arange(-w/2, w/2, density):
                local = np.array([x, y, 0.15])
                world = R @ local + pos
                points.append(world)
                colors.append(obj.color)
        
        # Top
        for x in np.arange(-l/2, l/2, density):
            for y in np.arange(-w/2, w/2, density):
                local = np.array([x, y, h])
                world = R @ local + pos
                points.append(world)
                colors.append(obj.color)
        
        # Sides
        for x in np.arange(-l/2, l/2, density):
            for z in np.arange(0.15, h, density):
                for y_side in [-w/2, w/2]:
                    local = np.array([x, y_side, z])
                    world = R @ local + pos
                    points.append(world)
                    colors.append(obj.color)
        
        # Front and back
        for y in np.arange(-w/2, w/2, density):
            for z in np.arange(0.15, h, density):
                for x_side in [-l/2, l/2]:
                    local = np.array([x_side, y, z])
                    world = R @ local + pos
                    points.append(world)
                    colors.append(obj.color)
        
        # Windows (darker) for cars/buses
        if obj.object_type in [ObjectType.CAR, ObjectType.BUS]:
            window_color = tuple(c * 0.3 for c in obj.color)
            win_h_start = h * 0.5
            win_h_end = h * 0.9
            for x in np.arange(-l/4, l/3, density):
                for z in np.arange(win_h_start, win_h_end, density):
                    for y_side in [-w/2 - 0.01, w/2 + 0.01]:
                        local = np.array([x, y_side, z])
                        world = R @ local + pos
                        points.append(world)
                        colors.append(window_color)
    
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    return np.array(points), np.array(colors)


class SceneConfig:
    """Configuration for the entire scene."""
    
    def __init__(self):
        self.objects: List[SceneObject] = []
        self.next_id: int = 0
        self.duration: float = 60.0  # Total scene duration in seconds
    
    def add_object(self, object_type: ObjectType, 
                   color: Optional[Tuple[float, float, float]] = None) -> SceneObject:
        """Add a new object to the scene."""
        if color is None:
            color = OBJECT_COLORS[object_type]
        
        obj = SceneObject(
            id=self.next_id,
            object_type=object_type,
            color=color
        )
        self.objects.append(obj)
        self.next_id += 1
        return obj
    
    def remove_object(self, obj_id: int):
        """Remove an object by ID."""
        self.objects = [o for o in self.objects if o.id != obj_id]
    
    def get_object(self, obj_id: int) -> Optional[SceneObject]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None
    
    def save(self, path: str):
        """Save scene configuration to JSON."""
        data = {
            'duration': self.duration,
            'next_id': self.next_id,
            'objects': [obj.to_dict() for obj in self.objects]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SceneConfig':
        """Load scene configuration from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls()
        config.duration = data['duration']
        config.next_id = data['next_id']
        config.objects = [SceneObject.from_dict(d) for d in data['objects']]
        return config
