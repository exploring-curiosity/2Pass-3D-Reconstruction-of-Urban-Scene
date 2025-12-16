#!/usr/bin/env python3
"""
Line-of-Sight (LOS) Audit System for Traffic Intersection Safety Analysis

Implements visibility analysis using:
1. Biological scale calibration (median pedestrian height = 1.7m)
2. Voxel grid spatial indexing (20cm leaf size)
3. Ray bundle casting to K=5 vehicle keypoints
4. Visibility scoring (>60% rays clear = visible)
5. Occluder identification (static, parked, moving)

Usage:
    python los_audit.py --scene scene_with_objects.ply [--output los_report.json]
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import open3d as o3d
from scipy.spatial import cKDTree
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

# Color mappings from PLY - EXACT colors found in the file
# Person (Blue): [51, 153, 230] → (0.20, 0.60, 0.90)
# Car (Red): [230, 51, 51] → (0.90, 0.20, 0.20)
# Truck (Green): [51, 179, 51] → (0.20, 0.70, 0.20)
# Cycle (Pink): [204, 51, 204] → (0.80, 0.20, 0.80)

# Reference colors (normalized 0-1)
OBJECT_COLORS_RGB = {
    'person': np.array([51, 153, 230]) / 255.0,   # Blue with medium green
    'car': np.array([230, 51, 51]) / 255.0,       # Red
    'truck': np.array([51, 179, 51]) / 255.0,     # Green
    'cycle': np.array([204, 51, 204]) / 255.0,    # Pink/Magenta
}

# Color matching tolerance (Euclidean distance in RGB space)
COLOR_TOLERANCE = 0.15  # ~15% tolerance

# Legacy thresholds (backup)
COLOR_THRESHOLDS = {
    'car': {'r': (0.6, 1.0), 'g': (0.0, 0.4), 'b': (0.0, 0.4)},
    'truck': {'r': (0.0, 0.4), 'g': (0.5, 1.0), 'b': (0.0, 0.4)},
    'person': {'r': (0.0, 0.4), 'g': (0.4, 0.8), 'b': (0.7, 1.0)},
    'cycle': {'r': (0.6, 1.0), 'g': (0.0, 0.4), 'b': (0.6, 1.0)},
}

def analyze_colors(colors: np.ndarray):
    """Debug: Analyze color distribution in point cloud."""
    print("\n  [DEBUG] Color Analysis:")
    print(f"    Color range: min={colors.min():.3f}, max={colors.max():.3f}")
    
    # If colors are in 0-255 range, normalize
    if colors.max() > 1.5:
        print("    Detected 0-255 range, normalizing...")
        colors = colors / 255.0
    
    # Find distinct colors by rounding
    rounded = np.round(colors, 1)
    unique_colors = np.unique(rounded, axis=0)
    print(f"    Unique colors (rounded): {len(unique_colors)}")
    
    # Count points per dominant color
    r_dom = np.sum((colors[:, 0] > colors[:, 1]) & (colors[:, 0] > colors[:, 2]))
    g_dom = np.sum((colors[:, 1] > colors[:, 0]) & (colors[:, 1] > colors[:, 2]))
    b_dom = np.sum((colors[:, 2] > colors[:, 0]) & (colors[:, 2] > colors[:, 1]))
    
    # Pink = high R and high B
    pink_mask = (colors[:, 0] > 0.5) & (colors[:, 2] > 0.5) & (colors[:, 1] < 0.3)
    pink_count = np.sum(pink_mask)
    
    # Pure blue = high B, low R and G
    blue_mask = (colors[:, 2] > 0.5) & (colors[:, 0] < 0.3) & (colors[:, 1] < 0.3)
    blue_count = np.sum(blue_mask)
    
    # Pure red = high R, low G and B
    red_mask = (colors[:, 0] > 0.5) & (colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
    red_count = np.sum(red_mask)
    
    # Pure green = high G, low R and B
    green_mask = (colors[:, 1] > 0.5) & (colors[:, 0] < 0.3) & (colors[:, 2] < 0.3)
    green_count = np.sum(green_mask)
    
    print(f"    Red-dominant points: {r_dom:,}")
    print(f"    Green-dominant points: {g_dom:,}")
    print(f"    Blue-dominant points: {b_dom:,}")
    print(f"    Pure RED (car) points: {red_count:,}")
    print(f"    Pure GREEN (truck) points: {green_count:,}")
    print(f"    Pure BLUE (person) points: {blue_count:,}")
    print(f"    PINK/Magenta (cycle) points: {pink_count:,}")
    
    # Sample some distinct colors
    if len(unique_colors) <= 20:
        print("    Sample unique colors:")
        for c in unique_colors[:10]:
            print(f"      RGB: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
    
    return colors

OBJECT_DIMS = {
    'car': (4.5, 1.8, 1.5),
    'truck': (7.0, 2.5, 3.0),
    'person': (0.5, 0.5, 1.7),
    'cycle': (1.8, 0.6, 1.1),
}

REFERENCE_PERSON_HEIGHT = 1.7
PERSON_EYE_HEIGHT = 1.6
VOXEL_SIZE = 0.20
VISIBILITY_THRESHOLD = 0.60
RAY_STEP_SIZE = 0.10  # Step size for ray marching

# Vehicle keypoints: (name, rel_x, rel_y, rel_z) normalized by dimensions
VEHICLE_KEYPOINTS = [
    ('bumper_center', 0.5, 0.0, 0.1),
    ('headlight_left', 0.45, 0.4, 0.25),
    ('headlight_right', 0.45, -0.4, 0.25),
    ('hood_center', 0.25, 0.0, 0.5),
    ('roof_front', 0.0, 0.0, 0.95),
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DynamicObject:
    obj_id: str
    obj_class: str
    centroid: np.ndarray
    points: np.ndarray
    is_stationary: bool = False
    heading: float = 0.0


@dataclass
class OcclusionInfo:
    occluder_type: str  # 'static', 'parked_vehicle', 'moving_vehicle', 'pedestrian'
    occluder_id: Optional[str] = None
    hit_point: Optional[np.ndarray] = None
    distance: float = 0.0


@dataclass
class RayResult:
    keypoint_name: str
    target_point: np.ndarray
    is_clear: bool
    occlusion: Optional[OcclusionInfo] = None


@dataclass
class VisibilityResult:
    pedestrian_id: str
    vehicle_id: str
    pedestrian_pos: np.ndarray
    vehicle_pos: np.ndarray
    visibility_score: float
    is_visible: bool
    ray_results: List[RayResult] = field(default_factory=list)
    primary_occluder: Optional[str] = None
    distance: float = 0.0


# =============================================================================
# POINT CLOUD SEGMENTATION
# =============================================================================

def classify_point_by_color(color: np.ndarray) -> Optional[str]:
    """
    Classify a point by its RGB color using distance to reference colors.
    
    Red = car, Green = truck, Blue = person, Pink/Magenta = cycle
    """
    # Normalize if in 0-255 range
    if np.max(color) > 1.5:
        color = color / 255.0
    
    # Find closest reference color
    min_dist = float('inf')
    best_class = None
    
    for cls, ref_color in OBJECT_COLORS_RGB.items():
        dist = np.linalg.norm(color - ref_color)
        if dist < min_dist and dist < COLOR_TOLERANCE:
            min_dist = dist
            best_class = cls
    
    return best_class  # None if no match within tolerance


def segment_scene(points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
    """
    Segment point cloud into static background and dynamic objects by color.
    
    Returns:
        static_points: Points belonging to static infrastructure
        dynamic_clusters: Dict mapping class -> list of point clusters
    """
    print("Segmenting scene by color...")
    
    # Analyze and normalize colors
    colors = analyze_colors(colors)
    
    # Normalize colors if in 0-255 range
    if colors.max() > 1.5:
        colors = colors / 255.0
    
    static_mask = np.ones(len(points), dtype=bool)
    class_masks = {cls: np.zeros(len(points), dtype=bool) for cls in OBJECT_COLORS_RGB}
    
    # Vectorized classification using distance to reference colors
    print("  Classifying points by color distance...")
    
    for cls, ref_color in OBJECT_COLORS_RGB.items():
        # Compute Euclidean distance to reference color
        distances = np.linalg.norm(colors - ref_color, axis=1)
        class_masks[cls] = distances < COLOR_TOLERANCE
    
    # Handle overlaps: assign to closest color
    print("  Resolving color overlaps...")
    all_distances = {}
    for cls, ref_color in OBJECT_COLORS_RGB.items():
        all_distances[cls] = np.linalg.norm(colors - ref_color, axis=1)
    
    # For each point, find the closest matching class
    for i, cls in enumerate(OBJECT_COLORS_RGB.keys()):
        # Only keep if this class is the closest among all matches
        is_closest = np.ones(len(points), dtype=bool)
        for other_cls in OBJECT_COLORS_RGB.keys():
            if other_cls != cls:
                is_closest &= (all_distances[cls] <= all_distances[other_cls])
        class_masks[cls] &= is_closest
    
    # Update static mask
    for cls_mask in class_masks.values():
        static_mask &= ~cls_mask
    
    # Print classification counts
    print("  Classification results:")
    for cls, mask in class_masks.items():
        count = np.sum(mask)
        print(f"    {cls}: {count:,} points")
    
    static_points = points[static_mask]
    print(f"  Static points: {len(static_points):,}")
    
    # Cluster dynamic objects using DBSCAN
    dynamic_clusters = {}
    for cls, mask in class_masks.items():
        cls_points = points[mask]
        if len(cls_points) < 10:
            dynamic_clusters[cls] = []
            continue
        
        # Use Open3D DBSCAN clustering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cls_points)
        
        # Clustering parameters based on object size
        eps = 1.0 if cls in ['car', 'truck'] else 0.5
        min_points = 50 if cls in ['car', 'truck'] else 20
        
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        clusters = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            cluster_points = cls_points[labels == label]
            clusters.append(cluster_points)
        
        dynamic_clusters[cls] = clusters
        print(f"  {cls}: {len(clusters)} objects")
    
    return static_points, dynamic_clusters


def extract_dynamic_objects(dynamic_clusters: Dict[str, List[np.ndarray]]) -> List[DynamicObject]:
    """Convert point clusters to DynamicObject instances."""
    objects = []
    obj_counter = defaultdict(int)
    
    for cls, clusters in dynamic_clusters.items():
        for cluster_points in clusters:
            obj_counter[cls] += 1
            obj_id = f"{cls}_{obj_counter[cls]}"
            
            centroid = cluster_points.mean(axis=0)
            
            # Estimate heading from point distribution (PCA)
            if len(cluster_points) > 10 and cls != 'person':
                centered = cluster_points[:, :2] - centroid[:2]
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                heading = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
            else:
                heading = 0.0
            
            objects.append(DynamicObject(
                obj_id=obj_id,
                obj_class=cls,
                centroid=centroid,
                points=cluster_points,
                heading=heading
            ))
    
    return objects


# =============================================================================
# SCALE CALIBRATION
# =============================================================================

def calibrate_scale(dynamic_objects: List[DynamicObject]) -> float:
    """
    Calculate scale factor using biological prior (median pedestrian height = 1.7m).
    """
    print("\nCalibrating scale using pedestrian heights...")
    
    pedestrians = [obj for obj in dynamic_objects if obj.obj_class == 'person']
    
    if len(pedestrians) == 0:
        print("  WARNING: No pedestrians found, using scale = 1.0")
        return 1.0
    
    heights = []
    for ped in pedestrians:
        z_min = ped.points[:, 2].min()
        z_max = ped.points[:, 2].max()
        height = z_max - z_min
        if height > 0.5:  # Filter noise
            heights.append(height)
    
    if len(heights) == 0:
        print("  WARNING: No valid pedestrian heights, using scale = 1.0")
        return 1.0
    
    median_height = np.median(heights)
    scale_factor = REFERENCE_PERSON_HEIGHT / median_height
    
    print(f"  Pedestrians analyzed: {len(heights)}")
    print(f"  Median height (unscaled): {median_height:.3f}")
    print(f"  Scale factor: {scale_factor:.3f}")
    
    return scale_factor


def apply_scale(points: np.ndarray, scale: float) -> np.ndarray:
    """Apply scale factor to points."""
    return points * scale


# =============================================================================
# VOXEL GRID OCCUPANCY
# =============================================================================

class VoxelOccupancyGrid:
    """Efficient voxel-based occupancy grid for ray casting."""
    
    def __init__(self, points: np.ndarray, voxel_size: float = VOXEL_SIZE):
        self.voxel_size = voxel_size
        
        # Compute grid bounds
        self.min_bound = points.min(axis=0) - voxel_size
        self.max_bound = points.max(axis=0) + voxel_size
        
        # Voxelize points
        voxel_indices = np.floor((points - self.min_bound) / voxel_size).astype(np.int32)
        
        # Create set of occupied voxels
        self.occupied = set(map(tuple, voxel_indices))
        
        print(f"  Voxel grid: {len(self.occupied):,} occupied voxels")
        print(f"  Bounds: {self.min_bound} to {self.max_bound}")
    
    def is_occupied(self, point: np.ndarray) -> bool:
        """Check if a point falls in an occupied voxel."""
        voxel_idx = tuple(np.floor((point - self.min_bound) / self.voxel_size).astype(int))
        return voxel_idx in self.occupied
    
    def ray_intersects(self, origin: np.ndarray, target: np.ndarray, 
                       step_size: float = RAY_STEP_SIZE) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if ray from origin to target intersects any occupied voxel.
        
        Returns:
            (is_blocked, hit_point)
        """
        direction = target - origin
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return False, None
        
        direction = direction / distance
        num_steps = int(distance / step_size)
        
        for i in range(1, num_steps):  # Skip origin
            point = origin + direction * (i * step_size)
            if self.is_occupied(point):
                return True, point
        
        return False, None


# =============================================================================
# DYNAMIC OBJECT OCCUPANCY
# =============================================================================

class DynamicOccupancyChecker:
    """Check ray intersections with dynamic objects using KD-trees."""
    
    def __init__(self, objects: List[DynamicObject], exclude_ids: Set[str] = None):
        self.objects = objects
        self.exclude_ids = exclude_ids or set()
        
        # Build KD-tree for each object
        self.kdtrees = {}
        for obj in objects:
            if obj.obj_id not in self.exclude_ids:
                self.kdtrees[obj.obj_id] = cKDTree(obj.points)
    
    def ray_intersects(self, origin: np.ndarray, target: np.ndarray,
                       step_size: float = RAY_STEP_SIZE,
                       hit_radius: float = 0.3) -> Tuple[bool, Optional[OcclusionInfo]]:
        """
        Check if ray intersects any dynamic object.
        
        Returns:
            (is_blocked, occlusion_info)
        """
        direction = target - origin
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return False, None
        
        direction = direction / distance
        num_steps = int(distance / step_size)
        
        for i in range(1, num_steps):
            point = origin + direction * (i * step_size)
            
            for obj_id, tree in self.kdtrees.items():
                if obj_id in self.exclude_ids:
                    continue
                dist, _ = tree.query(point)
                if dist < hit_radius:
                    obj = next(o for o in self.objects if o.obj_id == obj_id)
                    
                    if obj.obj_class == 'person':
                        occluder_type = 'pedestrian'
                    elif obj.is_stationary:
                        occluder_type = 'parked_vehicle'
                    else:
                        occluder_type = 'moving_vehicle'
                    
                    return True, OcclusionInfo(
                        occluder_type=occluder_type,
                        occluder_id=obj_id,
                        hit_point=point,
                        distance=i * step_size
                    )
        
        return False, None


# =============================================================================
# VEHICLE KEYPOINT GENERATION
# =============================================================================

def get_vehicle_keypoints(obj: DynamicObject, scale: float) -> List[Tuple[str, np.ndarray]]:
    """
    Generate 3D keypoints for a vehicle based on its position and dimensions.
    """
    dims = OBJECT_DIMS.get(obj.obj_class, OBJECT_DIMS['car'])
    length, width, height = dims
    
    # Apply scale
    length *= scale
    width *= scale
    height *= scale
    
    keypoints = []
    cos_h, sin_h = np.cos(obj.heading), np.sin(obj.heading)
    
    for name, rel_x, rel_y, rel_z in VEHICLE_KEYPOINTS:
        # Local coordinates
        local_x = rel_x * length - length / 2  # Center to front
        local_y = rel_y * width
        local_z = rel_z * height
        
        # Rotate by heading
        world_x = cos_h * local_x - sin_h * local_y + obj.centroid[0]
        world_y = sin_h * local_x + cos_h * local_y + obj.centroid[1]
        world_z = local_z + obj.centroid[2]
        
        keypoints.append((name, np.array([world_x, world_y, world_z])))
    
    return keypoints


# =============================================================================
# VISIBILITY ANALYSIS
# =============================================================================

def compute_visibility(
    pedestrian: DynamicObject,
    vehicle: DynamicObject,
    static_voxels: VoxelOccupancyGrid,
    dynamic_checker: DynamicOccupancyChecker,
    scale: float
) -> VisibilityResult:
    """
    Compute visibility score for a pedestrian-vehicle pair.
    """
    # Pedestrian eye point (1.6m above ground)
    eye_height = PERSON_EYE_HEIGHT * scale
    ground_z = pedestrian.centroid[2] - (REFERENCE_PERSON_HEIGHT * scale) / 2
    eye_point = np.array([
        pedestrian.centroid[0],
        pedestrian.centroid[1],
        ground_z + eye_height
    ])
    
    # Get vehicle keypoints
    keypoints = get_vehicle_keypoints(vehicle, scale)
    
    ray_results = []
    clear_count = 0
    occluder_counts = defaultdict(int)
    
    for kp_name, kp_pos in keypoints:
        # Check static occlusion
        static_blocked, static_hit = static_voxels.ray_intersects(eye_point, kp_pos)
        
        if static_blocked:
            ray_results.append(RayResult(
                keypoint_name=kp_name,
                target_point=kp_pos,
                is_clear=False,
                occlusion=OcclusionInfo(
                    occluder_type='static',
                    hit_point=static_hit,
                    distance=np.linalg.norm(static_hit - eye_point)
                )
            ))
            occluder_counts['static'] += 1
            continue
        
        # Check dynamic occlusion
        dynamic_blocked, dynamic_info = dynamic_checker.ray_intersects(eye_point, kp_pos)
        
        if dynamic_blocked:
            ray_results.append(RayResult(
                keypoint_name=kp_name,
                target_point=kp_pos,
                is_clear=False,
                occlusion=dynamic_info
            ))
            occluder_counts[dynamic_info.occluder_type] += 1
            continue
        
        # Ray is clear
        ray_results.append(RayResult(
            keypoint_name=kp_name,
            target_point=kp_pos,
            is_clear=True
        ))
        clear_count += 1
    
    visibility_score = clear_count / len(keypoints)
    is_visible = visibility_score >= VISIBILITY_THRESHOLD
    
    # Determine primary occluder
    primary_occluder = None
    if occluder_counts:
        primary_occluder = max(occluder_counts, key=occluder_counts.get)
    
    distance = np.linalg.norm(vehicle.centroid[:2] - pedestrian.centroid[:2])
    
    return VisibilityResult(
        pedestrian_id=pedestrian.obj_id,
        vehicle_id=vehicle.obj_id,
        pedestrian_pos=pedestrian.centroid,
        vehicle_pos=vehicle.centroid,
        visibility_score=visibility_score,
        is_visible=is_visible,
        ray_results=ray_results,
        primary_occluder=primary_occluder,
        distance=distance
    )


# =============================================================================
# MAIN LOS AUDIT
# =============================================================================

class LOSAuditor:
    """Main LOS audit system."""
    
    def __init__(self, scene_ply_path: str):
        print("=" * 70)
        print("LOS AUDIT SYSTEM")
        print("=" * 70)
        
        # Load scene
        print(f"\nLoading scene: {scene_ply_path}")
        pcd = o3d.io.read_point_cloud(scene_ply_path)
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)
        print(f"  Total points: {len(self.points):,}")
        
        # Segment scene
        self.static_points, dynamic_clusters = segment_scene(self.points, self.colors)
        self.dynamic_objects = extract_dynamic_objects(dynamic_clusters)
        
        # Calibrate scale
        self.scale = calibrate_scale(self.dynamic_objects)
        
        # Apply scale
        print("\nApplying scale to scene...")
        self.static_points = apply_scale(self.static_points, self.scale)
        for obj in self.dynamic_objects:
            obj.centroid = apply_scale(obj.centroid.reshape(1, -1), self.scale).flatten()
            obj.points = apply_scale(obj.points, self.scale)
        
        # Build voxel grid for static scene
        print("\nBuilding voxel occupancy grid...")
        self.static_voxels = VoxelOccupancyGrid(self.static_points)
        
        # Identify object types
        self.pedestrians = [o for o in self.dynamic_objects if o.obj_class == 'person']
        self.vehicles = [o for o in self.dynamic_objects 
                        if o.obj_class in ['car', 'truck', 'cycle']]
        
        print(f"\nObjects detected:")
        print(f"  Pedestrians: {len(self.pedestrians)}")
        print(f"  Vehicles: {len(self.vehicles)}")
    
    def run_audit(self, max_distance: float = 50.0) -> List[VisibilityResult]:
        """
        Run LOS audit for all pedestrian-vehicle pairs.
        
        Args:
            max_distance: Maximum distance to consider (meters)
        """
        print("\n" + "=" * 70)
        print("RUNNING VISIBILITY ANALYSIS")
        print("=" * 70)
        
        results = []
        
        for ped in tqdm(self.pedestrians, desc="Analyzing pedestrians"):
            # Build dynamic checker excluding current pedestrian
            exclude_ids = {ped.obj_id}
            dynamic_checker = DynamicOccupancyChecker(
                self.dynamic_objects, exclude_ids
            )
            
            for vehicle in self.vehicles:
                # Skip if too far
                dist = np.linalg.norm(vehicle.centroid[:2] - ped.centroid[:2])
                if dist > max_distance:
                    continue
                
                # Exclude target vehicle from occlusion check
                dynamic_checker.exclude_ids.add(vehicle.obj_id)
                
                result = compute_visibility(
                    ped, vehicle,
                    self.static_voxels,
                    dynamic_checker,
                    self.scale
                )
                results.append(result)
                
                dynamic_checker.exclude_ids.remove(vehicle.obj_id)
        
        return results
    
    def generate_report(self, results: List[VisibilityResult]) -> Dict:
        """Generate summary report."""
        print("\n" + "=" * 70)
        print("LOS AUDIT REPORT")
        print("=" * 70)
        
        total_pairs = len(results)
        visible_pairs = sum(1 for r in results if r.is_visible)
        occluded_pairs = total_pairs - visible_pairs
        
        print(f"\nTotal pedestrian-vehicle pairs: {total_pairs}")
        print(f"  Visible (>60% rays clear): {visible_pairs}")
        print(f"  Occluded: {occluded_pairs}")
        
        avg_visibility = 0.0
        if total_pairs > 0:
            avg_visibility = np.mean([r.visibility_score for r in results])
            print(f"  Average visibility score: {avg_visibility:.2%}")
        
        # Occluder breakdown
        occluder_stats = defaultdict(int)
        for r in results:
            if not r.is_visible and r.primary_occluder:
                occluder_stats[r.primary_occluder] += 1
        
        if occluder_stats and occluded_pairs > 0:
            print("\nOcclusion causes:")
            for occluder, count in sorted(occluder_stats.items(), 
                                          key=lambda x: -x[1]):
                pct = count / occluded_pairs * 100
                print(f"  {occluder}: {count} ({pct:.1f}%)")
        
        # Critical occlusions (close range, low visibility)
        critical = [r for r in results 
                   if r.distance < 15.0 and r.visibility_score < 0.4]
        
        if critical:
            print(f"\n⚠️  CRITICAL OCCLUSIONS ({len(critical)}):")
            for r in sorted(critical, key=lambda x: x.visibility_score)[:5]:
                print(f"  {r.pedestrian_id} → {r.vehicle_id}: "
                      f"{r.visibility_score:.0%} visible, "
                      f"{r.distance:.1f}m, occluder: {r.primary_occluder}")
        
        # Build object positions dict (for visualization)
        object_positions = {
            'pedestrians': {},
            'vehicles': {}
        }
        
        for ped in self.pedestrians:
            object_positions['pedestrians'][ped.obj_id] = {
                'centroid': ped.centroid.tolist(),
                'heading': float(ped.heading),
                'class': ped.obj_class
            }
        
        for veh in self.vehicles:
            # Compute bounding box from points
            bbox_min = veh.points.min(axis=0)
            bbox_max = veh.points.max(axis=0)
            bbox_size = (bbox_max - bbox_min).tolist()
            
            object_positions['vehicles'][veh.obj_id] = {
                'centroid': veh.centroid.tolist(),
                'heading': float(veh.heading),
                'class': veh.obj_class,
                'bbox_size': bbox_size
            }
        
        # Build report dict
        report = {
            'summary': {
                'total_pairs': total_pairs,
                'visible_pairs': visible_pairs,
                'occluded_pairs': occluded_pairs,
                'avg_visibility_score': float(avg_visibility) if total_pairs > 0 else 0,
                'scale_factor': self.scale,
                'voxel_size_m': VOXEL_SIZE,
                'visibility_threshold': VISIBILITY_THRESHOLD,
                'num_keypoints': len(VEHICLE_KEYPOINTS)
            },
            'object_positions': object_positions,  # NEW: Actual 3D positions from PLY
            'occluder_breakdown': dict(occluder_stats),
            'critical_occlusions': [
                {
                    'pedestrian': r.pedestrian_id,
                    'vehicle': r.vehicle_id,
                    'pedestrian_pos': r.pedestrian_pos.tolist(),  # NEW: Include position
                    'vehicle_pos': r.vehicle_pos.tolist(),        # NEW: Include position
                    'visibility_score': r.visibility_score,
                    'distance': r.distance,
                    'primary_occluder': r.primary_occluder
                }
                for r in critical
            ],
            'all_results': [
                {
                    'pedestrian': r.pedestrian_id,
                    'vehicle': r.vehicle_id,
                    'pedestrian_pos': r.pedestrian_pos.tolist(),  # NEW: Include position
                    'vehicle_pos': r.vehicle_pos.tolist(),        # NEW: Include position
                    'visibility_score': r.visibility_score,
                    'is_visible': r.is_visible,
                    'distance': r.distance,
                    'primary_occluder': r.primary_occluder,
                    'ray_results': [
                        {
                            'keypoint': rr.keypoint_name,
                            'target_point': rr.target_point.tolist(),  # NEW: Keypoint position
                            'is_clear': rr.is_clear,
                            'occluder': rr.occlusion.occluder_type if rr.occlusion else None
                        }
                        for rr in r.ray_results
                    ]
                }
                for r in results
            ]
        }
        
        return report
    
    def visualize_rays(self, results: List[VisibilityResult], 
                       output_path: str = "los_visualization.ply"):
        """
        Create visualization of rays for debugging.
        """
        print(f"\nGenerating visualization: {output_path}")
        
        all_points = [self.static_points.copy()]
        all_colors = [np.ones((len(self.static_points), 3)) * 0.5]  # Gray for static
        
        # Add dynamic objects
        for obj in self.dynamic_objects:
            all_points.append(obj.points)
            if obj.obj_class == 'person':
                color = [0.0, 0.0, 1.0]  # Blue
            elif obj.obj_class == 'car':
                color = [1.0, 0.0, 0.0]  # Red
            elif obj.obj_class == 'truck':
                color = [0.0, 1.0, 0.0]  # Green
            else:
                color = [1.0, 0.0, 1.0]  # Pink
            all_colors.append(np.tile(color, (len(obj.points), 1)))
        
        # Add rays
        for result in results[:50]:  # Limit to first 50 pairs
            eye_height = PERSON_EYE_HEIGHT * self.scale
            ped = next(p for p in self.pedestrians if p.obj_id == result.pedestrian_id)
            ground_z = ped.centroid[2] - (REFERENCE_PERSON_HEIGHT * self.scale) / 2
            eye_point = np.array([ped.centroid[0], ped.centroid[1], ground_z + eye_height])
            
            for rr in result.ray_results:
                # Sample points along ray
                direction = rr.target_point - eye_point
                distance = np.linalg.norm(direction)
                num_samples = int(distance / 0.5)
                
                ray_points = []
                for i in range(num_samples):
                    t = i / max(num_samples - 1, 1)
                    ray_points.append(eye_point + t * direction)
                
                if ray_points:
                    ray_points = np.array(ray_points)
                    all_points.append(ray_points)
                    
                    # Green for clear, red for blocked
                    if rr.is_clear:
                        ray_color = [0.0, 1.0, 0.0]
                    else:
                        ray_color = [1.0, 0.0, 0.0]
                    all_colors.append(np.tile(ray_color, (len(ray_points), 1)))
        
        # Combine and save
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"  Saved {len(combined_points):,} points")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LOS Audit System")
    parser.add_argument('--scene', type=str, default='scene_with_objects.ply',
                       help='Path to scene PLY file')
    parser.add_argument('--output', type=str, default='los_audit_report.json',
                       help='Output report path')
    parser.add_argument('--max-distance', type=float, default=50.0,
                       help='Maximum distance for analysis (meters)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization PLY')
    args = parser.parse_args()
    
    # Run audit
    auditor = LOSAuditor(args.scene)
    results = auditor.run_audit(max_distance=args.max_distance)
    report = auditor.generate_report(results)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")
    
    # Optional visualization
    if args.visualize:
        vis_path = args.output.replace('.json', '_visualization.ply')
        auditor.visualize_rays(results, vis_path)
    
    print("\n" + "=" * 70)
    print("LOS AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
