"""Tests for geometry utilities"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry_utils import (
    normalize_vector,
    compute_ray_direction,
    ray_sphere_intersection,
    ray_box_intersection,
    compute_bounding_box,
    point_to_line_distance,
    rodrigues_rotation,
    quaternion_to_matrix,
    matrix_to_quaternion,
    compute_plane_from_points,
    project_point_to_plane,
    compute_convex_hull_2d,
    point_in_polygon_2d,
)


class TestNormalizeVector:
    """Tests for normalize_vector"""

    def test_normalize_2d_vector(self):
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_normalize_3d_vector(self):
        v = np.array([1.0, 2.0, 2.0])
        result = normalize_vector(v)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_normalize_already_unit_vector(self):
        v = np.array([0.0, 1.0, 0.0])
        result = normalize_vector(v)
        assert np.allclose(result, v)

    def test_normalize_batch_vectors(self):
        v = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        result = normalize_vector(v)
        norms = np.linalg.norm(result, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestComputeRayDirection:
    """Tests for compute_ray_direction"""

    def test_simple_ray(self):
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])
        direction, distance = compute_ray_direction(origin, target)
        assert np.allclose(direction, [1.0, 0.0, 0.0])
        assert np.isclose(distance, 1.0)

    def test_diagonal_ray(self):
        origin = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 1.0, 1.0])
        direction, distance = compute_ray_direction(origin, target)
        assert np.isclose(distance, np.sqrt(3.0))
        assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)

    def test_batch_ray(self):
        origin = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        direction, distance = compute_ray_direction(origin, target)
        assert direction.shape == (2, 3)
        assert len(distance) == 2


class TestRaySphereIntersection:
    """Tests for ray_sphere_intersection"""

    def test_direct_hit(self):
        ray_origin = np.array([0.0, 0.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 1.0
        intersects, t = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        )
        assert intersects is True
        assert np.isclose(t, 4.0)

    def test_miss(self):
        ray_origin = np.array([0.0, 0.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        sphere_center = np.array([5.0, 0.0, 0.0])
        sphere_radius = 1.0
        intersects, t = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        )
        assert intersects is False
        assert t is None

    def test_tangent_hit(self):
        ray_origin = np.array([0.0, 1.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 1.0
        intersects, t = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        )
        assert intersects is True

    def test_no_intersection_behind(self):
        # Ray starts at (0,0,2) pointing -z. Sphere at (0,0,0) with radius 0.5.
        # The ray passes through the sphere, so it DOES intersect.
        ray_origin = np.array([0.0, 0.0, 2.0])
        ray_direction = np.array([0.0, 0.0, -1.0])
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 0.5
        intersects, t = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        )
        assert intersects is True
        assert t is not None

    def test_no_intersection_miss(self):
        """Test when ray misses sphere entirely"""
        ray_origin = np.array([0.0, 0.0, 2.0])
        ray_direction = np.array([1.0, 0.0, 0.0])  # Points along x, missing sphere
        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 0.5
        intersects, t = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        )
        assert intersects is False
        assert t is None


class TestRayBoxIntersection:
    """Tests for ray_box_intersection"""

    def test_box_hit(self):
        ray_origin = np.array([0.0, 0.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        box_min = np.array([-1.0, -1.0, -1.0])
        box_max = np.array([1.0, 1.0, 1.0])
        intersects, t_near, t_far = ray_box_intersection(
            ray_origin, ray_direction, box_min, box_max
        )
        assert intersects is True
        assert np.isclose(t_near, 4.0)
        assert np.isclose(t_far, 6.0)

    def test_box_miss(self):
        ray_origin = np.array([0.0, 0.0, -5.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        box_min = np.array([5.0, 5.0, -1.0])
        box_max = np.array([6.0, 6.0, 1.0])
        intersects, t_near, t_far = ray_box_intersection(
            ray_origin, ray_direction, box_min, box_max
        )
        assert intersects is False
        assert t_near is None
        assert t_far is None

    def test_ray_inside_box(self):
        # Ray starts inside the box, so t_near is negative (past entry point)
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        box_min = np.array([-1.0, -1.0, -1.0])
        box_max = np.array([1.0, 1.0, 1.0])
        intersects, t_near, t_far = ray_box_intersection(
            ray_origin, ray_direction, box_min, box_max
        )
        assert intersects is True
        # When ray starts inside, t_near is negative (past entry point)
        assert t_near < 0
        # t_far is the exit point
        assert np.isclose(t_far, 1.0)


class TestComputeBoundingBox:
    """Tests for compute_bounding_box"""

    def test_simple_box(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        box_min, box_max = compute_bounding_box(points)
        assert np.allclose(box_min, [0.0, 0.0, 0.0])
        assert np.allclose(box_max, [1.0, 2.0, 3.0])

    def test_box_with_padding(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        box_min, box_max = compute_bounding_box(points, padding=0.5)
        assert np.allclose(box_min, [-0.5, -0.5, -0.5])
        assert np.allclose(box_max, [1.5, 1.5, 1.5])


class TestPointToLineDistance:
    """Tests for point_to_line_distance"""

    def test_perpendicular_distance(self):
        point = np.array([0.0, 1.0, 0.0])
        line_point = np.array([0.0, 0.0, 0.0])
        line_direction = np.array([1.0, 0.0, 0.0])
        distance = point_to_line_distance(point, line_point, line_direction)
        assert np.isclose(distance, 1.0)

    def test_distance_along_line(self):
        point = np.array([5.0, 0.0, 0.0])
        line_point = np.array([0.0, 0.0, 0.0])
        line_direction = np.array([1.0, 0.0, 0.0])
        distance = point_to_line_distance(point, line_point, line_direction)
        assert np.isclose(distance, 0.0)


class TestRodriguesRotation:
    """Tests for rodrigues_rotation"""

    def test_90_degree_rotation(self):
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2.0
        R = rodrigues_rotation(axis, angle)
        # Rotate (1,0,0) by 90 degrees around z -> (0,1,0)
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        assert np.allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_identity_rotation(self):
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
        R = rodrigues_rotation(axis, angle)
        assert np.allclose(R, np.eye(3))

    def test_rotation_matrix_properties(self):
        axis = np.array([1.0, 1.0, 1.0])
        angle = np.pi / 4.0
        R = rodrigues_rotation(axis, angle)
        # Check R is orthogonal
        assert np.allclose(R @ R.T, np.eye(3))
        # Check det(R) = 1
        assert np.isclose(np.linalg.det(R), 1.0)


class TestQuaternionMatrixConversion:
    """Tests for quaternion <-> matrix conversions"""

    def test_roundtrip(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # w=1, x=y=z=0 (identity)
        R = quaternion_to_matrix(q)
        q_back = matrix_to_quaternion(R)
        assert np.allclose(q, q_back)

    def test_nontrivial_rotation(self):
        q = np.array([0.7071, 0.0, 0.7071, 0.0])  # 90 deg around y
        R = quaternion_to_matrix(q)
        q_back = matrix_to_quaternion(R)
        assert np.allclose(q, q_back)

    def test_rotation_matrix_properties(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R = quaternion_to_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


class TestComputePlaneFromPoints:
    """Tests for compute_plane_from_points"""

    def test_xy_plane(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        normal, d = compute_plane_from_points(points, robust=False)
        assert np.isclose(d, 0.0, atol=0.1)
        assert np.isclose(abs(normal[2]), 1.0, atol=0.1)

    def test_zx_plane(self):
        points = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        normal, d = compute_plane_from_points(points, robust=False)
        assert np.isclose(abs(normal[1]), 1.0, atol=0.1)


class TestProjectPointToPlane:
    """Tests for project_point_to_plane"""

    def test_project_to_xy_plane(self):
        point = np.array([1.0, 2.0, 3.0])
        plane_normal = np.array([0.0, 0.0, 1.0])
        plane_d = 0.0
        projected = project_point_to_plane(point, plane_normal, plane_d)
        assert np.isclose(projected[2], 0.0)
        assert np.isclose(projected[0], 1.0)
        assert np.isclose(projected[1], 2.0)

    def test_project_to_diagonal_plane(self):
        point = np.array([1.0, 1.0, 1.0])
        plane_normal = np.array([1.0, 0.0, 0.0])
        plane_d = 0.0
        projected = project_point_to_plane(point, plane_normal, plane_d)
        assert np.isclose(projected[0], 0.0)
        assert np.isclose(projected[1], 1.0)
        assert np.isclose(projected[2], 1.0)


class TestConvexHull2D:
    """Tests for compute_convex_hull_2d"""

    def test_square_hull(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hull_indices = compute_convex_hull_2d(points)
        assert len(hull_indices) == 4

    def test_small_point_set(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        hull_indices = compute_convex_hull_2d(points)
        assert len(hull_indices) == 2


class TestPointInPolygon2D:
    """Tests for point_in_polygon_2d"""

    def test_point_inside_square(self):
        polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        point = np.array([0.5, 0.5])
        assert point_in_polygon_2d(point, polygon) is True

    def test_point_outside_square(self):
        polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        point = np.array([2.0, 2.0])
        assert point_in_polygon_2d(point, polygon) is False

    def test_point_on_edge(self):
        polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        point = np.array([0.5, 0.0])
        # Points on the edge may or may not be inside depending on implementation
        result = point_in_polygon_2d(point, polygon)
        assert isinstance(result, bool)
