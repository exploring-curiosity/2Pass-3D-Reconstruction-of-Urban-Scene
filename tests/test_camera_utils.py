"""Tests for camera utilities"""

import numpy as np
import pytest
import sys
from pathlib import Path
import json
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.camera_utils import (
    CameraParams,
    CameraSet,
    project_points,
    unproject_points,
    is_in_view,
    compute_camera_frustum,
)


class TestCameraParams:
    """Tests for CameraParams dataclass"""

    def test_create_camera_params(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 5.0])
        cam = CameraParams(
            K=K, R=R, t=t, width=640, height=480, camera_id="cam1"
        )
        assert cam.camera_id == "cam1"
        assert cam.width == 640
        assert cam.height == 480

    def test_to_dict(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 5.0])
        cam = CameraParams(
            K=K, R=R, t=t, width=640, height=480, camera_id="cam1"
        )
        d = cam.to_dict()
        assert isinstance(d, dict)
        assert d["camera_id"] == "cam1"
        assert d["width"] == 640
        assert np.allclose(d["K"], K)

    def test_from_dict(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 5.0])
        cam = CameraParams(
            K=K, R=R, t=t, width=640, height=480, camera_id="cam1"
        )
        d = cam.to_dict()
        cam2 = CameraParams.from_dict(d)
        assert cam2.camera_id == cam.camera_id
        assert cam2.width == cam.width
        assert np.allclose(cam2.K, cam.K)
        assert np.allclose(cam2.R, cam.R)
        assert np.allclose(cam2.t, cam.t)

    def test_get_projection_matrix(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 5.0])
        cam = CameraParams(K=K, R=R, t=t, width=640, height=480, camera_id="cam1")
        P = cam.get_projection_matrix()
        assert P.shape == (3, 4)

    def test_get_camera_center(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        R = np.eye(3)
        t = np.array([0.0, 0.0, 5.0])
        cam = CameraParams(K=K, R=R, t=t, width=640, height=480, camera_id="cam1")
        center = cam.get_camera_center()
        # Camera center = -R.T @ t = -I @ [0,0,5] = [0,0,-5]
        assert np.allclose(center, [0.0, 0.0, -5.0])


class TestCameraSet:
    """Tests for CameraSet class"""

    def test_add_and_get_camera(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam1 = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                            width=640, height=480, camera_id="cam1")
        cam2 = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 10.0]),
                            width=640, height=480, camera_id="cam2")

        camera_set = CameraSet()
        camera_set.add_camera(cam1)
        camera_set.add_camera(cam2)

        assert len(camera_set) == 2
        assert camera_set.get_camera("cam1") is not None
        assert camera_set.get_camera("cam2") is not None
        assert camera_set.get_camera("nonexistent") is None

    def test_iterate_cameras(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        camera_set = CameraSet()
        for i in range(3):
            cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, float(i + 1)]),
                               width=640, height=480, camera_id=f"cam{i}")
            camera_set.add_camera(cam)

        cameras = list(camera_set)
        assert len(cameras) == 3

    def test_save_and_load(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        camera_set = CameraSet()
        for i in range(2):
            cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, float(i + 1)]),
                               width=640, height=480, camera_id=f"cam{i}")
            camera_set.add_camera(cam)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            camera_set.save(filepath)
            loaded = CameraSet.load(filepath)
            assert len(loaded) == 2
            for cam_id in ["cam0", "cam1"]:
                orig = camera_set.get_camera(cam_id)
                loaded_cam = loaded.get_camera(cam_id)
                assert np.allclose(orig.K, loaded_cam.K)
                assert np.allclose(orig.t, loaded_cam.t)
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestProjectPoints:
    """Tests for project_points"""

    def test_simple_projection(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        points_3d = np.array([[0.0, 0.0, 3.0]])
        points_2d, depth = project_points(points_3d, cam)
        assert points_2d.shape == (1, 2)
        assert len(depth) == 1
        assert depth[0] > 0

    def test_batch_projection(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        points_3d = np.array([
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 4.0],
            [-1.0, -1.0, 6.0],
        ])
        points_2d, depth = project_points(points_3d, cam)
        assert points_2d.shape == (3, 2)
        assert len(depth) == 3


class TestUnprojectPoints:
    """Tests for unproject_points"""

    def test_simple_unprojection(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        points_2d = np.array([[320.0, 240.0]])  # Center of image
        depth = np.array([5.0])
        points_3d = unproject_points(points_2d, depth, cam)
        assert points_3d.shape == (1, 3)
        # Center of image should project to (0, 0, 5-5=0) in world coords
        assert np.isclose(points_3d[0, 2], 0.0, atol=0.1)

    def test_roundtrip(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        points_3d_original = np.array([[1.0, 2.0, 3.0]])
        points_2d, depth = project_points(points_3d_original, cam)
        points_3d_recovered = unproject_points(points_2d, depth, cam)
        assert np.allclose(points_3d_original, points_3d_recovered, atol=0.01)


class TestIsInView:
    """Tests for is_in_view"""

    def test_point_in_view(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        points_3d = np.array([[0.0, 0.0, 3.0]])
        mask = is_in_view(points_3d, cam, margin=10)
        assert bool(mask[0]) is True

    def test_point_out_of_view(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        # Point far to the right, outside image bounds
        points_3d = np.array([[10.0, 0.0, 3.0]])
        mask = is_in_view(points_3d, cam, margin=10)
        assert bool(mask[0]) is False

    def test_point_behind_camera(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        # Camera center is at [0, 0, -5], looking +z direction.
        # Point at [0, 0, 10] is in front of the camera (z > camera_z), so it IS in view.
        points_3d = np.array([[0.0, 0.0, 10.0]])
        mask = is_in_view(points_3d, cam, margin=10)
        assert bool(mask[0]) is True

    def test_point_behind_camera_actual(self):
        """Test a point truly behind the camera"""
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        # Camera center is at [0, 0, -5], looking +z.
        # Point at [0, 0, -10] is behind the camera (z < camera_z), so NOT in view.
        points_3d = np.array([[0.0, 0.0, -10.0]])
        mask = is_in_view(points_3d, cam, margin=10)
        assert bool(mask[0]) is False


class TestComputeCameraFrustum:
    """Tests for compute_camera_frustum"""

    def test_frustum_corners(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        corners = compute_camera_frustum(cam, depth_range=(1.0, 10.0))
        assert corners.shape == (8, 3)

    def test_frustum_near_far(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        cam = CameraParams(K=K, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
                           width=640, height=480, camera_id="cam1")
        corners = compute_camera_frustum(cam, depth_range=(1.0, 10.0))
        # First 4 corners should be near plane (z closer to camera)
        # Last 4 corners should be far plane (z further from camera)
        near_z = corners[:4, 2]
        far_z = corners[4:, 2]
        assert np.all(near_z < far_z)
