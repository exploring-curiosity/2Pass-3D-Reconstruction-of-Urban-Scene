"""Tests for I/O utilities"""

import numpy as np
import pytest
import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import (
    save_ply,
    load_ply,
    save_json,
    load_json,
    save_npz,
    load_npz,
)


class TestSaveLoadPly:
    """Tests for save_ply and load_ply"""

    def test_save_and_load_ply_binary(self):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ])
            colors = np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
            ])
            save_ply(filepath, points, colors)

            loaded_points, loaded_colors, loaded_normals = load_ply(filepath)
            assert loaded_points.shape == (4, 3)
            assert loaded_colors.shape == (4, 3)
            assert loaded_normals is None
            assert np.allclose(loaded_points, points)
            assert np.allclose(loaded_colors, colors)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_and_load_ply_with_normals(self):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ])
            normals = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ])
            save_ply(filepath, points, normals=normals)

            loaded_points, loaded_colors, loaded_normals = load_ply(filepath)
            assert loaded_points.shape == (2, 3)
            assert loaded_colors is None
            assert loaded_normals is not None
            assert loaded_normals.shape == (2, 3)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_and_load_ply_ascii(self):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            points = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ])
            colors = np.array([
                [128, 128, 128],
                [200, 100, 50],
            ])
            save_ply(filepath, points, colors, ascii_format=True)

            # Verify it's valid ASCII
            with open(filepath, 'r') as f:
                content = f.read()
            assert 'ply' in content
            assert 'element vertex 2' in content

            loaded_points, loaded_colors, _ = load_ply(filepath)
            assert loaded_points.shape == (2, 3)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_ply_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "subdir" / "pointcloud.ply")
            points = np.array([[0.0, 0.0, 0.0]])
            save_ply(filepath, points)
            assert Path(filepath).exists()


class TestSaveLoadJson:
    """Tests for save_json and load_json"""

    def test_save_and_load_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
            save_json(filepath, data)

            loaded = load_json(filepath)
            assert loaded == data
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_json_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "subdir" / "data.json")
            data = {"test": "data"}
            save_json(filepath, data)
            assert Path(filepath).exists()


class TestSaveLoadNpz:
    """Tests for save_npz and load_npz"""

    def test_save_and_load_npz(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            array1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            array2 = np.array([10, 20, 30])
            save_npz(filepath, arr1=array1, arr2=array2)

            loaded = load_npz(filepath)
            assert "arr1" in loaded
            assert "arr2" in loaded
            assert np.allclose(loaded["arr1"], array1)
            assert np.allclose(loaded["arr2"], array2)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_npz_single_array(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            array1 = np.array([1.0, 2.0, 3.0])
            save_npz(filepath, data=array1)

            loaded = load_npz(filepath)
            assert "data" in loaded
            assert np.allclose(loaded["data"], array1)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_npz_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "subdir" / "data.npz")
            array = np.array([1, 2, 3])
            save_npz(filepath, data=array)
            assert Path(filepath).exists()
