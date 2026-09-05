"""Tests for config loader"""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader, load_config


class TestConfigLoader:
    """Tests for ConfigLoader class"""

    def test_load_valid_config(self):
        config_path = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
        loader = ConfigLoader(config_path)
        config = loader.load()
        assert config is not None
        assert "data" in config
        assert "pass1_static" in config
        assert "pass2_dynamic" in config

    def test_load_nonexistent_config(self):
        loader = ConfigLoader("/nonexistent/path/config.yaml")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_get_config_value(self):
        config_path = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
        loader = ConfigLoader(config_path)
        loader.load()
        cameras = loader.get("data.cameras")
        # OmegaConf.select returns an OmegaConf List, check length instead
        assert len(cameras) == 8

    def test_get_default_value(self):
        config_path = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
        loader = ConfigLoader(config_path)
        loader.load()
        value = loader.get("nonexistent.key", "default")
        assert value == "default"

    def test_save_and_reload(self):
        config_path = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
        loader = ConfigLoader(config_path)
        config = loader.load()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            loader.save(output_path)
            assert Path(output_path).exists()

            # Reload and verify
            loader2 = ConfigLoader(output_path)
            config2 = loader2.load()
            assert config2["project_name"] == config["project_name"]
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestLoadConfig:
    """Tests for load_config convenience function"""

    def test_load_config_returns_dict(self):
        config_path = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
        config = load_config(config_path)
        assert isinstance(config, dict)
        assert "data" in config

    def test_load_config_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestConfigValidation:
    """Tests for configuration validation"""

    def test_valid_config_passes_validation(self):
        config_path = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
        loader = ConfigLoader(config_path)
        # Should not raise
        config = loader.load()
        assert config is not None

    def test_missing_video_dir_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({"data": {"cameras": ["cam1"]}}, f)
            filepath = f.name

        try:
            loader = ConfigLoader(filepath)
            # OmegaConf raises ConfigAttributeError when accessing missing keys
            with pytest.raises(Exception):
                loader.load()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_no_cameras_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({"data": {"video_dir": "videos"}}, f)
            filepath = f.name

        try:
            loader = ConfigLoader(filepath)
            # OmegaConf raises ConfigAttributeError when accessing missing keys
            with pytest.raises(Exception):
                loader.load()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_no_passes_enabled_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({
                "data": {"video_dir": "videos", "cameras": ["cam1"]},
                "pass1_static": {"enabled": False},
                "pass2_dynamic": {"enabled": False},
                "hardware": {"device": "cpu"}
            }, f)
            filepath = f.name

        try:
            loader = ConfigLoader(filepath)
            with pytest.raises(ValueError, match="At least one pass must be enabled"):
                loader.load()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_invalid_device_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({
                "data": {"video_dir": "videos", "cameras": ["cam1"]},
                "pass1_static": {"enabled": True},
                "pass2_dynamic": {"enabled": True},
                "hardware": {"device": "invalid_device"}
            }, f)
            filepath = f.name

        try:
            loader = ConfigLoader(filepath)
            with pytest.raises(AssertionError):
                loader.load()
        finally:
            Path(filepath).unlink(missing_ok=True)
