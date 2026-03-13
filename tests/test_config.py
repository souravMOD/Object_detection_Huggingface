"""Tests for the DetectionConfig dataclass."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from src.config import DetectionConfig


class TestDetectionConfigDefaults:
    """Verify sensible defaults when no arguments are provided."""

    def test_default_model_checkpoint(self):
        cfg = DetectionConfig()
        assert cfg.model_checkpoint == "PekingU/rtdetr_v2_r18vd"

    def test_default_image_size(self):
        cfg = DetectionConfig()
        assert cfg.image_size == 480

    def test_default_hyperparameters(self):
        cfg = DetectionConfig()
        assert cfg.num_epochs == 40
        assert cfg.learning_rate == 5e-5
        assert cfg.batch_size == 8

    def test_default_mlops_disabled(self):
        cfg = DetectionConfig()
        assert cfg.use_wandb is False
        assert cfg.use_prometheus is False


class TestDetectionConfigValidation:
    """Verify that invalid values are caught at construction time."""

    def test_negative_image_size_raises(self):
        with pytest.raises(ValueError, match="image_size"):
            DetectionConfig(image_size=-1)

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            DetectionConfig(batch_size=0)

    def test_negative_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DetectionConfig(learning_rate=-0.001)

    def test_empty_checkpoint_raises(self):
        with pytest.raises(ValueError, match="model_checkpoint"):
            DetectionConfig(model_checkpoint="")

    def test_zero_epochs_raises(self):
        with pytest.raises(ValueError, match="num_epochs"):
            DetectionConfig(num_epochs=0)

    def test_zero_save_total_limit_raises(self):
        with pytest.raises(ValueError, match="save_total_limit"):
            DetectionConfig(save_total_limit=0)


class TestDetectionConfigFromYaml:
    """Test YAML loading."""

    def test_loads_valid_yaml(self, tmp_path):
        yaml_content = {"model_checkpoint": "facebook/detr-resnet-50", "batch_size": 16}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        cfg = DetectionConfig.from_yaml(str(config_file))
        assert cfg.model_checkpoint == "facebook/detr-resnet-50"
        assert cfg.batch_size == 16

    def test_missing_yaml_raises(self):
        with pytest.raises(FileNotFoundError):
            DetectionConfig.from_yaml("/nonexistent/config.yaml")

    def test_ignores_unknown_keys(self, tmp_path):
        yaml_content = {"model_checkpoint": "some-model", "unknown_key": 42}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        cfg = DetectionConfig.from_yaml(str(config_file))
        assert cfg.model_checkpoint == "some-model"
        assert not hasattr(cfg, "unknown_key")


class TestDetectionConfigFromDict:
    """Test dict construction."""

    def test_from_dict_basic(self):
        cfg = DetectionConfig.from_dict({"num_epochs": 5, "batch_size": 2})
        assert cfg.num_epochs == 5
        assert cfg.batch_size == 2

    def test_from_dict_ignores_unknown(self):
        cfg = DetectionConfig.from_dict({"model_checkpoint": "m", "foo": "bar"})
        assert cfg.model_checkpoint == "m"

    def test_to_dict_roundtrip(self):
        cfg = DetectionConfig(num_epochs=7)
        d = cfg.to_dict()
        cfg2 = DetectionConfig.from_dict(d)
        assert cfg2.num_epochs == 7
        assert cfg.to_dict() == cfg2.to_dict()


class TestDetectionConfigEnvOverrides:
    """Test environment variable override mechanism."""

    def test_env_overrides_int(self, monkeypatch):
        monkeypatch.setenv("HF_DET_BATCH_SIZE", "32")
        cfg = DetectionConfig()
        assert cfg.batch_size == 32

    def test_env_overrides_float(self, monkeypatch):
        monkeypatch.setenv("HF_DET_LEARNING_RATE", "0.001")
        cfg = DetectionConfig()
        assert cfg.learning_rate == 0.001

    def test_env_overrides_bool_true(self, monkeypatch):
        monkeypatch.setenv("HF_DET_USE_WANDB", "true")
        cfg = DetectionConfig()
        assert cfg.use_wandb is True

    def test_env_overrides_bool_false(self, monkeypatch):
        monkeypatch.setenv("HF_DET_USE_WANDB", "false")
        cfg = DetectionConfig()
        assert cfg.use_wandb is False

    def test_env_overrides_string(self, monkeypatch):
        monkeypatch.setenv("HF_DET_MODEL_CHECKPOINT", "my-custom-model")
        cfg = DetectionConfig()
        assert cfg.model_checkpoint == "my-custom-model"
