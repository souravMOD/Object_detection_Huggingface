"""
Typed configuration for object detection pipelines.

Provides a ``DetectionConfig`` dataclass that validates all settings
at construction time.  Configs can be loaded from YAML files, from
plain dicts, or built entirely in code — making the pipeline truly
plug-and-play.

Environment variables prefixed with ``HF_DET_`` override any YAML /
dict value.  For example, ``HF_DET_LEARNING_RATE=1e-4`` overrides the
``learning_rate`` field.

Usage::

    from src.config import DetectionConfig

    # From YAML
    cfg = DetectionConfig.from_yaml("config.yaml")

    # From dict
    cfg = DetectionConfig.from_dict({"model_checkpoint": "facebook/detr-resnet-50", ...})

    # Programmatic
    cfg = DetectionConfig(model_checkpoint="facebook/detr-resnet-50")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DetectionConfig:
    """Validated configuration for the object detection pipeline."""

    # ── Model ──────────────────────────────────────────────────────────
    model_checkpoint: str = "PekingU/rtdetr_v2_r18vd"
    image_size: int = 480

    # ── Dataset paths ──────────────────────────────────────────────────
    train_images_dir: str = "datasets/train"
    train_annotations: str = "datasets/train.json"
    val_images_dir: str = "datasets/val"
    val_annotations: str = "datasets/val.json"

    # ── Training hyperparameters ───────────────────────────────────────
    num_epochs: int = 40
    learning_rate: float = 5e-5
    batch_size: int = 8
    early_stopping_patience: int = 3
    map_threshold: float = 0.01

    # ── Trainer settings ───────────────────────────────────────────────
    metric_for_best_model: Optional[str] = "eval_loss"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    remove_unused_columns: bool = False
    report_to: str = "tensorboard"

    # ── Logging & checkpoints ──────────────────────────────────────────
    log_dir: str = "logs/"
    save_dir: str = "checkpoints/"

    # ── MLOps (optional) ───────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "object_detection"
    wandb_run_name: Optional[str] = None
    use_prometheus: bool = False
    prometheus_port: int = 8000

    # ── Inference ──────────────────────────────────────────────────────
    score_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._apply_env_overrides()
        self._validate()

    # ── Constructors ───────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> DetectionConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectionConfig:
        """Build a config from a plain dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    # ── Serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Export config as a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)

    # ── Internal helpers ───────────────────────────────────────────────

    def _apply_env_overrides(self) -> None:
        """Override fields from environment variables (``HF_DET_<FIELD>``)."""
        for f in fields(self):
            env_key = f"HF_DET_{f.name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                setattr(self, f.name, self._cast(env_val, f.type))

    @staticmethod
    def _cast(value: str, type_hint: str) -> Any:
        """Best-effort cast of an env-var string to the target type."""
        if "bool" in str(type_hint):
            return value.lower() in ("1", "true", "yes")
        if "int" in str(type_hint):
            return int(value)
        if "float" in str(type_hint):
            return float(value)
        return value

    def _validate(self) -> None:
        """Run basic sanity checks on the configuration."""
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not self.model_checkpoint:
            raise ValueError("model_checkpoint must not be empty")
        if self.save_total_limit < 1:
            raise ValueError(f"save_total_limit must be >= 1, got {self.save_total_limit}")
        if self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}")
