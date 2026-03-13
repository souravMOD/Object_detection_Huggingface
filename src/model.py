"""
Model utilities for object detection.

Provides helpers to construct a Hugging Face
``AutoModelForObjectDetection`` with correct label mappings and
adapted classification heads for a custom dataset.

Example::

    from src.config import DetectionConfig
    from src.data_processing import load_datasets
    from src.model import get_label_mappings, load_model, adapt_model_for_custom_classes

    cfg = DetectionConfig.from_yaml("config.yaml")
    train_ds, _ = load_datasets(cfg)
    id2label, label2id = get_label_mappings(train_ds)
    model = load_model(cfg, id2label, label2id)
    model = adapt_model_for_custom_classes(model, len(id2label))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForObjectDetection

from .config import DetectionConfig

logger = logging.getLogger(__name__)


def get_label_mappings(dataset: Any) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Derive ``id2label`` and ``label2id`` mappings from a dataset.

    The dataset must expose a ``categories`` attribute mapping integer
    IDs to category names (as ``CustomCOCODataset`` does).

    Args:
        dataset: An object with a ``categories`` dict attribute.

    Returns:
        A tuple ``(id2label, label2id)``.

    Raises:
        AttributeError: If the dataset has no ``categories`` attribute.
    """
    if not hasattr(dataset, "categories"):
        raise AttributeError(
            f"Dataset of type {type(dataset).__name__} has no 'categories' attribute. "
            "Provide a dataset with a categories mapping (e.g. CustomCOCODataset)."
        )
    id2label: Dict[int, str] = dataset.categories
    if not id2label:
        raise ValueError("Dataset categories mapping is empty — cannot build label maps.")
    label2id: Dict[str, int] = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_model(
    cfg: DetectionConfig | Dict[str, Any],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> Any:
    """Load a pretrained object detection model configured for custom labels.

    Args:
        cfg: A ``DetectionConfig`` or dict containing ``model_checkpoint``.
        id2label: Mapping from integer IDs to label strings.
        label2id: Mapping from label strings to integer IDs.

    Returns:
        An ``AutoModelForObjectDetection`` instance.
    """
    checkpoint = cfg.model_checkpoint if isinstance(cfg, DetectionConfig) else cfg["model_checkpoint"]
    logger.info("Loading pretrained model from %s", checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def adapt_model_for_custom_classes(model: Any, num_custom_classes: int) -> Any:
    """Replace classification heads to match the number of target classes.

    Args:
        model: A Hugging Face object detection model.
        num_custom_classes: Number of classes in your dataset.

    Returns:
        The modified model (in-place, also returned for convenience).

    Raises:
        ValueError: If ``num_custom_classes`` is not positive.
    """
    if num_custom_classes <= 0:
        raise ValueError(f"num_custom_classes must be positive, got {num_custom_classes}")

    if hasattr(model, "class_embed"):
        for i in range(len(model.class_embed)):
            hidden_dim = model.class_embed[i].in_features
            model.class_embed[i] = torch.nn.Linear(hidden_dim, num_custom_classes)
        logger.info(
            "Adapted %d classification heads to %d classes", len(model.class_embed), num_custom_classes
        )

    model.config.num_labels = num_custom_classes
    if hasattr(model.config, "num_classes"):
        model.config.num_classes = num_custom_classes
    return model
