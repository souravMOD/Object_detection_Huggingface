"""
Model utilities for object detection.

This module provides helper functions to construct a Hugging Face
``AutoModelForObjectDetection`` with the correct label mappings for a
custom dataset. It also includes a convenience function to replace
classification heads to match the number of target classes.

Usage:

    from model import get_label_mappings, load_model, adapt_model_for_custom_classes
    cfg = load_config("config.yaml")
    train_ds, val_ds = load_datasets(cfg)
    id2label, label2id = get_label_mappings(train_ds)
    model = load_model(cfg, id2label, label2id)
    model = adapt_model_for_custom_classes(model, len(id2label))

These functions perform no heavy work at import time. They require
explicit configuration and dataset objects in order to run, keeping
your module side‑effect free and easy to test.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import torch
from transformers import AutoModelForObjectDetection


def get_label_mappings(dataset) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Derive id→label and label→id mappings from a dataset.

    The dataset must expose a ``categories`` attribute mapping integer
    IDs to category names. Many COCO‑style datasets, including the
    ``CustomCOCODataset`` defined in :mod:`data_processing`, provide
    exactly this mapping.

    Args:
        dataset: An object with a ``categories`` attribute that maps
            integer category IDs to string names.

    Returns:
        A tuple ``(id2label, label2id)`` where ``id2label`` maps
        integer IDs to strings and ``label2id`` maps strings back to
        integer IDs.
    """
    id2label: Dict[int, str] = getattr(dataset, "categories", {})
    label2id: Dict[str, int] = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_model(cfg: Dict[str, Any], id2label: Dict[int, str], label2id: Dict[str, int]) -> Any:
    """
    Load a pretrained object detection model configured for custom labels.

    Args:
        cfg: Configuration dictionary containing at least the key
            ``model_checkpoint``.
        id2label: Mapping from integer IDs to label strings.
        label2id: Mapping from label strings to integer IDs.

    Returns:
        An instance of ``AutoModelForObjectDetection`` with custom label
        mappings. The model is not yet modified to match the number of
        classes; call :func:`adapt_model_for_custom_classes` after
        instantiation if needed.
    """
    model = AutoModelForObjectDetection.from_pretrained(
        cfg["model_checkpoint"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def adapt_model_for_custom_classes(model: Any, num_custom_classes: int) -> Any:
    """
    Adjust the model's classification heads to a custom number of classes.

    Some object detection architectures expose multiple classification
    heads (e.g. for auxiliary losses). This function iterates over
    ``model.class_embed`` to replace each linear layer with one whose
    output dimension matches ``num_custom_classes``. It also updates
    the ``model.config`` to reflect the new number of labels.

    Args:
        model: A Hugging Face object detection model instance.
        num_custom_classes: The number of classes in your dataset.

    Returns:
        The modified model instance (modification is in‐place but also
        returned for convenience).
    """
    # Update classification heads if present
    if hasattr(model, "class_embed"):
        for i in range(len(model.class_embed)):
            hidden_dim = model.class_embed[i].in_features
            model.class_embed[i] = torch.nn.Linear(hidden_dim, num_custom_classes)
    # Update configuration fields
    model.config.num_labels = num_custom_classes
    if hasattr(model.config, "num_classes"):
        model.config.num_classes = num_custom_classes
    return model