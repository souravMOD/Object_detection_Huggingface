"""
Data processing utilities for object detection.

Provides helpers to define augmentation pipelines, instantiate the
Hugging Face image processor, and build PyTorch datasets from
COCO-style annotations.  All heavy I/O happens inside function calls —
importing this module is cheap.

Example::

    from src.config import DetectionConfig
    from src.data_processing import load_datasets

    cfg = DetectionConfig.from_yaml("config.yaml")
    train_ds, val_ds = load_datasets(cfg)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .config import DetectionConfig

logger = logging.getLogger(__name__)


# ── Legacy helper (kept for backward compat) ──────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary.

    .. deprecated::
        Prefer ``DetectionConfig.from_yaml()`` for validated configs.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Transforms ────────────────────────────────────────────────────────

def get_transforms(train: bool = True) -> A.Compose:
    """Construct an augmentation pipeline for training or validation.

    Args:
        train: ``True`` for the training pipeline with random augmentations,
            ``False`` for a validation-only identity pipeline.

    Returns:
        An ``albumentations.Compose`` configured for COCO bounding boxes.
    """
    if train:
        transform = A.Compose(
            [
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category"],
                clip=True,
                min_area=25,
                min_width=1,
                min_height=1,
            ),
        )
    else:
        transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category"],
                clip=True,
                min_area=1,
                min_width=1,
                min_height=1,
            ),
        )
    return transform


# ── Image processor ───────────────────────────────────────────────────

def get_image_processor(model_checkpoint: str, image_size: int) -> AutoImageProcessor:
    """Instantiate the Hugging Face image processor with the desired size.

    Args:
        model_checkpoint: Name or path of the pretrained model checkpoint.
        image_size: Target width and height of the processed images.

    Returns:
        A configured ``AutoImageProcessor`` instance.
    """
    return AutoImageProcessor.from_pretrained(
        model_checkpoint,
        size={"width": image_size, "height": image_size},
        use_fast=True,
    )


# ── Dataset ───────────────────────────────────────────────────────────

class CustomCOCODataset(Dataset):
    """PyTorch dataset for COCO-style object detection tasks.

    Reads annotations from a JSON file following the COCO format.
    Images and annotations are loaded on demand (not at init time).

    Attributes:
        categories: Mapping of category id to category name.
    """

    def __init__(
        self,
        images_dir: str,
        annotation_file: str,
        image_processor: AutoImageProcessor,
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        images_path = Path(images_dir)
        ann_path = Path(annotation_file)

        if not images_path.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        self.images_dir = str(images_path)
        self.transform = transform
        self.image_processor = image_processor

        with open(ann_path) as f:
            self.coco_data: Dict[str, Any] = json.load(f)

        required_keys = {"images", "annotations", "categories"}
        missing = required_keys - set(self.coco_data.keys())
        if missing:
            raise ValueError(f"Annotation file missing required COCO keys: {missing}")

        self.image_id_to_filename: Dict[int, str] = {
            img["id"]: img["file_name"] for img in self.coco_data["images"]
        }
        self.annotations: List[Dict[str, Any]] = self.coco_data["annotations"]
        self.categories: Dict[int, str] = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        # Build a fast lookup index: image_id -> list of annotations
        self._ann_index: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            self._ann_index.setdefault(ann["image_id"], []).append(ann)

        logger.info(
            "Loaded dataset: %d images, %d annotations, %d categories from %s",
            len(self.coco_data["images"]),
            len(self.annotations),
            len(self.categories),
            ann_path,
        )

    def __len__(self) -> int:
        return len(self.coco_data["images"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_info = self.coco_data["images"][idx]
        image_id = image_info["id"]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, image_filename)

        try:
            with Image.open(image_path) as img:
                image = np.array(img.convert("RGB"))
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Failed to load image {image_path}: {exc}") from exc

        # Use the pre-built index instead of filtering all annotations
        annotations_for_image = self._ann_index.get(image_id, [])

        boxes: List[List[float]] = []
        categories: List[int] = []
        for ann in annotations_for_image:
            x_min, y_min, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x_min, y_min, w, h])
                categories.append(ann["category_id"])

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=categories)
            image = transformed["image"]
            boxes = list(transformed["bboxes"])
            categories = list(transformed["category"])

        formatted_annotations: Dict[str, Any] = {"image_id": image_id, "annotations": []}
        for cat, bbox in zip(categories, boxes):
            formatted_annotations["annotations"].append(
                {
                    "category_id": cat,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                }
            )

        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )
        return {k: v[0] for k, v in result.items()}


# ── Dataset factory ───────────────────────────────────────────────────

def load_datasets(cfg: DetectionConfig | Dict[str, Any]) -> Tuple[CustomCOCODataset, CustomCOCODataset]:
    """Create training and validation datasets from a config.

    Args:
        cfg: A ``DetectionConfig`` instance **or** a plain dictionary
            (for backward compatibility).

    Returns:
        A tuple ``(train_dataset, val_dataset)``.
    """
    if isinstance(cfg, dict):
        cfg = DetectionConfig.from_dict(cfg)

    processor = get_image_processor(cfg.model_checkpoint, cfg.image_size)
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = CustomCOCODataset(
        images_dir=cfg.train_images_dir,
        annotation_file=cfg.train_annotations,
        image_processor=processor,
        transform=train_transform,
    )
    val_dataset = CustomCOCODataset(
        images_dir=cfg.val_images_dir,
        annotation_file=cfg.val_annotations,
        image_processor=processor,
        transform=val_transform,
    )
    return train_dataset, val_dataset
