"""
Data processing utilities for object detection.

This module contains helpers to load a configuration from a YAML file,
define augmentation pipelines for training and validation, instantiate
the Hugging Face image processor, and build PyTorch datasets from
COCO‑style annotations. It is deliberately free of side effects on
import: configuration, transforms, and datasets are all created via
function calls rather than at module import time.

To use this module you typically call ``load_config`` to read the
configuration, followed by ``load_datasets`` to create your
``train_dataset`` and ``val_dataset``. Each dataset instance exposes
its category mapping through the ``categories`` attribute which you
can pass to the model loader to correctly configure the label space.

Example:

    >>> from data_processing import load_config, load_datasets
    >>> cfg = load_config("config.yaml")
    >>> train_ds, val_ds = load_datasets(cfg)
    >>> id2label = train_ds.categories

Note: All heavy operations (I/O, augmentation definitions) happen
inside functions so importing this module is cheap.
"""

from __future__ import annotations

import os
import json
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file cannot be parsed as YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(train: bool = True) -> A.Compose:
    """
    Construct an augmentation pipeline for training or validation.

    Albumentations is used here to perform common augmentations. When
    `train` is ``True`` a set of random geometric and photometric
    augmentations is applied; otherwise only an identity transform is
    used. Both cases include bounding box handling appropriate for
    COCO‐style annotations.

    Args:
        train: Whether to create the training pipeline (``True``) or
            validation pipeline (``False``).

    Returns:
        An ``albumentations.Compose`` object configured for bounding
        boxes.
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


def get_image_processor(model_checkpoint: str, image_size: int) -> AutoImageProcessor:
    """
    Instantiate the Hugging Face image processor with the desired size.

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


class CustomCOCODataset(Dataset):
    """
    PyTorch dataset for COCO‑style object detection tasks.

    This dataset reads annotations from a JSON file following the COCO
    format. During iteration it returns dicts compatible with
    Hugging Face transformers, including pixel values and labels. No
    heavy processing happens at initialization; images and
    annotations are loaded on demand.

    Attributes:
        categories: Mapping of category id to category name derived
            from the annotation file. Useful for deriving id2label
            mappings for the model.
    """

    def __init__(
        self,
        images_dir: str,
        annotation_file: str,
        image_processor: AutoImageProcessor,
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        self.image_processor = image_processor

        # Load annotations lazily. The file is expected to be small
        # enough to fit in memory; if very large consider streaming.
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_id_to_filename = {img["id"]: img["file_name"] for img in self.coco_data["images"]}
        self.annotations = self.coco_data["annotations"]
        self.categories: Dict[int, str] = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}

    def __len__(self) -> int:
        return len(self.coco_data["images"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.coco_data["images"][idx]["id"]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, image_filename)

        # Load and convert image to numpy array
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = np.array(image)

        # Filter annotations for this image
        annotations_for_image = [ann for ann in self.annotations if ann["image_id"] == image_id]

        # Prepare bounding boxes and category ids
        boxes: List[List[float]] = []
        categories: List[int] = []
        for ann in annotations_for_image:
            x_min, y_min, w, h = ann["bbox"]
            # Skip invalid boxes
            if w > 0 and h > 0:
                boxes.append([x_min, y_min, w, h])
                categories.append(ann["category_id"])

        # Apply augmentations (Albumentations expects 'bboxes' and 'category' keys)
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=categories)
            image = transformed["image"]
            boxes = list(transformed["bboxes"])
            categories = list(transformed["category"])

        # Build annotations dict for the processor
        formatted_annotations = {"image_id": image_id, "annotations": []}
        for cat, bbox in zip(categories, boxes):
            formatted_annotations["annotations"].append(
                {
                    "category_id": cat,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                }
            )

        # Use the image processor to create model inputs. The processor
        # will pad images and encode labels appropriately. We remove
        # batch dimension since Trainer will batch them later.
        result = self.image_processor(images=image, annotations=formatted_annotations, return_tensors="pt")
        return {k: v[0] for k, v in result.items()}


def load_datasets(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """
    Instantiate training and validation datasets based on a configuration.

    The configuration dictionary must contain keys ``model_checkpoint`` and
    ``image_size`` as well as dataset paths: ``train_images_dir``,
    ``train_annotations``, ``val_images_dir`` and ``val_annotations``.

    Args:
        cfg: Configuration dictionary.

    Returns:
        A tuple ``(train_dataset, val_dataset)``.
    """
    processor = get_image_processor(cfg["model_checkpoint"], cfg["image_size"])
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = CustomCOCODataset(
        images_dir=cfg["train_images_dir"],
        annotation_file=cfg["train_annotations"],
        image_processor=processor,
        transform=train_transform,
    )
    val_dataset = CustomCOCODataset(
        images_dir=cfg["val_images_dir"],
        annotation_file=cfg["val_annotations"],
        image_processor=processor,
        transform=val_transform,
    )
    return train_dataset, val_dataset