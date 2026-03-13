"""Tests for data processing utilities."""

from __future__ import annotations

import json

import albumentations as A
import numpy as np
import pytest

from src.data_processing import get_transforms, load_config


class TestLoadConfig:
    def test_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("batch_size: 16\nnum_epochs: 5\n")
        cfg = load_config(str(cfg_file))
        assert cfg["batch_size"] == 16
        assert cfg["num_epochs"] == 5

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestGetTransforms:
    def test_train_returns_compose(self):
        t = get_transforms(train=True)
        assert isinstance(t, A.Compose)

    def test_val_returns_compose(self):
        t = get_transforms(train=False)
        assert isinstance(t, A.Compose)

    def test_train_transforms_apply(self):
        t = get_transforms(train=True)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = t(image=image, bboxes=[[10, 10, 30, 30]], category=[0])
        assert "image" in result
        assert result["image"].shape[2] == 3

    def test_val_transforms_preserve_image(self):
        t = get_transforms(train=False)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = t(image=image, bboxes=[[10, 10, 30, 30]], category=[0])
        np.testing.assert_array_equal(result["image"], image)


class TestCustomCOCODataset:
    """Tests that require building a COCO annotation fixture."""

    @pytest.fixture()
    def coco_fixture(self, tmp_path):
        """Create a minimal COCO dataset on disk."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        # Create a tiny 10x10 red image
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        img.save(images_dir / "img_001.jpg")

        annotations = {
            "images": [{"id": 1, "file_name": "img_001.jpg", "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0}
            ],
            "categories": [{"id": 0, "name": "object"}],
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations))
        return str(images_dir), str(ann_file)

    def test_dataset_length(self, coco_fixture):
        from unittest.mock import MagicMock
        from src.data_processing import CustomCOCODataset

        images_dir, ann_file = coco_fixture
        processor = MagicMock()
        ds = CustomCOCODataset(images_dir, ann_file, image_processor=processor)
        assert len(ds) == 1

    def test_categories_populated(self, coco_fixture):
        from unittest.mock import MagicMock
        from src.data_processing import CustomCOCODataset

        images_dir, ann_file = coco_fixture
        processor = MagicMock()
        ds = CustomCOCODataset(images_dir, ann_file, image_processor=processor)
        assert ds.categories == {0: "object"}

    def test_missing_images_dir_raises(self, tmp_path):
        from unittest.mock import MagicMock
        from src.data_processing import CustomCOCODataset

        ann_file = tmp_path / "ann.json"
        ann_file.write_text("{}")
        with pytest.raises(FileNotFoundError, match="Images directory"):
            CustomCOCODataset("/nonexistent", str(ann_file), image_processor=MagicMock())

    def test_missing_annotation_file_raises(self, tmp_path):
        from unittest.mock import MagicMock
        from src.data_processing import CustomCOCODataset

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Annotation file"):
            CustomCOCODataset(str(images_dir), "/nonexistent.json", image_processor=MagicMock())

    def test_invalid_coco_format_raises(self, tmp_path):
        from unittest.mock import MagicMock
        from src.data_processing import CustomCOCODataset

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ann_file = tmp_path / "bad.json"
        ann_file.write_text(json.dumps({"images": [], "annotations": []}))
        with pytest.raises(ValueError, match="missing required COCO keys"):
            CustomCOCODataset(str(images_dir), str(ann_file), image_processor=MagicMock())
