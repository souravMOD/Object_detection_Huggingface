"""Tests for model utility functions."""

from __future__ import annotations

import pytest
import torch

from src.model import adapt_model_for_custom_classes, get_label_mappings


class FakeDataset:
    """Minimal stand-in for CustomCOCODataset."""

    def __init__(self, categories):
        self.categories = categories


class TestGetLabelMappings:
    def test_basic_mapping(self):
        ds = FakeDataset({0: "cat", 1: "dog"})
        id2label, label2id = get_label_mappings(ds)
        assert id2label == {0: "cat", 1: "dog"}
        assert label2id == {"cat": 0, "dog": 1}

    def test_missing_categories_attribute_raises(self):
        with pytest.raises(AttributeError, match="categories"):
            get_label_mappings(object())

    def test_empty_categories_raises(self):
        ds = FakeDataset({})
        with pytest.raises(ValueError, match="empty"):
            get_label_mappings(ds)


class TestAdaptModelForCustomClasses:
    def _make_fake_model(self, num_heads=3, hidden_dim=256, original_classes=80):
        """Create a minimal object with class_embed layers."""
        model = type("FakeModel", (), {})()
        model.class_embed = [torch.nn.Linear(hidden_dim, original_classes) for _ in range(num_heads)]
        model.config = type("Config", (), {"num_labels": original_classes, "num_classes": original_classes})()
        return model

    def test_replaces_heads(self):
        model = self._make_fake_model(num_heads=3, hidden_dim=256, original_classes=80)
        adapted = adapt_model_for_custom_classes(model, 5)
        for layer in adapted.class_embed:
            assert layer.out_features == 5
        assert adapted.config.num_labels == 5
        assert adapted.config.num_classes == 5

    def test_returns_same_model(self):
        model = self._make_fake_model()
        adapted = adapt_model_for_custom_classes(model, 10)
        assert adapted is model

    def test_zero_classes_raises(self):
        model = self._make_fake_model()
        with pytest.raises(ValueError, match="positive"):
            adapt_model_for_custom_classes(model, 0)

    def test_negative_classes_raises(self):
        model = self._make_fake_model()
        with pytest.raises(ValueError, match="positive"):
            adapt_model_for_custom_classes(model, -3)

    def test_model_without_class_embed(self):
        """Models without class_embed should still get config updated."""
        model = type("FakeModel", (), {})()
        model.config = type("Config", (), {"num_labels": 80})()
        adapted = adapt_model_for_custom_classes(model, 5)
        assert adapted.config.num_labels == 5
