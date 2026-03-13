"""
Evaluation script for object detection models.

Loads a trained model and computes mAP metrics on the validation set.

CLI usage::

    python -m src.evaluate --config config.yaml --checkpoint path/to/checkpoint
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForObjectDetection, Trainer, TrainingArguments

from .config import DetectionConfig
from .data_processing import get_image_processor, load_datasets
from .model import get_label_mappings
from .train import MAPEvaluator, collate_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(
    cfg: DetectionConfig | Dict[str, Any],
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a model on the validation dataset.

    Args:
        cfg: A ``DetectionConfig`` or plain dictionary.
        checkpoint_path: Optional path to a fine-tuned model directory.

    Returns:
        A dictionary of computed metrics.
    """
    if isinstance(cfg, dict):
        cfg = DetectionConfig.from_dict(cfg)

    train_dataset, val_dataset = load_datasets(cfg)
    id2label, label2id = get_label_mappings(train_dataset)

    model_checkpoint = checkpoint_path or cfg.model_checkpoint
    logger.info("Loading model from %s", model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        model_checkpoint, id2label=id2label, label2id=label2id
    )

    image_processor = get_image_processor(cfg.model_checkpoint, cfg.image_size)
    metrics_fn = MAPEvaluator(
        image_processor=image_processor,
        threshold=cfg.map_threshold,
        id2label=id2label,
    )

    eval_args = TrainingArguments(
        output_dir=cfg.save_dir,
        per_device_eval_batch_size=cfg.batch_size,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    logger.info("Running evaluation on %d samples", len(val_dataset))
    return trainer.evaluate()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an object detection model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a fine-tuned model directory.")
    args = parser.parse_args(argv)
    cfg = DetectionConfig.from_yaml(args.config)
    metrics = evaluate(cfg, checkpoint_path=args.checkpoint)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
