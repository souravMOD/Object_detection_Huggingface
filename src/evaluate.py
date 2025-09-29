"""
Evaluation script for object detection models.

Loads a trained model (optionally from a checkpoint directory) and
computes metrics on the validation set using the same configuration
file used for training. Relies on the ``MAPEvaluator`` defined in
``train.py`` to report mAP metrics.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Any, Dict

import torch
from transformers import TrainingArguments, Trainer, AutoModelForObjectDetection

from .data_processing import load_config, load_datasets, get_image_processor
from .model import get_label_mappings
from .train import MAPEvaluator, collate_fn


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(cfg: Dict[str, Any], checkpoint_path: str | None = None) -> Dict[str, float]:
    """Evaluate a model on the validation dataset defined in cfg.

    Args:
        cfg: Configuration dictionary.
        checkpoint_path: Optional path to a directory containing a fine‑tuned model.
            If None, the model specified by ``cfg['model_checkpoint']`` will be used.

    Returns:
        A dictionary of computed metrics.
    """
    # Load datasets (train is used here only for label mapping)
    train_dataset, val_dataset = load_datasets(cfg)
    id2label, label2id = get_label_mappings(train_dataset)

    # Load model from checkpoint or config
    model_checkpoint = checkpoint_path or cfg["model_checkpoint"]
    logger.info("Loading model from %s", model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    # Instantiate image processor for metric computation
    image_processor = get_image_processor(cfg["model_checkpoint"], cfg["image_size"])
    metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=cfg.get("map_threshold", 0.01), id2label=id2label)

    # Minimal evaluation arguments
    eval_args = TrainingArguments(
        output_dir="eval_tmp/",
        per_device_eval_batch_size=cfg.get("batch_size", 4),
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    logger.info("Running evaluation on %d samples", len(val_dataset))
    metrics = trainer.evaluate()
    return metrics


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an object detection model.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a directory containing a fine‑tuned model."
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    metrics = evaluate(cfg, checkpoint_path=args.checkpoint)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()