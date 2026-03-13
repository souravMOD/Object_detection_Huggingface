"""
Training script for object detection using Hugging Face Transformers.

Wires together the data pipeline, model, optimizer, and metric
computation using the ``transformers.Trainer`` API.

CLI usage::

    python -m src.train --config config.yaml

Programmatic usage::

    from src.config import DetectionConfig
    from src.train import train

    cfg = DetectionConfig.from_yaml("config.yaml")
    train(cfg)
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

try:
    from prometheus_client import Gauge, start_http_server  # type: ignore
    from transformers import TrainerCallback
except ImportError:
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore
    TrainerCallback = object  # type: ignore

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .config import DetectionConfig
from .data_processing import get_image_processor, load_config, load_datasets
from .model import adapt_model_for_custom_classes, get_label_mappings, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Collate ───────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack pixel values and keep labels as a list of dicts."""
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ── Metric evaluator ─────────────────────────────────────────────────

class MAPEvaluator:
    """Compute mean-average-precision (mAP) during evaluation.

    Passed to ``Trainer`` via the ``compute_metrics`` argument.
    """

    def __init__(
        self,
        image_processor: Any,
        threshold: float = 0.0,
        id2label: Optional[Dict[int, str]] = None,
    ):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label or {}

    def _collect_image_sizes(self, targets: List[List[Dict[str, Any]]]) -> List[torch.Tensor]:
        image_sizes: List[torch.Tensor] = []
        for batch in targets:
            sizes = [torch.tensor(t["size"]) for t in batch]
            image_sizes.append(torch.stack(sizes))
        return image_sizes

    def _collect_targets(
        self, targets: List[List[Dict[str, Any]]], image_sizes: List[torch.Tensor]
    ) -> List[Dict[str, Any]]:
        post_processed: List[Dict[str, Any]] = []
        for target_batch, size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, size_batch):
                height, width = size.tolist()
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([width, height, width, height])
                labels = torch.tensor(target["class_labels"])
                post_processed.append({"boxes": boxes, "labels": labels})
        return post_processed

    def _collect_predictions(
        self, predictions: List[Any], image_sizes: List[torch.Tensor]
    ) -> List[Dict[str, Any]]:
        post_processed: List[Dict[str, Any]] = []
        for batch_pred, sizes in zip(predictions, image_sizes):
            _, logits, pred_boxes = batch_pred
            output = type("ModelOutput", (), {})()
            output.logits = torch.tensor(logits)
            output.pred_boxes = torch.tensor(pred_boxes)
            processed = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=sizes
            )
            post_processed.extend(processed)
        return post_processed

    @torch.no_grad()
    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        predictions, targets = eval_preds.predictions, eval_preds.label_ids
        image_sizes = self._collect_image_sizes(targets)
        post_targets = self._collect_targets(targets, image_sizes)
        post_predictions = self._collect_predictions(predictions, image_sizes)
        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_predictions, post_targets)
        metrics = evaluator.compute()

        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_id_int = class_id.item() if hasattr(class_id, "item") else int(class_id)
            class_name = self.id2label.get(class_id_int, f"class_{class_id_int}")
            metrics[f"map_{class_name}"] = class_map.item() if hasattr(class_map, "item") else float(class_map)
            metrics[f"mar_100_{class_name}"] = class_mar.item() if hasattr(class_mar, "item") else float(class_mar)

        return {k: round(float(v), 4) for k, v in metrics.items()}


# ── Training ──────────────────────────────────────────────────────────

def train(cfg: DetectionConfig | Dict[str, Any]) -> None:
    """Execute a full training loop.

    Args:
        cfg: A ``DetectionConfig`` instance or a plain dictionary.
    """
    if isinstance(cfg, dict):
        cfg = DetectionConfig.from_dict(cfg)

    train_dataset, val_dataset = load_datasets(cfg)
    logger.info("Loaded %d training and %d validation samples", len(train_dataset), len(val_dataset))

    id2label, label2id = get_label_mappings(train_dataset)
    model = load_model(cfg, id2label, label2id)
    model = adapt_model_for_custom_classes(model, len(id2label))

    image_processor = get_image_processor(cfg.model_checkpoint, cfg.image_size)
    metrics_fn = MAPEvaluator(
        image_processor=image_processor,
        threshold=cfg.map_threshold,
        id2label=id2label,
    )

    report_to = cfg.report_to
    use_wandb = cfg.use_wandb and wandb is not None
    if use_wandb:
        report_to = "wandb"
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            config=cfg.to_dict(),
            name=cfg.wandb_run_name,
        )
        logger.info("Initialized wandb run: %s", wandb_run.id)

    training_args = TrainingArguments(
        output_dir=cfg.save_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        logging_dir=cfg.log_dir,
        report_to=report_to,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        load_best_model_at_end=cfg.load_best_model_at_end,
        evaluation_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        remove_unused_columns=cfg.remove_unused_columns,
        eval_accumulation_steps=None,
    )

    callbacks: List[Any] = [
        EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
    ]

    use_prometheus = cfg.use_prometheus and Gauge is not None and start_http_server is not None
    if use_prometheus:
        start_http_server(cfg.prometheus_port)
        logger.info("Started Prometheus metrics server on port %d", cfg.prometheus_port)

        class PrometheusLoggingCallback(TrainerCallback):
            """Export training metrics to Prometheus."""

            def __init__(self) -> None:
                super().__init__()
                self.training_loss = Gauge("training_loss", "Training loss per evaluation step")
                self.map_metric = Gauge("validation_map", "Validation mean average precision")

            def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
                if logs is None:
                    return
                loss_val = logs.get("loss")
                if loss_val is not None:
                    try:
                        self.training_loss.set(float(loss_val))
                    except (TypeError, ValueError) as exc:
                        logger.warning("Failed to set Prometheus training_loss: %s", exc)

            def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any, **kwargs: Any) -> None:
                map_val = metrics.get("map")
                if map_val is not None:
                    try:
                        self.map_metric.set(float(map_val))
                    except (TypeError, ValueError) as exc:
                        logger.warning("Failed to set Prometheus validation_map: %s", exc)

        callbacks.append(PrometheusLoggingCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
        callbacks=callbacks,
    )

    logger.info("Starting training for %d epochs", training_args.num_train_epochs)
    trainer.train()

    if use_wandb:
        wandb.finish()


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description="Train an object detection model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args(argv)
    cfg = DetectionConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
