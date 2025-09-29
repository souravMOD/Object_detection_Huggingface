"""
Training script for object detection using Hugging Face transformers.

This script wires together the data pipeline, model, optimizer and
metric computation using the ``transformers.Trainer`` API. It avoids
expensive operations at import time and exposes a ``main`` function
that can be called from other Python code or via the command line.

Example usage from the command line:

    python -m updated_code.src.train --config config.yaml

The configuration file defines dataset paths, model checkpoint,
hyperparameters and logging options. See ``config.yaml`` for an example.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# Optional ML‑Ops integrations
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # wandb is optional and only used if installed and enabled

try:
    from prometheus_client import Gauge, start_http_server  # type: ignore
    from transformers import TrainerCallback
except ImportError:
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore
    TrainerCallback = object  # fallback base class

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    EvalPrediction,
)
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .data_processing import (
    load_config,
    load_datasets,
    get_image_processor,
)
from .model import (
    get_label_mappings,
    load_model,
    adapt_model_for_custom_classes,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to stack pixel values and keep labels list.

    The Hugging Face Trainer expects a dict with ``pixel_values``
    stacked into a tensor and ``labels`` as a list of dicts. See
    transformers documentation for details.
    """
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]
    return {"pixel_values": pixel_values, "labels": labels}


class MAPEvaluator:
    """
    Compute mean average precision (mAP) over a validation dataset.

    This callable is passed to ``Trainer`` via the ``compute_metrics``
    argument. It converts model outputs and targets into the format
    expected by ``torchmetrics.detection.mean_ap.MeanAveragePrecision``.
    """

    def __init__(self, image_processor, threshold: float = 0.0, id2label: Dict[int, str] | None = None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label or {}

    def _collect_image_sizes(self, targets: List[List[Dict[str, Any]]]) -> List[torch.Tensor]:
        """Collect original image sizes from target annotations.

        Each element of ``targets`` is a list of dictionaries, one per
        image in the batch. The ``size`` field is a tuple of (height,
        width). These are converted into tensors for use by the HF
        post‑processing API.
        """
        image_sizes: List[torch.Tensor] = []
        for batch in targets:
            sizes = [torch.tensor(t["size"]) for t in batch]
            image_sizes.append(torch.stack(sizes))
        return image_sizes

    def _collect_targets(self, targets: List[List[Dict[str, Any]]], image_sizes: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Convert targets from relative [cx, cy, w, h] to absolute [x1, y1, x2, y2]."""
        post_processed_targets: List[Dict[str, Any]] = []
        for target_batch, size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, size_batch):
                height, width = size.tolist()
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([width, height, width, height])
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def _collect_predictions(self, predictions: List[Any], image_sizes: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Post‑process model predictions into bounding boxes and labels."""
        post_processed_predictions: List[Dict[str, Any]] = []
        for batch_pred, sizes in zip(predictions, image_sizes):
            # ``batch_pred`` is a tuple: (loss, logits, pred_boxes)
            # Trainer returns predictions as a tuple of (losses, outputs) per batch.
            # Here we ignore the loss and treat logits and boxes separately.
            _, logits, pred_boxes = batch_pred
            output = type("ModelOutput", (), {})()
            output.logits = torch.tensor(logits)
            output.pred_boxes = torch.tensor(pred_boxes)
            processed = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=sizes
            )
            post_processed_predictions.extend(processed)
        return post_processed_predictions

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
        # Extract per‑class metrics and flatten into top‑level keys
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_id_int = class_id.item() if hasattr(class_id, "item") else int(class_id)
            class_name = self.id2label.get(class_id_int, f"class_{class_id_int}")
            metrics[f"map_{class_name}"] = class_map.item() if hasattr(class_map, "item") else float(class_map)
            metrics[f"mar_100_{class_name}"] = class_mar.item() if hasattr(class_mar, "item") else float(class_mar)
        # Round all metrics to four decimals for readability
        return {k: round(float(v), 4) for k, v in metrics.items()}


def train(cfg: Dict[str, Any]) -> None:
    """Execute a full training loop based on a configuration dictionary.

    This function wires together the data pipeline, model and training loop. It also
    optionally integrates with ML‑Ops tooling such as Weights & Biases (wandb) and
    Prometheus/Grafana for metric reporting. These integrations are enabled via
    configuration flags in ``cfg`` and require the corresponding packages to
    be installed (see ``requirements.txt``).
    """
    # Load datasets
    train_dataset, val_dataset = load_datasets(cfg)
    logger.info("Loaded %d training samples and %d validation samples", len(train_dataset), len(val_dataset))

    # Derive label mappings and model
    id2label, label2id = get_label_mappings(train_dataset)
    model = load_model(cfg, id2label, label2id)
    model = adapt_model_for_custom_classes(model, len(id2label))

    # Instantiate image processor for metrics
    image_processor = get_image_processor(cfg["model_checkpoint"], cfg["image_size"])
    metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=cfg.get("map_threshold", 0.01), id2label=id2label)

    # Determine reporting backend for Hugging Face Trainer. Default is ``none``.
    report_to = cfg.get("report_to", "none")
    # Optionally override with wandb if enabled
    use_wandb = bool(cfg.get("use_wandb", False)) and wandb is not None
    if use_wandb:
        report_to = "wandb"
        wandb_project = cfg.get("wandb_project", "object_detection")
        wandb_run = wandb.init(project=wandb_project, config=cfg, name=cfg.get("wandb_run_name"))
        logger.info("Initialized wandb run: %s", wandb_run.id)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=cfg.get("save_dir", "checkpoints/"),
        num_train_epochs=cfg.get("num_epochs", 10),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        per_device_eval_batch_size=cfg.get("batch_size", 4),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        logging_dir=cfg.get("log_dir", "logs/"),
        report_to=report_to,
        metric_for_best_model=cfg.get("metric_for_best_model", None),
        greater_is_better=cfg.get("greater_is_better", True),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        evaluation_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        eval_accumulation_steps=None,
    )

    # Set up Prometheus/Grafana metrics if requested and available
    callbacks: List[Any] = [EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience", 3))]
    use_prometheus = bool(cfg.get("use_prometheus", False)) and Gauge is not None and start_http_server is not None
    if use_prometheus:
        prometheus_port = int(cfg.get("prometheus_port", 8000))
        start_http_server(prometheus_port)
        logger.info("Started Prometheus metrics server on port %d", prometheus_port)

        class PrometheusLoggingCallback(TrainerCallback):
            """Callback to expose key metrics to Prometheus for Grafana dashboards."""

            def __init__(self) -> None:
                super().__init__()
                # Define gauges for metrics we care about. Additional metrics can be added here.
                self.training_loss = Gauge("training_loss", "Training loss per evaluation step")
                self.map_metric = Gauge("validation_map", "Validation mean average precision")

            def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # type: ignore[override]
                if logs is None:
                    return
                # Capture training loss if present
                loss_val = logs.get("loss")
                if loss_val is not None:
                    try:
                        self.training_loss.set(float(loss_val))
                    except Exception:
                        pass

            def on_evaluate(self, args, state, control, metrics, **kwargs) -> None:  # type: ignore[override]
                # Capture mAP if present in evaluation metrics
                map_val = metrics.get("map")
                if map_val is not None:
                    try:
                        self.map_metric.set(float(map_val))
                    except Exception:
                        pass

        callbacks.append(PrometheusLoggingCallback())

    # Instantiate Trainer
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

    # Finish wandb run if used
    if use_wandb:
        wandb.finish()


def main(argv: List[str] | None = None) -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description="Train an object detection model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()