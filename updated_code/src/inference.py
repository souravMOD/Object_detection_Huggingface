"""
Inference script for object detection models.

This module provides a simple command‑line interface to run
object detection on a single image. The model is loaded from a
configuration file or a fine‑tuned checkpoint directory. Predictions
are post‑processed to obtain bounding boxes, labels and scores, which
are printed to stdout.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForObjectDetection

from .data_processing import load_config, get_image_processor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference(image_path: str, cfg: Dict[str, Any], checkpoint_path: str | None = None, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Run object detection on a single image.

    Args:
        image_path: Path to the image file.
        cfg: Configuration dictionary.
        checkpoint_path: Optional path to a fine‑tuned model directory.
        threshold: Score threshold for filtering predictions.

    Returns:
        A list of predictions, each a dict with keys ``boxes``, ``scores`` and ``labels``.
    """
    # Determine which model checkpoint to load
    model_checkpoint = checkpoint_path or cfg["model_checkpoint"]
    logger.info("Loading model from %s", model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(model_checkpoint)

    # Load image processor using the original model checkpoint for size
    processor = get_image_processor(cfg["model_checkpoint"], cfg["image_size"])

    # Prepare image
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Post‑process predictions; target_size expects (height, width)
    # We use the processed image size from the config here
    target_sizes = torch.tensor([[cfg["image_size"], cfg["image_size"]]])
    results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    # Map label ids back to names if available
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    predictions: List[Dict[str, Any]] = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = id2label.get(int(label_id), str(label_id)) if id2label else str(label_id)
        predictions.append(
            {
                "label": label,
                "score": float(score),
                "box": [float(coord) for coord in box],
            }
        )
    return predictions


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a fine‑tuned model directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for predictions.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    preds = run_inference(args.image, cfg, checkpoint_path=args.checkpoint, threshold=args.threshold)
    if not preds:
        print("No objects detected above the threshold.")
    for pred in preds:
        label, score, box = pred["label"], pred["score"], pred["box"]
        print(f"{label}: {score:.3f} | bbox: {box}")


if __name__ == "__main__":
    main()