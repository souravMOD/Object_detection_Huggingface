"""
Inference utilities for object detection.

Run detection on a single image via CLI or Python API.

CLI usage::

    python -m src.inference --image photo.jpg --config config.yaml

Python usage::

    from src.inference import run_inference
    from src.config import DetectionConfig

    cfg = DetectionConfig()
    preds = run_inference("photo.jpg", cfg, checkpoint_path="checkpoints/best")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForObjectDetection

from .config import DetectionConfig
from .data_processing import get_image_processor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference(
    image_path: str,
    cfg: DetectionConfig | Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run object detection on a single image.

    Args:
        image_path: Path to the image file.
        cfg: A ``DetectionConfig`` or plain dictionary.
        checkpoint_path: Optional path to a fine-tuned model directory.
        threshold: Score threshold for filtering predictions.
            Defaults to ``cfg.score_threshold``.

    Returns:
        A list of dicts with ``label``, ``score``, and ``box`` keys.

    Raises:
        FileNotFoundError: If the image does not exist.
    """
    if isinstance(cfg, dict):
        cfg = DetectionConfig.from_dict(cfg)

    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")

    if threshold is None:
        threshold = cfg.score_threshold

    model_checkpoint = checkpoint_path or cfg.model_checkpoint
    logger.info("Loading model from %s", model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(model_checkpoint)
    processor = get_image_processor(cfg.model_checkpoint, cfg.image_size)

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[cfg.image_size, cfg.image_size]])
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    id2label = getattr(model.config, "id2label", None)
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a fine-tuned model directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold for predictions.")
    args = parser.parse_args(argv)

    cfg = DetectionConfig.from_yaml(args.config)
    preds = run_inference(args.image, cfg, checkpoint_path=args.checkpoint, threshold=args.threshold)
    if not preds:
        print("No objects detected above the threshold.")
    for pred in preds:
        print(f"{pred['label']}: {pred['score']:.3f} | bbox: {pred['box']}")


if __name__ == "__main__":
    main()
