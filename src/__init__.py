"""
Plug-and-play object detection with Hugging Face Transformers.

Quick start — detect objects in 3 lines::

    from src import detect
    predictions = detect("photo.jpg", checkpoint="path/to/model")
    print(predictions)

Train on your own dataset::

    from src import train_model
    train_model(config="config.yaml")

Evaluate a checkpoint::

    from src import evaluate_model
    metrics = evaluate_model(config="config.yaml", checkpoint="checkpoints/best")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import DetectionConfig

__all__ = [
    "DetectionConfig",
    "detect",
    "train_model",
    "evaluate_model",
]


def detect(
    image: str,
    *,
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    threshold: float = 0.5,
    model_checkpoint: str = "PekingU/rtdetr_v2_r18vd",
) -> List[Dict[str, Any]]:
    """Run object detection on a single image — the simplest entry point.

    Args:
        image: Path to an image file.
        checkpoint: Path to a fine-tuned model directory.
        config: Optional path to a YAML config file.
        threshold: Score threshold for predictions (default 0.5).
        model_checkpoint: HuggingFace model ID (used when no config is given).

    Returns:
        A list of dicts, each with ``label``, ``score``, and ``box`` keys.

    Example::

        >>> from src import detect
        >>> preds = detect("cat.jpg", checkpoint="checkpoints/best")
        >>> for p in preds:
        ...     print(f"{p['label']}: {p['score']:.2f}")
    """
    from .inference import run_inference

    if config:
        cfg = DetectionConfig.from_yaml(config)
    else:
        cfg = DetectionConfig(model_checkpoint=model_checkpoint)

    return run_inference(image, cfg, checkpoint_path=checkpoint, threshold=threshold)


def train_model(
    config: str = "config.yaml",
    **overrides: Any,
) -> None:
    """Train an object detection model — one-liner API.

    Args:
        config: Path to a YAML configuration file.
        **overrides: Any config field can be overridden as a keyword arg.

    Example::

        >>> from src import train_model
        >>> train_model("config.yaml", num_epochs=10, batch_size=4)
    """
    from .train import train

    data = DetectionConfig.from_yaml(config).to_dict()
    data.update(overrides)
    cfg = DetectionConfig.from_dict(data)
    train(cfg)


def evaluate_model(
    config: str = "config.yaml",
    checkpoint: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, float]:
    """Evaluate a model checkpoint — one-liner API.

    Args:
        config: Path to a YAML configuration file.
        checkpoint: Optional path to a fine-tuned model directory.
        **overrides: Any config field can be overridden as a keyword arg.

    Returns:
        A dictionary of evaluation metrics.

    Example::

        >>> from src import evaluate_model
        >>> metrics = evaluate_model("config.yaml", checkpoint="checkpoints/best")
        >>> print(f"mAP: {metrics.get('eval_map', 'N/A')}")
    """
    from .evaluate import evaluate

    data = DetectionConfig.from_yaml(config).to_dict()
    data.update(overrides)
    cfg = DetectionConfig.from_dict(data)
    return evaluate(cfg, checkpoint_path=checkpoint)
