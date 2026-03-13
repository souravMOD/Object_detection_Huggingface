# Object Detection with Hugging Face

A **plug-and-play** object detection pipeline built on [Hugging Face Transformers](https://huggingface.co/docs/transformers). Drop in your COCO-format dataset, point at any HuggingFace detection model, and start training — or run inference in three lines of Python.

## Features

- **3-line inference** — `from src import detect; detect("photo.jpg")`
- **Typed, validated config** — dataclass with env-var overrides (`HF_DET_*`)
- **Modular data pipeline** — COCO dataset loader with Albumentations augmentations
- **Auto model adaptation** — swaps classification heads to match your class count
- **HF Trainer integration** — early stopping, checkpointing, mixed precision
- **MLOps ready** — optional Weights & Biases and Prometheus/Grafana support
- **CI/CD** — GitHub Actions for lint + test + Docker build
- **Installable package** — `pip install -e .` with CLI entry points

## Quick Start

### Install

```bash
git clone https://github.com/souravMOD/Object_detection_Huggingface.git
cd Object_detection_Huggingface

# Option A: pip install (recommended)
pip install -e .

# Option B: requirements.txt
pip install -r requirements.txt

# With MLOps extras
pip install -e ".[mlops]"

# With dev tools (pytest, ruff, pre-commit)
pip install -e ".[dev]"
```

### Detect objects (plug-and-play)

```python
from src import detect

# With a fine-tuned checkpoint
predictions = detect("photo.jpg", checkpoint="checkpoints/best")

# With a pretrained HuggingFace model (zero config)
predictions = detect("photo.jpg", model_checkpoint="facebook/detr-resnet-50")

for p in predictions:
    print(f"{p['label']}: {p['score']:.2f} at {p['box']}")
```

### Train on your dataset

1. Prepare your dataset in COCO format
2. Copy and edit the config:

```bash
cp config.example.yaml config.yaml
# Edit paths in config.yaml to point at your dataset
```

3. Train:

```python
from src import train_model

# One-liner
train_model("config.yaml")

# With overrides
train_model("config.yaml", num_epochs=10, batch_size=4)
```

Or via CLI:

```bash
python -m src.train --config config.yaml
# or after pip install -e .
hf-detect-train --config config.yaml
```

### Evaluate

```python
from src import evaluate_model

metrics = evaluate_model("config.yaml", checkpoint="checkpoints/best")
print(f"mAP: {metrics.get('eval_map', 'N/A')}")
```

```bash
python -m src.evaluate --config config.yaml --checkpoint checkpoints/best
```

## Configuration

All settings live in a single YAML file. See [`config.example.yaml`](config.example.yaml) for the full reference.

Every field can be overridden via environment variables prefixed with `HF_DET_`:

```bash
HF_DET_BATCH_SIZE=16 HF_DET_LEARNING_RATE=1e-4 python -m src.train --config config.yaml
```

Or programmatically:

```python
from src.config import DetectionConfig

cfg = DetectionConfig(model_checkpoint="facebook/detr-resnet-50", batch_size=16)
cfg = DetectionConfig.from_yaml("config.yaml")
cfg = DetectionConfig.from_dict({"batch_size": 16})
```

## Project Structure

```
├── src/
│   ├── __init__.py          # Plug-and-play API (detect, train_model, evaluate_model)
│   ├── config.py            # Typed, validated DetectionConfig dataclass
│   ├── data_processing.py   # COCO dataset, augmentations, image processor
│   ├── model.py             # Model loading and head adaptation
│   ├── train.py             # Training loop with Trainer API
│   ├── evaluate.py          # mAP evaluation
│   └── inference.py         # Single-image inference
├── tests/                   # pytest test suite
├── config.yaml              # Your config (git-ignored or committed)
├── config.example.yaml      # Template config with documentation
├── pyproject.toml           # Package metadata, deps, tool config
├── requirements.txt         # Flat dependency list
├── Dockerfile               # Multi-stage (production + dev)
├── Makefile                 # Convenience commands
├── .pre-commit-config.yaml  # Code quality hooks
└── .github/workflows/ci.yml # CI pipeline
```

## Development

```bash
# Install with dev extras
pip install -e ".[all]"

# Run tests
make test

# Lint
make lint

# Auto-format
make format

# Set up pre-commit hooks
pre-commit install
```

## Docker

```bash
# Build production image
docker build --target production -t hf-detect .

# Train
docker run -v $(pwd)/datasets:/app/datasets -v $(pwd)/checkpoints:/app/checkpoints hf-detect

# Build dev image and run tests
docker build --target dev -t hf-detect-dev .
docker run hf-detect-dev
```

## MLOps Integrations

### Weights & Biases

```yaml
use_wandb: true
wandb_project: "my-detection-project"
wandb_run_name: "experiment-1"
```

### Prometheus & Grafana

```yaml
use_prometheus: true
prometheus_port: 8000
```

Exports `training_loss` and `validation_map` gauges for Grafana dashboards.

## License

This project is licensed under the GNU GPLv3 License. See [LICENSE](LICENSE) for details.
