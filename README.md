# Object Detection with Hugging Face

This repository contains an object detection pipeline built on top of the Hugging Face Transformers library.  It provides a full training loop, evaluation tools and inference utilities for detecting objects in images using modern transformer‑based detectors.

The code is organized under the `updated_code` package and is designed to be production‑ready: imports are lightweight, configuration is externalized, and you can invoke functions from your own Python code or use the provided command‑line interfaces.

## Features

- **Modular data pipeline** – dataset loading and pre‑processing are encapsulated in `updated_code/src/data_processing.py`.
- **Model adaptation** – automatically adapts a pretrained detector to the number of classes in your dataset.
- **Training script** – configurable via YAML; supports early stopping and model checkpointing via the `transformers.Trainer` API.
- **Evaluation and inference** – compute mean average precision (mAP) on a validation set or run detection on single images.
- **ML‑Ops integration** – optional logging to [Weights & Biases](https://wandb.ai/) and metric export via [Prometheus](https://prometheus.io/) for use in dashboards such as Grafana.

## Installation

1. Clone this repository and navigate into it:

   ```bash
   git clone https://github.com/souravMOD/Object_detection_Huggingface.git
   cd Object_detection_Huggingface
   ```

2. Install the required Python packages.  We recommend using a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r updated_code/requirements.txt
   ```

   Additional optional dependencies for ML‑Ops are also included in the `requirements.txt` (e.g. `wandb` for experiment tracking and `prometheus-client` for metrics export).

## Quick Start

1. **Prepare your dataset**

   Create a YAML configuration file (see `updated_code/config.yaml` for an example) that specifies the paths to your training and validation datasets in COCO format, the pretrained model checkpoint to adapt, and training hyper‑parameters.

2. **Training**

   Run the training script with your configuration file:

   ```bash
   python -m updated_code.src.train --config updated_code/config.yaml
   ```

   The script will load the datasets, adapt the pretrained model, and train for the specified number of epochs.  Checkpoints are saved in the directory defined by `save_dir` in the config.  You can enable Weights & Biases logging and Prometheus metrics by adding the following fields to your configuration:

   ```yaml
   use_wandb: true        # set to true to enable wandb logging
   wandb_project: my-project  # optional project name for wandb
   use_prometheus: true   # export metrics via Prometheus
   prometheus_port: 8000  # port on which to expose Prometheus metrics
   ```

3. **Evaluation**

   To evaluate a trained model checkpoint on the validation set:

   ```bash
   python -m updated_code.src.evaluate --config updated_code/config.yaml --checkpoint path/to/checkpoint
   ```

   The evaluator computes mAP at multiple IoU thresholds and prints the results.  If wandb logging is enabled, evaluation metrics will also be reported there.

4. **Inference**

   To run inference on a single image and print the detected objects:

   ```bash
   python -m updated_code.src.inference --image path/to/image.jpg \
     --config updated_code/config.yaml --checkpoint path/to/checkpoint
   ```

   The script loads the model and image processor, performs detection, and prints out class labels, scores and bounding boxes.

## ML‑Ops Integrations

### Weights & Biases

The training script supports experiment tracking via [Weights & Biases](https://wandb.ai/).  Set `use_wandb: true` in your configuration and provide an optional `wandb_project` name to start logging metrics, loss curves and model checkpoints to your wandb workspace.  You may also set `wandb_run_name` if you want a custom run name.

### Prometheus & Grafana

For operational monitoring, the training script can export key metrics such as training loss and validation mAP via a Prometheus HTTP server.  To enable this, set `use_prometheus: true` and optionally `prometheus_port` in your configuration.  Grafana can then scrape these metrics from the specified port and visualize them on a dashboard of your choice.


## License

This project is licensed under the Apache 2.0 License.  See the [LICENSE](LICENSE) file for details.
