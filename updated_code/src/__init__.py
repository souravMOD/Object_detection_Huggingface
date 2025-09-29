"""
Subpackage containing the core implementation of the production‑ready
object detection project. Modules included here provide data
processing utilities, model helpers, training routines, evaluation
scripts and inference support.

To train a model programmatically:

    from updated_code.src.train import train
    from updated_code.src.data_processing import load_config

    cfg = load_config("config.yaml")
    train(cfg)

Alternatively, invoke the CLI entrypoints directly with ``python -m``.
"""
