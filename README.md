# Object Detection with Huggingface

This project demonstrates how to perform object detection using the Huggingface library. It includes scripts and notebooks to train and evaluate object detection models.

## Project Structure

```
Object_detection_Huggingface/
│
├── data/                   # Directory for datasets
│   └── ...                 # Your dataset files
│
├── notebooks/              # Jupyter notebooks for experiments
│   └── ...                 # Your notebook files
│
├── scripts/                # Python scripts for training, evaluation, and inference
│   ├── train.py            # Script to train the model
│   ├── evaluate.py         # Script to evaluate the model
│   └── inference.py        # Script to run inference on new images
│
├── config.yaml             # Configuration file for training and evaluation
├── requirements.txt        # List of dependencies
├── LICENSE                 # License file
└── README.md               # Project README file
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Object_detection_Huggingface.git
    cd Object_detection_Huggingface
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place it in the `data` directory.

2. Train the model:
    ```bash
    python scripts/train.py 
    ```

3. Evaluate the model:
    ```bash
    python scripts/evaluate.py 
    ```

4. Run inference on new images:
    ```bash
    python scripts/inference.py 
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
