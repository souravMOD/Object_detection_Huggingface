import torch
from transformers import AutoModelForObjectDetection
import yaml
from data_processing import load_custom_data, image_processor
# Load initial data and get categories
train_dataset, val_dataset = load_custom_data()
categories = train_dataset.categories

# Get mapping from category id to category name directly from the dataset
id2label = train_dataset.categories
label2id = {v: k for k, v in id2label.items()}
# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_model():
    model = AutoModelForObjectDetection.from_pretrained(
        config["model_checkpoint"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,

    )
    return model
