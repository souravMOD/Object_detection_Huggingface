import torch
from transformers import Trainer
from model import load_model
from data_processing import valid_dataset

# Load Model
model = load_model()

# Load Trainer
trainer = Trainer(model=model)

# Evaluate
metrics = trainer.evaluate(valid_dataset)
print("Evaluation Metrics:", metrics)
