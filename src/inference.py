import torch
from PIL import Image
from transformers import AutoImageProcessor
from model import load_model
import yaml
import os

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Model & Processor
model = load_model()
image_processor = AutoImageProcessor.from_pretrained(config["model_checkpoint"])

def run_inference(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Predictions:", outputs)

if __name__ == "__main__":
    test_images_dir = config["test_images_dir"]
    test_images = os.listdir(test_images_dir)
    
    if test_images:
        sample_image = os.path.join(test_images_dir, test_images[0])
        run_inference(sample_image)
    else:
        print("No test images found.")
