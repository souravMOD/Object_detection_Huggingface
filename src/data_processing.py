import os
import json
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
import yaml
from torch.utils.data import Dataset

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define Augmentations
train_transform = A.Compose([
    A.Perspective(p=0.1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.1) # ✅ Ensures bounding box transformation is applied
], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1))


#valid_transform = A.Compose([], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True))
valid_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
)
# Load Image Processor
image_processor = AutoImageProcessor.from_pretrained(
    config["model_checkpoint"],
    size={"width": config["image_size"], "height": config["image_size"]},
    use_fast=True
)


class CustomCOCODataset(Dataset):
    def __init__(self, images_dir, annotation_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_id_to_filename = {img["id"]: img["file_name"] for img in self.coco_data["images"]}
        self.annotations = self.coco_data["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, idx):
        image_id = self.coco_data["images"][idx]["id"]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, image_filename)

        # Load image and convert to numpy array
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Get annotations for this image
        annotations = [ann for ann in self.annotations if ann["image_id"] == image_id]
        boxes = []
        for ann in annotations:
            x_min, y_min, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x_min, y_min, w, h])  # Keep COCO format
            else:
                print(f"⚠️ Skipping invalid bbox: {ann['bbox']} (image_id: {image_id})")

        # Get categories corresponding to boxes
        categories = [ann["category_id"] for ann in annotations]

        # Apply transformations (which expect COCO-format boxes)
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=categories)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        # Format annotations for the image processor (if needed, you can convert them later)
        formatted_annotations = {"image_id": image_id, "annotations": []}
        for cat, bbox in zip(categories, boxes):
            formatted_annotations["annotations"].append({
                "category_id": cat,
                "bbox": bbox,  # still in COCO format
                "iscrowd": 0,
                "area": bbox[2] * bbox[3]
            })

        # Process the image using the Hugging Face image processor
        result = image_processor(images=image, annotations=formatted_annotations, return_tensors="pt")
        result = {k: v[0] for k, v in result.items()}  # remove batch dimension

        return result


def load_custom_data():
    train_dataset = CustomCOCODataset(config["train_images_dir"], config["train_annotations"], transform=train_transform)
    val_dataset = CustomCOCODataset(config["val_images_dir"], config["val_annotations"], transform=valid_transform)
    return train_dataset, val_dataset
