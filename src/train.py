import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from data_processing import load_custom_data, image_processor
from model import load_model
import yaml
from transformers import EarlyStoppingCallback
import numpy as np
from dataclasses import dataclass
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image, ImageDraw
from data_processing import load_custom_data, image_processor
# Load initial data and get categories
train_dataset, val_dataset = load_custom_data()
categories = train_dataset.categories

# Get mapping from category id to category name directly from the dataset
id2label = train_dataset.categories
label2id = {v: k for k, v in id2label.items()}


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as a list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):
                # Here we have "yolo" format (x_center, y_center, width, height) in relative coordinates 0..1
                # and we need to convert it to "pascal" format (x_min, y_min, x_max, y_max) in absolute coordinates.
                height, width = size
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([[width, height, width, height]])
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits), 
                pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):
        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per-class metrics with separate metric for each class.
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            # Use .get() to handle missing keys gracefully.
            class_name = self.id2label.get(class_id.item(), f"class_{class_id.item()}")
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics



# Instantiate the metric computation function.
eval_compute_metrics_fn = MAPEvaluator(
    image_processor=image_processor, threshold=0.01, id2label=id2label
)


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]  # Labels are a list of dictionaries.
    return {"pixel_values": pixel_values, "labels": labels}


# Load configuration.
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data.
train_dataset, val_dataset = load_custom_data()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
)

# Load model.
# Load and update model
# Load the model (pre-trained on COCO)
model = load_model()

# Define your custom number of classes
num_custom_classes = len(id2label)

# Replace all classification heads (final and auxiliary) in the model
for i in range(len(model.class_embed)):
    hidden_dim = model.class_embed[i].in_features
    model.class_embed[i] = torch.nn.Linear(hidden_dim, num_custom_classes)

# Update the model's configuration to match your custom number of classes
model.config.num_labels = num_custom_classes  # commonly used in transformer models
if hasattr(model.config, 'num_classes'):
    model.config.num_classes = num_custom_classes

# Now continue setting up your training arguments and Trainer
training_args = TrainingArguments(
    output_dir=config["save_dir"],
    num_train_epochs=config["num_epochs"],
    per_device_train_batch_size=config["batch_size"],
    learning_rate=float(config["learning_rate"]),
    logging_dir=config["log_dir"],
    report_to=config["report_to"],
    metric_for_best_model=config["metric_for_best_model"],
    greater_is_better=config["greater_is_better"],
    load_best_model_at_end=config["load_best_model_at_end"],
    eval_strategy=config["eval_strategy"],
    save_strategy=config["save_strategy"],
    save_total_limit=config["save_total_limit"],
    remove_unused_columns=config["remove_unused_columns"],
    eval_do_concat_batches=config["eval_do_concat_batches"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

if __name__ == "__main__":
    trainer.train()
