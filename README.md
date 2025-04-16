# Power Transmission Lines Damage Detection

<img src="https://www4.djicdn.com/cms_uploads/ckeditor/pictures/1187/content_9Q3A8646.JPG" width="500" height="300">


## Overview

This project leverages Computer Vision with Transformers to detect damages in power transmission lines. Modern societies depend on uninterrupted electricity, making transmission system resilience critical. Power lines are routinely inspected for damage detection - a cumbersome but essential process. This project automates damage detection using drone or inspection robot footage.

## Key Features

- **Vision Transformer Approach**: Uses DETR (Detection Transformer) instead of traditional CNN methods
- **Real-time Capability**: Optimized for processing video streams from drones/robots
- **Damage Classification**: Identifies two types of damage:
  - Cable breaks
  - Thunderbolt damage

## Demo

![Demo Image](https://www.fortnightly.com/sites/default/files/styles/story_large/public/1604-COL1.jpg?itok=SQyLIVqa)

*Add your model output visualizations here*

## Technologies Used

- **PyTorch** & **PyTorch Lightning** - Deep learning framework
- **Transformers** - DETR model architecture
- **Supervision** - For annotation and visualization
- **OpenCV** - Image processing
- **Roboflow** - Dataset management

## Dataset

The project uses the Cable Damage dataset available on Roboflow:
- 2 damage classes: **Break** and **Thunderbolt**
- Balanced class distribution
- Properly split into training, validation, and test sets

Dataset distribution:
- Training set: ~70%
- Validation set: ~15%
- Test set: ~15%

## Model Architecture

This project implements Facebook Research's DETR (Detection Transformer) architecture, which:

- Uses a CNN backbone (ResNet-50) to extract features
- Applies a transformer encoder-decoder architecture
- Directly predicts object bounding boxes and classes
- Eliminates the need for non-maximum suppression and anchor generation

The model was fine-tuned from the pre-trained `facebook/detr-resnet-50` checkpoint.

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/power-line-damage-detection.git
cd power-line-damage-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset
# (Instructions in notebook or use the Roboflow link)
```

## Usage

```python
# Example code to load the model and make predictions
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2

# Load model and processor
processor = DetrImageProcessor.from_pretrained("path/to/saved/model")
model = DetrForObjectDetection.from_pretrained("path/to/saved/model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Make prediction on an image
image = cv2.imread("your_image.jpg")
inputs = processor(images=image, return_tensors='pt').to(device)
outputs = model(**inputs)

# Post-process results
target_sizes = torch.tensor([image.shape[:2]]).to(device)
results = processor.post_process_object_detection(
    outputs=outputs,
    threshold=0.5,
    target_sizes=target_sizes
)[0]

# Display results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [int(i) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score.item():.2f} at {box}")
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
```

## Results

The model was trained for 126 epochs using PyTorch Lightning on GPU hardware. Testing shows successful detection of both break and thunderbolt damage with confidence scores above 0.5.

*Add metrics, confusion matrix, or other evaluation results*

## Future Improvements

- Expand the dataset with more damage types
- Implement real-time processing for drone footage
- Explore other transformer architectures for comparison
- Add severity classification for detected damage
- Develop a web application for easy deployment

## Acknowledgments

- [Facebook Research DETR](https://github.com/facebookresearch/detr) for the model architecture
- [Roboflow](https://universe.roboflow.com/roboflow-100/cable-damage/dataset/2) for the dataset
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for the training framework

## References

- [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872)
- [Cable Damage Dataset](https://universe.roboflow.com/roboflow-100/cable-damage/dataset/2)

---

*This project is part of my machine learning portfolio showcasing practical applications of computer vision in critical infrastructure.*
