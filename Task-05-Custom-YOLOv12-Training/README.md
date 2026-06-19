# Task 05 - Training a Custom YOLO Model

## Overview

This project focuses on training a custom YOLO model for specialized object detection tasks. The implementation covers dataset preparation, transfer learning, model training, performance evaluation, and custom inference using a trained model for vehicle and license plate detection.

The objective of this task was to understand the complete workflow of custom object detector development, from preparing labeled datasets to deploying a trained model for real-world applications.

---

## Technologies Used

- Python
- YOLOv8n / YOLOv12
- Ultralytics
- OpenCV
- NumPy
- YAML

---

## Features Implemented

### Dataset Preparation

- YOLO format dataset organization
- Image and annotation management
- Custom class labeling
- Training and validation dataset setup

### Data Configuration

- Creation of `data.yaml`
- Dataset path configuration
- Class definition setup
- Training metadata management

### Transfer Learning

- Fine-tuning pretrained YOLO weights
- Custom object detection training
- Faster convergence using pretrained features
- Reduced training requirements

### Model Training

- Custom training pipeline
- Epoch configuration
- Batch size optimization
- Image size configuration
- Automatic checkpoint generation

### Custom Inference

- Detection using trained model weights
- Video-based inference
- Real-time object localization
- Confidence score generation

### Cascaded Detection Pipeline

- Vehicle detection using pretrained YOLO
- License plate detection using custom-trained YOLO
- Region-based secondary detection
- Multi-stage inference workflow

### Model Evaluation

- mAP@0.5 evaluation
- mAP@0.5:0.95 evaluation
- Detection performance analysis
- Training metric monitoring

---

## Workflow

1. Prepare and label the custom dataset.
2. Organize images and annotations in YOLO format.
3. Configure dataset settings using `data.yaml`.
4. Load pretrained YOLO weights.
5. Fine-tune the model on the custom license plate dataset.
6. Train the model using specified epochs, batch size, and image size.
7. Save the best-performing model weights.
8. Evaluate performance using mAP metrics.
9. Detect vehicles using a pretrained YOLO model.
10. Detect license plates within vehicle regions using the custom-trained model.
11. Draw bounding boxes and confidence scores.
12. Generate annotated videos and detection logs.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Custom Dataset Creation
- Object Detection Training Pipelines
- Transfer Learning
- YOLO Training Configuration
- Performance Evaluation Metrics
- Mean Average Precision (mAP)
- Multi-Stage Detection Systems
- Model Optimization and Validation
- Real-World Computer Vision Deployment

---

## Results

The custom YOLO model successfully learned to detect license plates after training on the custom dataset.

### Key Observations

- Transfer learning significantly accelerated training convergence.
- The model achieved reliable detection performance after a relatively small number of training epochs.
- Cascaded detection effectively combined vehicle and license plate localization.
- Training metrics showed continuous improvement throughout the learning process.
- Generated videos displayed accurate vehicle and license plate detection simultaneously.
- mAP scores indicated strong detection performance and model generalization.

---

## Generated Output Files

### best.pt

Best-performing trained model weights generated during training and used for custom inference.

### output_video.mp4

Annotated output video containing:

- Vehicle Bounding Boxes
- License Plate Bounding Boxes
- Detection Labels
- Confidence Scores

### license_plate_detections.json

Detection log containing:

- Frame Number
- Vehicle Coordinates
- License Plate Coordinates
- Confidence Scores
- Detection Metadata

### training_logs.txt

Training statistics including:

- Epoch Information
- Loss Values
- Validation Metrics
- Training Progress

### mAP.txt

Model evaluation report containing:

- mAP@0.5
- mAP@0.5:0.95
- Detection Accuracy Metrics

### data.yaml

Dataset configuration file containing:

- Dataset Paths
- Class Definitions
- Training Configuration

---

## Files Included

- Training Scripts
- Inference Scripts
- Dataset Configuration Files
- Detection Logs
- Evaluation Metrics
- Screenshots
- Output Videos
- Training Results
