# Task 04 - YOLO Pretrained Inference

## Overview

This project focuses on real-time object detection using pretrained YOLO (You Only Look Once) models. The implementation explores multiple YOLO versions, performs object detection on video streams, generates annotated output videos, and stores structured detection data for further analysis.

The objective of this task was to understand modern object detection workflows, compare different YOLO versions, and gain practical experience with real-time inference using pretrained models.

---

## Technologies Used

- Python
- OpenCV (DNN Module)
- Ultralytics
- YOLOv4
- YOLOv8
- YOLOv12
- NumPy
- JSON

---

## Features Implemented

### YOLO Object Detection

- Real-time object detection
- Multi-class object recognition
- Confidence score generation
- Bounding box visualization

### Multiple YOLO Versions

- YOLOv4
- YOLOv8
- YOLOv12
- Performance and architecture comparison

### Pretrained Model Inference

- COCO dataset pretrained weights
- Detection of 80 object classes
- Ready-to-use object recognition

### Video Processing

- Frame-by-frame video analysis
- Video annotation
- Output video generation

### Non-Maximum Suppression (NMS)

- Removal of duplicate detections
- Overlap filtering
- Selection of highest-confidence detections

### Detection Logging

- JSON output generation
- Frame-wise detection records
- Confidence score storage
- Bounding box coordinate storage

---

## Workflow

1. Select the desired YOLO version.
2. Load pretrained model weights and configuration files.
3. Read video input frame by frame.
4. Perform object detection on each frame.
5. Generate confidence scores and bounding boxes.
6. Apply Non-Maximum Suppression (NMS).
7. Draw bounding boxes and class labels.
8. Save annotated frames to output video.
9. Store detection results in JSON format.
10. Generate final detection reports and output videos.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Fundamentals of Object Detection
- YOLO Architecture
- Real-Time Inference Pipelines
- Video Processing Techniques
- Bounding Box Generation
- Confidence Score Evaluation
- Non-Maximum Suppression (NMS)
- Detection Data Serialization
- Multi-Version Model Comparison

---

## Results

The implementation successfully detected and classified objects in video streams using all three YOLO versions.

### Key Observations

- YOLOv4 required manual implementation of the complete detection pipeline using OpenCV DNN.
- YOLOv8 and YOLOv12 significantly simplified inference through the Ultralytics framework.
- Non-Maximum Suppression effectively eliminated duplicate detections.
- Annotated videos clearly displayed detected objects with labels and confidence scores.
- Detection logs provided structured output suitable for downstream analytics and tracking systems.

---

## Generated Output Files

### output_video.mp4

Annotated output videos generated for:

- YOLOv4
- YOLOv8
- YOLOv12

Features included:

- Bounding Boxes
- Object Labels
- Confidence Scores

### detections.json

Detection logs containing:

- Frame Number
- Class ID
- Class Label
- Confidence Score
- Bounding Box Coordinates

### yolov4-tiny.weights

Pretrained YOLOv4 model weights used for inference.

### yolov4-tiny.cfg

YOLOv4 network configuration file defining the architecture used during detection.

---

## Files Included

- Source Code
- Input Videos
- Output Videos
- Detection JSON Files
- Screenshots
- YOLO Configuration Files
- Inference Results
