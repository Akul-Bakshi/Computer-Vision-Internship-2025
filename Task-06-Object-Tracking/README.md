# Task 06 - Object Tracking Integration

## Overview

This project focuses on integrating object detection with multi-object tracking systems. The implementation combines YOLO-based object detection with the OCSORT tracking algorithm to maintain consistent object identities across video frames.

The objective of this task was to understand the tracking-by-detection approach, assign persistent IDs to detected objects, and generate detailed tracking logs for further analysis.

---

## Technologies Used

- Python
- YOLOv12s
- OCSORT
- OpenCV
- NumPy
- JSON

---

## Features Implemented

### Object Detection

- YOLOv12s-based object detection
- Confidence score filtering
- Real-time detection pipeline
- Bounding box generation

### Multi-Object Tracking

- OCSORT Tracker Integration
- Observation-Centric SORT Algorithm
- Consistent Object Identity Management
- Multi-Object Tracking Support

### Track ID Assignment

- Unique ID generation for each object
- Persistent identity tracking across frames
- Re-identification through motion tracking

### Detection Confidence Filtering

- Confidence threshold filtering
- Removal of low-confidence detections
- Improved tracking stability

### Track Visualization

- Bounding box visualization
- Track ID display
- Real-time tracking overlays
- Object identity monitoring

### Tracking Log Generation

- Frame-by-frame tracking records
- Track history storage
- Structured JSON output
- Position tracking analysis

---

## Workflow

1. Load video input.
2. Process each frame using YOLOv12s.
3. Filter detections using confidence threshold.
4. Convert detections into tracker-compatible format.
5. Pass detections to the OCSORT tracker.
6. Assign unique IDs to detected objects.
7. Maintain object identities across frames.
8. Draw bounding boxes and track IDs on video frames.
9. Record tracking information for each frame.
10. Save annotated output video.
11. Export tracking data to JSON format.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Tracking-by-Detection Systems
- Multi-Object Tracking
- Object Identity Management
- OCSORT Tracking Algorithm
- Detection and Tracking Integration
- Motion-Based Tracking
- Video Analytics
- Tracking Data Logging
- Real-Time Computer Vision Pipelines

---

## Results

The OCSORT tracker successfully maintained consistent object identities throughout video sequences.

### Key Observations

- Objects retained consistent track IDs across multiple frames.
- Tracking remained stable during short-term occlusions.
- The tracking-by-detection approach combined accurate detection with reliable tracking.
- Real-time visualization clearly displayed object trajectories and identities.
- Generated tracking logs provided comprehensive object movement records.
- The system effectively handled multiple objects simultaneously.

---

## Generated Output Files

### output_video.mp4

Annotated output video containing:

- Object Bounding Boxes
- Track IDs
- Real-Time Tracking Visualization
- Object Identity Labels

### tracking_log.json

Tracking log containing:

- Frame Number
- Track ID
- Bounding Box Coordinates
- Object Position Information
- Tracking Metadata

---

## Files Included

- Detection Scripts
- Tracking Scripts
- Tracking Logs
- Output Videos
- Screenshots
- Tracking Results
- Analysis Data
