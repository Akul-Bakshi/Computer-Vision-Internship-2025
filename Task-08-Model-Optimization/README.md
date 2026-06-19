# Task 08 - Model Optimization and Deployment

## Overview

This project focuses on optimizing computer vision models and inference pipelines for deployment in real-world applications. The implementation includes model conversion to ONNX format, inference acceleration, frame-skipping strategies, memory optimization, and performance benchmarking.

The objective of this task was to improve the speed and efficiency of object detection and license plate recognition systems while maintaining detection accuracy, making the pipeline suitable for near real-time deployment scenarios.

---

## Technologies Used

- Python
- YOLOv12
- ONNX
- OpenCV
- NumPy
- JSON

---

## Features Implemented

### Model Optimization

- Conversion of YOLO models to ONNX format
- Faster inference compared to PyTorch models
- Deployment-ready model generation
- Reduced runtime overhead

### ONNX Inference

- Vehicle detection using optimized ONNX models
- License plate detection using ONNX models
- Improved inference performance
- Production-oriented deployment pipeline

### Frame Skipping Strategy

- Processing every alternate frame
- Reduced computational load
- Increased throughput
- Maintained detection consistency

### Detection Buffering

- Best-frame selection for tracked vehicles
- Elimination of redundant image storage
- Improved data quality
- Efficient result management

### Memory Optimization

- Efficient image cropping
- Reduced memory usage
- Optimized storage workflow
- Resource-conscious processing

### Performance Benchmarking

- Processing time measurement
- FPS calculation
- Pipeline performance analysis
- Optimization impact evaluation

### Performance Logging

- Runtime statistics collection
- FPS reporting
- Processing metrics generation
- Benchmark result storage

---

## Workflow

1. Load trained YOLO models.
2. Convert PyTorch models to ONNX format.
3. Load optimized ONNX models for inference.
4. Process video frames using optimized detectors.
5. Apply frame-skipping strategy to reduce computation.
6. Detect vehicles and license plates.
7. Track detections across frames.
8. Buffer and store only the best detections.
9. Measure processing time for each frame.
10. Calculate average FPS.
11. Save optimized outputs and detection logs.
12. Generate performance metrics for analysis.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Model Optimization Techniques
- ONNX Model Conversion
- Deployment-Oriented Inference
- Performance Benchmarking
- Resource Optimization
- Real-Time Computer Vision Systems
- FPS Analysis
- Memory Management
- Production Pipeline Design

---

## Results

The optimized pipeline achieved significant performance improvements while preserving detection quality.

### Key Observations

- ONNX conversion reduced inference latency compared to native PyTorch execution.
- Frame skipping effectively improved throughput without significantly affecting detection performance.
- Detection buffering ensured only high-quality vehicle captures were stored.
- Memory optimization reduced unnecessary storage operations.
- Performance benchmarking provided measurable indicators of optimization success.
- The optimized pipeline achieved an average processing speed of **18.78 FPS**, making it suitable for near real-time applications.

---

## Generated Output Files

### yolo12s.onnx

Optimized ONNX model used for vehicle detection.

### best.onnx

Optimized ONNX model used for license plate detection.

### output_optimized.mp4

Output video generated using the optimized inference pipeline.

Features included:

- Vehicle Detection
- License Plate Detection
- Optimized Processing Pipeline
- Real-Time Visualization

### detection_log.json

Detection log containing:

- Timestamps
- Detection Metadata
- File References
- Vehicle Information

### vehicle_idX.jpg

Best-quality cropped vehicle images selected by the buffering system.

### vehicle_plate_images.zip

Archive containing:

- Vehicle Crops
- License Plate Crops
- Detection Results

### Performance Metrics

Benchmark results including:

- Average FPS: **18.78**
- Processing Speed Analysis
- Optimization Performance Statistics

---

## Files Included

- ONNX Conversion Scripts
- Optimized Inference Scripts
- Performance Benchmark Results
- Detection Logs
- Output Videos
- Screenshots
- Optimized Models
- Evaluation Metrics
