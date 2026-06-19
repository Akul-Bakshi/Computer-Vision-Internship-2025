# Task 01 - OpenCV Basics

## Overview

This project introduces fundamental computer vision concepts using OpenCV. The implementation covers image processing, video processing, color space manipulation, filtering techniques, edge detection, contour analysis, and region extraction.

The objective of this task was to build a strong foundation in computer vision workflows and understand how images and video streams can be processed programmatically.

---

## Technologies Used

- Python
- OpenCV (cv2)
- NumPy

---

## Features Implemented

### Image Processing

- Load images using OpenCV
- Save processed images
- Perform image transformations

### Color Space Conversion

- RGB Conversion
- Grayscale Conversion
- HSV Conversion

### Image Filtering

- Gaussian Blur (7×7 Kernel)
- Median Blur (5×5 Kernel)

### Edge Detection

- Sobel Edge Detection (X Direction)
- Sobel Edge Detection (Y Direction)
- Combined Gradient Detection
- Canny Edge Detection

### Contour Detection

- Detection of object boundaries
- Contour visualization
- Area-based contour filtering

### Region of Interest (ROI) Extraction

- Bounding box generation
- ROI extraction from detected contours
- Filtering based on contour area

### Real-Time Video Processing

- Webcam capture using OpenCV
- Live contour detection
- Real-time image processing pipeline

---

## Workflow

1. Read input image.
2. Convert image into multiple color spaces.
3. Apply Gaussian and Median filtering.
4. Detect edges using Sobel and Canny operators.
5. Extract contours from detected edges.
6. Filter contours based on area threshold.
7. Draw contours and bounding boxes.
8. Extract Regions of Interest (ROI).
9. Apply similar processing pipeline on live webcam feed.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Fundamentals of digital image processing
- Color space representations
- Noise reduction techniques
- Edge detection algorithms
- Contour analysis
- Object localization using bounding boxes
- Real-time computer vision using webcam streams

---

## Results

The system successfully:

- Processed static images
- Detected object boundaries
- Extracted Regions of Interest (ROI)
- Performed real-time contour detection on webcam feeds

Example outputs and screenshots are available in this project folder.

---

## Project Structure

```text
Task-01-OpenCV-Basics
│
├── README.md
├── source_code.py
├── input_images/
├── output_images/
└── screenshots/
```

---

## Files Included

- Source Code
- Input Images
- Output Images
- Screenshots
