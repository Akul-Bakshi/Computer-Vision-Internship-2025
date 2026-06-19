# Task 07 - Automatic Number Plate Recognition (ANPR)

## Overview

This project focuses on developing a complete Automatic Number Plate Recognition (ANPR) system by combining object detection, object tracking, image rectification, and Optical Character Recognition (OCR). The implementation creates an end-to-end pipeline capable of detecting vehicles, identifying license plates, tracking vehicles across video frames, extracting plate regions, correcting perspective distortions, and recognizing license plate text.

The objective of this task was to understand how multiple computer vision techniques can be integrated into a practical real-world application for automated vehicle identification.

---

## Technologies Used

- Python
- OpenCV
- YOLO
- OCSORT
- Tesseract OCR
- NumPy
- JSON

---

## Features Implemented

### Vehicle Detection

- Vehicle localization using YOLO
- Bounding box generation
- Confidence-based filtering
- Multi-vehicle support

### License Plate Detection

- Custom YOLO-based plate detection
- Plate localization within vehicle regions
- Region extraction for further processing

### Object Tracking

- OCSORT tracker integration
- Persistent vehicle ID assignment
- Multi-frame vehicle tracking
- Vehicle identity maintenance

### ROI Extraction

- Vehicle image cropping
- License plate image extraction
- Track-based image organization
- Automated region saving

### Perspective Correction

- License plate contour detection
- Four-point perspective transformation
- Plate rectification
- Skew correction

### Image Preprocessing

- Bilateral Filtering
- CLAHE Enhancement
- Edge Detection
- Morphological Operations
- OCR-ready image generation

### OCR Integration

- Tesseract OCR
- License plate text extraction
- Text cleanup and validation
- Plate number recognition

### Detection Logging

- Timestamp recording
- Detection metadata storage
- File reference management
- Recognition result logging

---

## Workflow

1. Load video input.
2. Detect vehicles using YOLO.
3. Detect license plates within vehicle regions.
4. Track vehicles using OCSORT.
5. Assign persistent IDs to detected vehicles.
6. Extract and save vehicle crops.
7. Extract and save license plate crops.
8. Detect plate contours and boundaries.
9. Apply perspective correction using four-point transformation.
10. Generate rectified license plate images.
11. Apply image preprocessing techniques.
12. Perform OCR using Tesseract.
13. Extract and clean recognized text.
14. Update detection logs with recognized plate numbers.
15. Store all outputs for future analysis and auditing.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Automatic Number Plate Recognition (ANPR)
- Multi-Stage Computer Vision Pipelines
- Vehicle Detection Systems
- License Plate Localization
- Object Tracking
- Image Rectification
- Perspective Transformation
- OCR-Based Text Recognition
- Data Logging and Audit Systems
- Real-World Computer Vision Applications

---

## Results

The ANPR system successfully detected vehicles, localized license plates, tracked vehicles across frames, and extracted license plate information from video sequences.

### Key Observations

- Vehicle tracking maintained consistent IDs throughout video sequences.
- License plate localization accurately identified plate regions.
- Perspective correction significantly improved OCR readability.
- Image preprocessing enhanced character visibility before OCR.
- Tesseract OCR successfully extracted plate text with reasonable accuracy.
- The complete pipeline produced structured outputs suitable for real-world deployment.
- Detection logs provided a detailed audit trail of all recognized vehicles and license plates.

---

## Generated Output Files

### output_detection.mp4

Annotated output video containing:

- Vehicle Bounding Boxes
- License Plate Bounding Boxes
- Vehicle Track IDs
- Detection Labels

### vehicle_idX.jpg

Cropped vehicle images generated for each tracked vehicle.

### plate_idX.jpg

Extracted license plate images from detected vehicles.

### warped_plate_idX.jpg

Perspective-corrected license plate images used for OCR processing.

### detection_log.json

Comprehensive detection log containing:

- Timestamps
- Vehicle IDs
- Bounding Box Coordinates
- File References
- Extracted Plate Numbers
- Detection Metadata

### best.pt

Custom-trained YOLO model used for license plate detection.

### output_files.zip

Archive containing generated outputs, cropped images, processed plates, and detection logs.

---

## Files Included

- Detection Scripts
- Tracking Scripts
- OCR Scripts
- Detection Logs
- Processed Images
- Output Videos
- Screenshots
- Trained Model Files
- Recognition Results
