# Task 02 - Morphological Operations & Image Transformations

## Overview

This project focuses on advanced image enhancement and geometric transformation techniques using OpenCV. The implementation explores morphological operations for image refinement, noise reduction methods, contrast enhancement techniques, and perspective correction for distorted images.

The objective of this task was to understand how image quality can be improved through preprocessing techniques and how geometric transformations can be used to correct perspective distortions in real-world applications.

---

## Technologies Used

- Python
- OpenCV (cv2)
- NumPy

---

## Features Implemented

### Morphological Operations

- Erosion
- Dilation
- Opening
- Closing
- Binary image refinement
- Noise removal and gap filling

### Noise Reduction

- Bilateral Filtering
- Edge-preserving image smoothing
- Uniform region enhancement

### Contrast Enhancement

- Histogram Equalization
- Global contrast improvement
- Intensity redistribution

### CLAHE (Adaptive Histogram Equalization)

- Localized contrast enhancement
- Contrast Limited Adaptive Histogram Equalization
- Prevention of noise over-amplification

### Perspective Transformation

- Interactive corner point selection
- Transformation matrix computation
- Perspective correction
- Geometric image rectification

---

## Workflow

1. Read the input image.
2. Convert the image to grayscale.
3. Apply binary thresholding.
4. Create a 5×5 rectangular kernel.
5. Perform erosion to reduce object boundaries.
6. Perform dilation to expand object regions.
7. Apply opening operation to remove small noise.
8. Apply closing operation to fill gaps and holes.
9. Perform bilateral filtering for edge-preserving smoothing.
10. Apply histogram equalization for contrast enhancement.
11. Apply CLAHE for localized contrast improvement.
12. Select four corner points interactively.
13. Compute perspective transformation matrix.
14. Warp the image to obtain a corrected output.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Morphological image processing
- Binary image refinement techniques
- Noise reduction methods
- Edge-preserving filtering
- Image contrast enhancement
- Adaptive histogram equalization
- Geometric image transformations
- Perspective correction and rectification

---

## Results

The system successfully:

- Removed unwanted noise from binary images
- Filled gaps and improved object structures
- Preserved edges while reducing image noise
- Enhanced image contrast using Histogram Equalization and CLAHE
- Corrected perspective distortions in skewed images
- Generated cleaner images suitable for OCR and document processing applications

### Key Observations

- Morphological operations effectively improved binary image quality.
- Bilateral filtering preserved edges better than conventional smoothing techniques.
- CLAHE produced superior localized contrast enhancement compared to standard histogram equalization.
- Perspective transformation successfully corrected distorted images for further analysis.

---

## Applications

- Document Scanning
- OCR Preprocessing
- Image Enhancement
- Digital Image Restoration
- Computer Vision Pipelines
- Perspective Correction Systems

---

## Files Included

- Source Code
- Input Images
- Output Images
- Screenshots
