# Task 03 - Deep Learning with CNNs

## Overview

This project introduces deep learning concepts through the implementation of Convolutional Neural Networks (CNNs) for image classification. The task focuses on understanding neural network architecture, training deep learning models, evaluating performance, and saving trained models for future inference.

The objective of this task was to build a CNN capable of recognizing handwritten digits from the MNIST dataset while gaining practical experience with model training, evaluation, and deployment workflows.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib

---

## Features Implemented

### CNN Architecture

- Convolutional Layers (Conv2D)
- Max Pooling Layers
- Fully Connected Dense Layers
- Softmax Classification Output

### Dataset Handling

- MNIST Handwritten Digit Dataset
- Training and Testing Data Split
- Image Reshaping for CNN Input
- Label Encoding

### Data Preprocessing

- Pixel Value Normalization
- Image Reshaping
- One-Hot Encoding of Labels
- Dataset Preparation for Training

### Model Training

- Adam Optimizer
- Categorical Cross-Entropy Loss
- Validation Split
- Multi-Epoch Training

### Model Evaluation

- Accuracy Calculation
- Loss Analysis
- Classification Report
- Confusion Matrix Generation

### Model Persistence

- Saving Trained Models
- Loading Saved Models
- Performing Inference on Test Data

---

## Workflow

1. Load the MNIST handwritten digit dataset.
2. Normalize pixel values for improved model convergence.
3. Reshape images into CNN-compatible format.
4. Convert labels into categorical format.
5. Build the CNN architecture using convolutional and pooling layers.
6. Compile the model using Adam optimizer and categorical cross-entropy loss.
7. Train the model using training data and validation split.
8. Save the trained model as `cnn_model.h5`.
9. Load the saved model for testing.
10. Generate predictions on test data.
11. Evaluate performance using accuracy, loss, classification reports, and confusion matrices.
12. Save evaluation results for further analysis.

---

## Learning Outcomes

Through this task, the following concepts were explored:

- Fundamentals of Deep Learning
- Convolutional Neural Networks (CNNs)
- Image Classification Techniques
- Dataset Preprocessing
- Model Training and Validation
- Performance Evaluation Metrics
- Confusion Matrix Analysis
- Model Saving and Loading
- Deep Learning Deployment Workflow

---

## Results

The CNN model successfully learned to classify handwritten digits from the MNIST dataset with high accuracy.

### Key Observations

- The model achieved strong classification performance on unseen test data.
- Training accuracy improved consistently across epochs.
- The CNN architecture effectively learned spatial features from handwritten digit images.
- Classification reports provided detailed precision, recall, and F1-scores for each digit class.
- Confusion matrix analysis revealed prediction strengths and common misclassification patterns.

---

## Generated Output Files

### cnn_model.h5

Saved trained CNN model containing learned weights and network architecture. The model can be loaded directly for inference without requiring retraining.

### cnn_evaluation_output.txt

Comprehensive evaluation report containing:

- Test Accuracy
- Test Loss
- Classification Report
- Precision Scores
- Recall Scores
- F1-Scores
- Confusion Matrix

---

## Files Included

- Training Script (`train_cnn.py`)
- Testing Script (`test_cnn.py`)
- Saved CNN Model (`cnn_model.h5`)
- Evaluation Report (`cnn_evaluation_output.txt`)
- Screenshots
- Output Results
