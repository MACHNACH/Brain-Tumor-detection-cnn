# Scan Sleuth- Unveiling Brain tumors using CNN
Scan Sleuth is a deep learning-based solution for detecting and classifying brain tumors from medical images using Convolutional Neural Networks (CNN). This project aims to assist medical professionals in accurately diagnosing brain tumors.

Features
Image Classification: Detects four categories of brain scans:
No Tumor
Glioma Tumor
Meningioma Tumor
Pituitary Tumor
Data Augmentation: Improves model generalization using techniques such as rescaling, shear, zoom, and horizontal flipping.
Model Architecture:
Convolutional and MaxPooling layers for feature extraction.
Fully connected layers with Softmax activation for classification.
Performance:
Training accuracy: ~95.8%
Validation accuracy: ~89.8%
Supports data visualization for loss and accuracy trends.
Dataset
Structure:
Training images divided into the four classes.
Testing images for validation and evaluation.
Preprocessing:
Resized to 224x224 pixels.
Normalized for better gradient convergence.
Balancing:
Random oversampling for handling class imbalance.
