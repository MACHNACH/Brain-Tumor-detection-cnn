# Brain-Tumor-detection-cnn
from google.colab import drive
drive.mount('/content/drive')

import os

# Verify if the directory exists
data_dir = '/content/drive/MyDrive/major/BrainTumor'
if os.path.exists(data_dir):
    print("Directory exists")
    print(os.listdir(data_dir))  # List all categories or files
else:
    print("Directory does not exist")

  import os

# Specify the root directory you want to start from
root_dir = '/content/drive/MyDrive/major'

# Traverse the directory tree
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        print(f"Reading file: {file_path}")

        # Example: Read text files
        if file.endswith('.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
                print(content)  

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset directories
train_dir = '/content/drive/MyDrive/major/BrainTumor/Training'
test_dir = '/content/drive/MyDrive/major/BrainTumor/Testing'

# Check categories in train and test directories
categories = os.listdir(train_dir)
print("Categories:", categories)

# Use ImageDataGenerator for preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2  # Set aside 20% of training data for validation
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create training, validation, and test data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Validation data
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
#DATA CLEANING
import os
from PIL import Image

def check_and_remove_corrupted_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()  # Verify that the image is valid
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image found and removed: {file}")
                os.remove(os.path.join(root, file))  # Remove corrupted image

# Clean training and testing directories
train_dir = '/content/drive/MyDrive/major/BrainTumor/Training'
test_dir = '/content/drive/MyDrive/major/BrainTumor/Testing'

check_and_remove_corrupted_images(train_dir)
check_and_remove_corrupted_images(test_dir)
#DATA PREPROCESSING
# RESIZING IMAGES
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Resize and preprocess images
image_size = (224, 224)  # Set the image size to 224x224 for input into CNN

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    rotation_range=20,  # Augment data with rotations
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    zoom_range=0.2,  # Apply zoom augmentation
    horizontal_flip=True,  # Flip images horizontally
    validation_split=0.2  # Split training data into training and validation sets
)

# For testing data, only apply rescaling (no augmentations)
test_datagen = ImageDataGenerator(rescale=1./255)
