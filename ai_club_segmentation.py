
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import sys


print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# Mount drive (keep your data loading as is)
from google.colab import drive
drive.mount('/content/drive')
print("Starting script execution...")


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
print("Random seeds set for reproducibility")


try:
   print("Attempting to load data...")
   my_path = '/content/drive/MyDrive/Colab Notebooks/AIClubData'
   train_images = np.load(os.path.join(my_path, 'scan_fall_2019.npy')).astype(np.float32)
   train_masks = np.load(os.path.join(my_path, 'labels_fall_2019.npy')).astype(np.int8)
   print(f"Data loaded successfully!")
   print(f"Images shape: {train_images.shape}, dtype: {train_images.dtype}")
   print(f"Masks shape: {train_masks.shape}, dtype: {train_masks.dtype}")


   # Check for NaNs or abnormal values
   print(f"Images - Min: {np.min(train_images)}, Max: {np.max(train_images)}, NaNs: {np.isnan(train_images).any()}")
   print(f"Masks - Min: {np.min(train_masks)}, Max: {np.max(train_masks)}, NaNs: {np.isnan(train_masks).any()}")
   print(f"Unique mask values: {np.unique(train_masks)}")  # Should match the number of classes
except Exception as e:
   print(f"Error loading data: {e}")
   raise


# Normalize and reshape
print("Normalizing and reshaping data...")
train_images = train_images / 255.0
train_images = train_images.reshape(-1, 256, 256, 1)
train_masks = train_masks.reshape(-1, 256, 256)
print(f"After normalization - Images Min: {np.min(train_images)}, Max: {np.max(train_images)}")
print(f"After reshaping - Images shape: {train_images.shape}, Masks shape: {train_masks.shape}")


# Hold out the last image as test
print("Separating test image...")
test_image = train_images[-1:]
test_mask = train_masks[-1:]
train_images = train_images[:-1]
train_masks = train_masks[:-1]
print(f"Test set: {test_image.shape[0]} image")
print(f"Training set: {train_images.shape[0]} images")


# Debug: Display sample images
print("Displaying sample images and masks for verification...")
plt.figure(figsize=(12, 6))
for i in range(min(3, train_images.shape[0])):
   plt.subplot(2, 3, i+1)
   plt.imshow(train_images[i, :, :, 0], cmap='gray')
   plt.title(f'Train Image {i}')
   plt.axis('off')


   plt.subplot(2, 3, i+4)
   plt.imshow(train_masks[i], cmap='tab10')
   plt.title(f'Train Mask {i}')
   plt.axis('off')
plt.tight_layout()
plt.show()


# Patch extraction
print("Extracting image patches and corresponding labels...")


def extract_patches(images, masks=None, patch_size=16, stride=16, sample_rate=1.0):
   """
   Extract patches from images (and corresponding center pixel labels from masks).


   Parameters:
     images: Array of images with shape (n_images, H, W, C)
     masks:  Array of masks with shape (n_images, H, W) or None
     patch_size: Size of the patch (patch_size x patch_size)
     stride: Step size between patches
     sample_rate: Fraction (between 0 and 1) of patches to sample (1.0 means take all patches)


   Returns:
     patches: Array of extracted patches.
     patch_labels: Array of labels for the patches (if masks is provided).
     positions: Array of positions (img index, h_start, w_start) for each patch.
   """
   patches = []
   positions = []
   patch_labels = []
   n_images = images.shape[0]
   h, w = images.shape[1], images.shape[2]
   n_h = (h - patch_size) // stride + 1
   n_w = (w - patch_size) // stride + 1


   for img_idx in range(n_images):
       for i in range(n_h):
           for j in range(n_w):
               # Sample the patch with probability sample_rate.
               if np.random.rand() > sample_rate:
                   continue
               h_start, w_start = i * stride, j * stride
               patch = images[img_idx, h_start:h_start+patch_size, w_start:w_start+patch_size, :]
               patches.append(patch)
               positions.append((img_idx, h_start, w_start))
               if masks is not None:
                   center_h, center_w = h_start + patch_size // 2, w_start + patch_size // 2
                   label = masks[img_idx, center_h, center_w]
                   patch_labels.append(label)


   patches = np.array(patches)
   positions = np.array(positions)
   if masks is not None:
       patch_labels = np.array(patch_labels)
       return patches, patch_labels, positions
   else:
       return patches, positions


# Train patch extraction
try:
   print("Extracting training patches...")
   train_patches, train_patch_labels, train_positions = extract_patches(train_images, train_masks)
   print("Converting patch labels to one-hot encoding...")
   train_patch_labels = tf.keras.utils.to_categorical(train_patch_labels, num_classes=4)
   print(f"One-hot encoded labels shape: {train_patch_labels.shape}")


   # Flatten patches for FFNN input
   print("Flattening patches for FFNN input...")
   train_patches_flat = train_patches.reshape(train_patches.shape[0], -1)
   print(f"Flattened patches shape: {train_patches_flat.shape}")


   # Check memory usage
   print(f"Memory usage - Train patches: {train_patches_flat.nbytes / (1024 * 1024):.2f} MB")
   print(f"Memory usage - Train labels: {train_patch_labels.nbytes / (1024 * 1024):.2f} MB")
except Exception as e:
   print(f"Error in patch extraction: {e}")
   raise


# Test patch extraction
try:
   print("Extracting test patches...")
   test_patches, test_positions = extract_patches(test_image)
   test_patches_flat = test_patches.reshape(test_patches.shape[0], -1)
   print(f"Test patches shape: {test_patches.shape}")
   print(f"Flattened test patches shape: {test_patches_flat.shape}")
except Exception as e:
   print(f"Error in test patch extraction: {e}")
   raise


# Split training data into train and validation sets
print("Splitting data into train and validation sets...")
val_size = int(train_patches_flat.shape[0] * 0.1)
indices = np.arange(train_patches_flat.shape[0])
np.random.shuffle(indices)
val_indices = indices[:val_size]
train_indices = indices[val_size:]


val_patches_flat = train_patches_flat[val_indices]
val_patch_labels = train_patch_labels[val_indices]
train_patches_flat = train_patches_flat[train_indices]
train_patch_labels = train_patch_labels[train_indices]


print(f"Training patches: {train_patches_flat.shape[0]}")
print(f"Validation patches: {val_patches_flat.shape[0]}")


# FFNN model
print("Creating FFNN model...")


def create_ffn_model(input_size, num_classes=4):
   """Create a feedforward neural network model"""
   print(f"  Creating FFNN with input size {input_size} and {num_classes} output classes")
   model = models.Sequential([
       layers.Input(shape=(input_size,)),
       layers.Dense(512, activation='relu', name='dense_1'),
       layers.BatchNormalization(name='bn_1'),
       layers.Dropout(0.3, name='dropout_1'),
       layers.Dense(256, activation='relu', name='dense_2'),
       layers.BatchNormalization(name='bn_2'),
       layers.Dropout(0.3, name='dropout_2'),
       layers.Dense(128, activation='relu', name='dense_3'),
       layers.BatchNormalization(name='bn_3'),
       layers.Dropout(0.2, name='dropout_3'),
       layers.Dense(64, activation='relu', name='dense_4'),
       layers.BatchNormalization(name='bn_4'),
       layers.Dense(num_classes, activation='softmax', name='output')
   ])
   return model


try:
   # Create FFNN model
   input_size = train_patches_flat.shape[1]
   print(f"Input size (flattened patch): {input_size}")
   model = create_ffn_model(input_size)
   model.summary()  # Print model summary for debugging
   print("Model created successfully")
except Exception as e:
   print(f"Error creating model: {e}")
   raise


# Compile the model
print("Compiling model...")
try:
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   print("Model compiled successfully")
except Exception as e:
   print(f"Error compiling model: {e}")
   raise


# Setup callbacks
print("Setting up callbacks...")
callbacks = [
   EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
   ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]


# Train model
print("Starting model training...")
start_time = time.time()
try:
   history = model.fit(
       train_patches_flat, train_patch_labels,
       validation_data=(val_patches_flat, val_patch_labels),
       epochs=1,  # Increased from 5 in your original code
       batch_size=128,
       callbacks=callbacks,
       verbose=1
   )


   training_time = time.time() - start_time
   print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
except Exception as e:
   print(f"Error during training: {e}")
   raise


# Predict on test patches
print("Generating predictions for test patches...")
try:
   test_patch_preds = model.predict(test_patches_flat)
   print(f"Prediction shape: {test_patch_preds.shape}")
   print(f"Prediction values - Min: {np.min(test_patch_preds)}, Max: {np.max(test_patch_preds)}")
   test_patch_preds = np.argmax(test_patch_preds, axis=-1)
   print(f"Predicted classes shape: {test_patch_preds.shape}")
   print(f"Unique predicted classes: {np.unique(test_patch_preds, return_counts=True)}")
except Exception as e:
   print(f"Error during prediction: {e}")
   raise


# Reconstruct predictions into full image
print("Reconstructing full segmentation mask from patch predictions...")


def reconstruct_from_patches(patch_predictions, positions, image_shape, n_images, patch_size=16):
   """Reconstruct full segmentation masks from patch predictions"""
   print(f"  Reconstructing {n_images} images with shape {image_shape}")
   reconstructed = np.zeros((n_images, image_shape[0], image_shape[1]), dtype=np.int32)
   count_maps = np.zeros((n_images, image_shape[0], image_shape[1]), dtype=np.int32)


   print(f"  Processing {len(patch_predictions)} patch predictions")
   for i, (img_idx, h_start, w_start) in enumerate(positions):
       pred = patch_predictions[i]
       patch_pred = np.ones((patch_size, patch_size), dtype=np.int32) * pred
       reconstructed[img_idx, h_start:h_start+patch_size, w_start:w_start+patch_size] += patch_pred
       count_maps[img_idx, h_start:h_start+patch_size, w_start:w_start+patch_size] += 1


   # Avoid division by zero
   count_maps = np.maximum(count_maps, 1)
   reconstructed = reconstructed / count_maps
   result = np.round(reconstructed).astype(np.int32)
   print(f"  Reconstruction complete with shape {result.shape}")
   return result


try:
   # Reconstruct held-out test image
   test_pred_mask = reconstruct_from_patches(
       test_patch_preds, test_positions, (256, 256), n_images=1
   )[0]
   print(f"Reconstructed mask shape: {test_pred_mask.shape}")
   print(f"Unique values in reconstructed mask: {np.unique(test_pred_mask, return_counts=True)}")
except Exception as e:
   print(f"Error during reconstruction: {e}")
   raise


# Save the model
print("Saving model...")
try:
   model.save('/content/drive/MyDrive/Colab Notebooks/AIClubData/ffnn_segmentation_model.h5')
   print("Model saved successfully")
except Exception as e:
   print(f"Error saving model: {e}")


# Visualize results
print("Visualizing results...")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(test_image[0, :, :, 0], cmap='gray')
plt.title('Test Image')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(test_mask[0], cmap='tab10')
plt.title('Ground Truth Mask')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(test_pred_mask, cmap='tab10')
plt.title('Predicted Mask')
plt.axis('off')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Colab Notebooks/AIClubData/ffnn_segmentation_results.png')
plt.show()


# Plot training history
print("Plotting training history...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Colab Notebooks/AIClubData/ffnn_training_history.png')


