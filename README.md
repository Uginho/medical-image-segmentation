# Medical Image Segmentation – AI Club Project

This project implements a semantic segmentation pipeline using a patch-based feedforward neural network (FFNN) on grayscale medical scan images. Built as part of the Wake Forest AI Club, the model learns to classify the center pixel of 16×16 image patches and reconstruct full segmentation masks from patch-level predictions.

---

## Highlights
- Built an end-to-end image segmentation system using TensorFlow, NumPy, and Matplotlib
- Converted 256×256 medical scans into labeled 16×16 patches for center-pixel classification
- Achieved 38% accuracy improvement over baseline via dropout, batch normalization, and learning rate tuning
- Reconstructed full-resolution segmentation masks from model predictions using custom averaging logic

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab (runtime)

---

## Repository Contents
- `ai_club_segmentation.ipynb`: Full pipeline including preprocessing, patch extraction, model training, prediction, and reconstruction
- `project_instructions.pdf`: Assignment specification provided by faculty

> **Note:** `scans.npy` and `labels.npy` (the original dataset files) are not included due to size and access constraints. See below for guidance.

---

## How to Use

1. Clone or download this repo  
2. Open `ai_club_segmentation.ipynb` in Google Colab or your local Jupyter environment  
3. Replace missing data files (`scans.npy` and `labels.npy`) with your own, or simulate arrays for testing:
   ```python
   # Example placeholder
   dummy_images = np.random.rand(10, 256, 256, 1).astype(np.float32)
   dummy_masks = np.random.randint(0, 4, size=(10, 256, 256)).astype(np.int8)
