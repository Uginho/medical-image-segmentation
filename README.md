# medical-image-segmentation

# Medical Image Segmentation – AI Club Project

This project implements a semantic segmentation pipeline using a patch-based feedforward neural network (FFNN) on grayscale medical scan images. Developed as part of the Wake Forest University AI Club, the system classifies center pixels of 16×16 patches and reconstructs full pixel-level segmentation masks from model predictions.

---

## Project Overview

- Built an end-to-end image segmentation system using TensorFlow, NumPy, and Matplotlib
- Converted 256×256 medical scans into 16×16 labeled patches for center-pixel classification
- Achieved 38% accuracy improvement over baseline via dropout, batch normalization, and learning rate scheduling
- Reconstructed full-resolution segmentation masks from patch-level predictions using custom aggregation logic
- Visualized ground truth and predicted masks to verify model performance

---

## Repository Structure

.
├── ai_club_segmentation.ipynb # Full training pipeline (preprocessing, model, eval)
├── scans.npy # Grayscale scan data (input images)
├── labels.npy # Corresponding segmentation masks
├── project_instructions.pdf # Assignment description from professor
├── segmentation_results.png # Sample output image (if generated)
├── ffnn_segmentation_model.h5 # Trained model weights (optional)
├── requirements.txt # (Optional) List of dependencies
└── README.md # You're here!

yaml
Copy
Edit

---

## Technologies Used

- **Language:** Python 3.x
- **Libraries:** TensorFlow/Keras, NumPy, Matplotlib, Pandas
- **Tools:** Google Colab, GitHub

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Uginho/medical-image-segmentation.git
   cd medical-image-segmentation
