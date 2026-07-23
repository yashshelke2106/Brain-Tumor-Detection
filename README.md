# Brain Tumor Classification Using CNN

**Capstone Project — Innomatics Research Labs**
**Author:** Yash Shelke | **Date:** July 2026

---

## Overview

A Convolutional Neural Network (CNN) that classifies brain MRI images into three tumor types: **Meningioma**, **Glioma**, and **Pituitary**. Includes a Streamlit web app for real-time prediction with single-image, multi-image, and folder-based batch input.

| Metric | Value |
|--------|-------|
| Test Accuracy | **82.8%** |
| Total Parameters | 8,711,971 |
| Training Images | 3,064 |
| Classes | 3 |
| Input Size | 128 x 128 x 3 |
| Framework | TensorFlow 2.21 / Keras |

---

## Dataset

| Class | Tumor Type | Images | Proportion |
|-------|-----------|--------|------------|
| 1 | Meningioma | 708 | 23.1% |
| 2 | Glioma | 1,426 | 46.5% |
| 3 | Pituitary | 930 | 30.4% |

- **Format:** 512x512 RGB images stored as pickle files (`training_data.pickle` ~2.4 GB)
- **Test Set:** 13 separate unseen MRI images for final demo
- **Class Imbalance:** 2.01x ratio (Glioma/Meningioma) — handled via computed class weights

---

## Pipeline

```
Raw Pickle Data (3,064 images)
    |
Exploratory Data Analysis
    - Class distribution charts
    - Sample image gallery
    - Pixel intensity histograms
    - Mean image per class
    - Brightness outlier detection
    |
Data Cleaning
    - 0 corrupt images, 0 NaN values
    - All shapes consistent (512x512x3)
    - All labels valid {1, 2, 3}
    |
Preprocessing
    - Resize: 512x512 -> 128x128 (bilinear interpolation)
    - Normalize: [0, 255] -> [0.0, 1.0]
    - Label encode: {1,2,3} -> one-hot vectors
    - Stratified split: 70% train / 15% val / 15% test
    - Augmentation: rotation +/-20 deg, flip, zoom +/-10%, shear +/-5%
    |
CNN Model (26 layers, 8.7M params)
    - Conv Block 1: Conv2D(32)x2 -> BN -> MaxPool -> Dropout(0.25)
    - Conv Block 2: Conv2D(64)x2 -> BN -> MaxPool -> Dropout(0.25)
    - Conv Block 3: Conv2D(128)x2 -> BN -> MaxPool -> Dropout(0.25)
    - Classifier: Flatten -> Dense(256) -> Dense(128) -> Dense(3, softmax)
    - Regularization: BatchNorm, Dropout, L2(0.001)
    |
Training
    - Optimizer: Adam (lr=0.001, reduced to 0.0005 at epoch 7)
    - Loss: Categorical Crossentropy
    - Class weights: Meningioma 1.44, Glioma 0.72, Pituitary 1.10
    - Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau
    - Best model at epoch 13 (val_accuracy: 80.9%)
    |
Evaluation
    - Test accuracy: 82.8%
    - Confusion matrix, classification report, sample predictions
    |
Deployment
    - Streamlit web app (app.py)
    - Single/multi-image upload, folder input, batch prediction
    - Tumor info: symptoms, treatment, prognosis, risk factors
```

---

## Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Meningioma | 0.73 | 0.52 | 0.61 | 107 |
| Glioma | 0.84 | 0.91 | 0.87 | 214 |
| Pituitary | 0.87 | 0.94 | 0.90 | 139 |
| **Weighted Avg** | **0.82** | **0.83** | **0.82** | **460** |

- **Pituitary:** Best performance — distinct central location makes it easiest to classify
- **Glioma:** Strong performance — largest class with most training samples
- **Meningioma:** Weakest — fewest samples + visual similarity with glioma

---

## Project Structure

```
CNN brain_tumor/
├── Brain_Tumor_CNN_Project.ipynb    # Main notebook (62 cells, full pipeline)
├── app.py                           # Streamlit web app
├── brain_tumor_cnn_model.h5         # Trained model (HDF5, git-ignored)
├── brain_tumor_cnn_model.keras      # Trained model (Keras, git-ignored)
├── brain_tumor_cnn_savedmodel/      # TF SavedModel export (git-ignored)
├── training_history.json            # Epoch-wise training metrics
├── requirements.txt                 # Python dependencies
├── setup_and_run.bat                # Windows setup script
├── brain_tumor_mri/                 # Dataset (git-ignored, ~2.4 GB)
│   └── new_dataset/
│       ├── training_data.pickle
│       ├── labels.pickle
│       └── bt_images/               # 3,064 raw JPG images
├── test_images-.../test_images/     # 13 unseen test images
├── project_outputs/                 # Pre-generated visualization PNGs
└── README.md
```

---

## Setup and Run

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd "CNN brain_tumor"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn scikit-image streamlit pillow
```

### Train the Model

```bash
jupyter notebook Brain_Tumor_CNN_Project.ipynb
```

Run all cells. This trains the CNN and saves `brain_tumor_cnn_model.h5` (~100 MB). Training takes approximately 30 minutes on CPU.

### Run the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload MRI images or load a folder for batch prediction.

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10 | Core programming language |
| TensorFlow 2.21 / Keras | CNN model building and training |
| NumPy | Array operations and preprocessing |
| Scikit-learn | Train/test split, metrics, class weights |
| Scikit-image | Image resizing with anti-aliasing |
| Matplotlib / Seaborn | Data visualization and charts |
| Streamlit | Web application for deployment |
| Pillow (PIL) | Image loading in the web app |

---

## Key Technical Decisions

| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| CNN from scratch | Transfer learning (ResNet, VGG) | Demonstrates CNN fundamentals for capstone |
| 128x128 input | 224x224 or 512x512 | 16x less compute; feasible on CPU |
| Class weights | Oversampling / SMOTE | Simpler; no duplicate artifacts |
| Adam optimizer | SGD with momentum | Faster convergence on small datasets |
| Stratified split | Random split | Preserves class proportions across splits |
| Streamlit | Flask / Gradio | Rapid prototyping; built-in file upload |

---

## Future Improvements

- Transfer Learning with EfficientNet or ResNet50 for 90%+ accuracy
- Grad-CAM visualization for model explainability
- Larger input (224x224) with GPU training
- K-fold cross-validation for robust accuracy estimates
- Normal brain class for clinical utility
- More meningioma samples to reduce class imbalance

---

## Notes

Large model artifacts (~100 MB each) and the training dataset (~2.4 GB) are intentionally excluded from this repository via `.gitignore` to keep the Git history lightweight and compatible with GitHub's file size limits.

---

*Built by Yash Shelke — Innomatics Research Labs, July 2026*
