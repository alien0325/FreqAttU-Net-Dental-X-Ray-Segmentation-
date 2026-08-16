# FreqAttU-Net: Frequency-Enhanced Attention U-Net for Dental X-Ray Segmentation

A frequency-enhanced Attention U-Net for **binary semantic segmentation of teeth from panoramic dental X-rays**.

The project introduces a lightweight **Frequency Enhancement Module (FEM)** that uses 2D Fast Fourier Transform (FFT) features to improve the detection of tooth boundaries while retaining the spatial attention capabilities of Attention U-Net.

## Overview

Automatic tooth segmentation from panoramic dental X-rays can assist downstream applications such as tooth counting, orthodontic analysis, treatment planning, and dental diagnosis.

Traditional U-Net-based segmentation models primarily operate in the spatial domain. However, sharp tooth boundaries correspond to high-frequency components in the Fourier domain.

**FreqAttU-Net** addresses this by:

* Extracting frequency-domain features using a 2D FFT
* Applying log-magnitude compression to the frequency spectrum
* Learning frequency-aware features using a lightweight convolutional block
* Fusing frequency features with the spatial encoder
* Using attention gates to suppress irrelevant background regions
* Combining BCE and Dice loss to handle foreground-background class imbalance

The proposed model contains approximately **7.88M trainable parameters**.

---

## Key Idea

The core pipeline of the Frequency Enhancement Module is:

```text
Input X-ray
     │
     ▼
   2D FFT
     │
     ▼
Magnitude Spectrum
     │
     ▼
 Log-Magnitude
     │
     ▼
2 × Conv-BN-ReLU
     │
     ▼
Frequency Feature Map
     │
     ├──────────────► Encoder Level 1
     │
     ▼
Frequency-aware U-Net
     │
     ▼
Attention Gates
     │
     ▼
Binary Tooth Segmentation
```

The FFT-derived features are fused into the encoder at early stages, providing additional information about tooth boundaries.

---

## Architecture

FreqAttU-Net is based on a four-level Attention U-Net architecture.

### Main Components

**1. Frequency Enhancement Module (FEM)**

The FEM:

1. Computes a 2D real FFT using `torch.fft.rfft2`
2. Extracts the magnitude of the complex spectrum
3. Applies `log1p` compression
4. Resizes the frequency representation to the original spatial resolution
5. Passes it through two convolutional layers
6. Produces frequency-aware feature maps

This allows the network to explicitly exploit high-frequency information associated with tooth boundaries.

**2. Attention U-Net**

Attention gates are applied to all four skip connections. They help the decoder emphasize tooth regions while suppressing irrelevant structures such as jawbone and soft tissue.

**3. Frequency Fusion**

Frequency features are:

* Additively fused with encoder level 1 features
* Also concatenated with downsampled encoder features at level 2

The encoder uses channel widths of:

```text
32 → 64 → 128 → 256
```

with a 512-channel bottleneck.

---

## Dataset

The project uses the **Children's Dental Panoramic Radiographs Dataset** from Kaggle.

After filtering for valid image-mask pairs, the project uses **2,885 image-mask pairs**.

The data is divided into:

| Split      | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

The split uses random seed `42`.

### Dataset

[Children's Dental Panoramic Radiographs Dataset](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)

> **Note:** The dataset is not included in this repository. Download it from Kaggle and update the dataset path in the notebook if required.

---

## Preprocessing & Augmentation

Input images are converted to grayscale and resized to:

```text
512 × 512
```

Images are normalized to approximately `[-1, 1]`.

Training augmentation includes:

* Horizontal flipping
* Shift/scale/rotation
* Brightness and contrast adjustment
* Gaussian noise

The project uses a batch size of `4` and trains for `40` epochs.

---

## Loss Function

To address the strong imbalance between tooth and background pixels, a combined BCE + Dice loss is used:

```text
L = 0.5 × BCE + 0.5 × Dice Loss
```

Dice loss improves overlap-based learning while BCE provides pixel-level supervision.

---

## Training

The model is trained using:

| Parameter     |            Value |
| ------------- | ---------------: |
| Optimizer     |            AdamW |
| Learning Rate |           `1e-4` |
| Weight Decay  |           `1e-5` |
| Scheduler     | Cosine Annealing |
| Epochs        |               40 |
| Batch Size    |                4 |
| Input Size    |        512 × 512 |
| Base Filters  |               32 |
| Random Seed   |               42 |

Training was performed using an NVIDIA T4 GPU.

---

## Installation

The notebook was developed primarily for a **Kaggle GPU environment**.

Install the required packages using:

```bash
pip install "numpy<2.0" "scipy<1.13" "opencv-python-headless<4.10"
pip install torch torchvision albumentations matplotlib tqdm scikit-learn
```

Main libraries used:

* PyTorch
* Torchvision
* NumPy
* OpenCV
* Albumentations
* Matplotlib
* Scikit-learn
* tqdm

---

## Running the Project

### 1. Download the dataset

Download the Children's Dental Panoramic Radiographs dataset from Kaggle.

### 2. Open the notebook

Open:

```text
EE655_Project_Report.ipynb
```

### 3. Update the dataset path

In the configuration section:

```python
CFG = {
    'DATA_DIR': 'path/to/childrens-dental-panoramic-radiographs-dataset',
    ...
}
```

### 4. Run the notebook

Execute the notebook cells sequentially.

The notebook performs:

```text
Dataset scanning
      ↓
Exploratory visualization
      ↓
Preprocessing & augmentation
      ↓
Train/validation/test split
      ↓
FEM construction
      ↓
FreqAttU-Net construction
      ↓
Training
      ↓
Validation
      ↓
Test evaluation
      ↓
Ablation study
      ↓
Qualitative visualization
      ↓
FFT/FEM visualization
```

---

## Evaluation Metrics

The project uses:

### Dice Similarity Coefficient

```text
Dice = 2TP / (2TP + FP + FN)
```

### Intersection over Union

```text
IoU = TP / (TP + FP + FN)
```

Predictions are thresholded at `0.5` after applying the sigmoid function.

---

## Results

FreqAttU-Net was compared against an Attention U-Net baseline without the Frequency Enhancement Module.

| Model            |        Dice |         IoU |
| ---------------- | ----------: | ----------: |
| Attention U-Net  |      0.8971 |      0.8395 |
| **FreqAttU-Net** |  **0.9131** |  **0.8554** |
| Improvement      | **+0.0160** | **+0.0159** |

The proposed model improves both Dice and IoU by approximately **1.6%** over the Attention U-Net baseline.

### Ablation Study

Removing FEM while keeping the rest of the architecture unchanged reduces the Dice score from `0.9131` to `0.8971`.

The FEM adds only approximately **37K parameters**, corresponding to less than **0.5%** of the total model parameters.

---

## Qualitative Results

The model successfully segments both upper and lower dental arches and is able to exclude surrounding jawbone and soft tissue.

Representative test samples achieved Dice scores of:

```text
0.925
0.940
```

The model also demonstrated good segmentation in the presence of bright dental metal artifacts.

---

## Repository Contents

```text
├── EE655_Project_Report.ipynb
├── README.md
└── results/
    ├── training_curves.png
    ├── qualitative_results.png
    ├── fem_visualization.png
    └── comparison_table.csv
```

The complete implementation is contained in the accompanying Jupyter notebook.

---

## References

1. O. Ronneberger, P. Fischer, and T. Brox.
   **U-Net: Convolutional Networks for Biomedical Image Segmentation**, MICCAI, 2015.

2. O. Oktay et al.
   **Attention U-Net: Learning Where to Look for the Pancreas**, MIDL, 2018.

3. L. Chi, B. Jiang, and Y. Mu.
   **Fast Fourier Convolution**, NeurIPS, 2020.

4. truthisneverlinear.
   **Children's Dental Panoramic Radiographs Dataset**, Kaggle, 2023.

---

## Team

| Name                | Roll No. |
| ------------------- | -------: |
| Amey Dikshit        |   230120 |
| Anjan Das           |   230149 |
| Atit Kumar Satsangi |   230241 |
| Deepak Kumar Meena  |   230349 |
| Piyush Kumar        |   230752 |

The project was completed as part of **EE655 – Computer Vision & Deep Learning**.

---

## Acknowledgements

We thank the authors of the Children's Dental Panoramic Radiographs Dataset for making the dataset publicly available.

We also thank **Professor Koteswar Rao Jerripothula** for the foundational Computer Vision and Deep Learning concepts taught in EE655 that contributed to this project.
