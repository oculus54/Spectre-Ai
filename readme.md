# Spectre AI - AI vs Real Image Detection

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge\&logo=Python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge\&logo=PyTorch\&logoColor=white)
![Torchvision](https://img.shields.io/badge/-Torchvision-EE4C2C?style=for-the-badge)
![timm](https://img.shields.io/badge/-timm-0099CC?style=for-the-badge)
![ResNet50](https://img.shields.io/badge/-ResNet50-4B5563?style=for-the-badge)
![Vision Transformer](https://img.shields.io/badge/-Vision_Transformer-6E57E0?style=for-the-badge)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=for-the-badge\&logo=NVIDIA\&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge\&logo=NumPy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge)
![Pillow](https://img.shields.io/badge/-Pillow-3776AB?style=for-the-badge)
![Jupyter Notebook](https://img.shields.io/badge/-Jupyter-F37626?style=for-the-badge\&logo=Jupyter\&logoColor=white)

---

## Overview

**Spectre AI** is a deep learning-based image forensics framework designed to classify images as **AI-Generated** or **Real**. It combines the local feature extraction capabilities of **ResNet50** with the global contextual understanding of a **Vision Transformer (ViT-B/16)**, resulting in a robust hybrid architecture for binary image classification.

The repository contains two versions of the model:

* **Spectre V1:** Hybrid ResNet50 + Vision Transformer (ViT-B/16)
* **Spectre V2:** ResNet50-based classifier

---

## Features

* AI vs Real image classification
* Hybrid CNN + Vision Transformer architecture
* CUDA-enabled inference
* Confidence score visualization
* Grad-CAM explainability
* Fast single-image inference
* PyTorch and timm implementation

---

## Model Architecture

```text
                 Input Image (224×224)
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
      ResNet50                    Vision Transformer
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
             Feature Concatenation
                       ▼
            Fully Connected Fusion
                       ▼
             Binary Classification
```

---

## Project Structure

```text
Spectre/
│
├── code/
│   ├── model.py
│   └── model2.py
│
├── weights/
│   ├── Spectre.pth
│   └── SpectreV2.pth
│
├── images/
├── requirements.txt
├── Spectre_V1.ipynb
├── Spectre_V2.ipynb
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Spectre.git
cd Spectre
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install torch torchvision timm matplotlib pillow
```

---

## Usage

1. Place the trained model (`Spectre.pth`) inside the **weights/** directory.
2. Place the input image inside the **images/** directory.
3. Run the inference script:

```bash
cd code
python model.py
```

---

## Performance

### Spectre V1 (Hybrid ResNet50 + ViT)

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **99.05%** |
| Macro Precision | **98.77%** |
| Macro Recall    | **98.63%** |
| Macro F1-Score  | **98.70%** |

### Spectre V2 (ResNet50)

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **98.71%** |
| Macro Precision | **98.67%** |
| Macro Recall    | **97.81%** |
| Macro F1-Score  | **98.23%** |

**Test Samples:** **2,100 Images**

---

## Classification Report

### Spectre V1

| Class        |  Precision |     Recall |   F1-Score |
| ------------ | ---------: | ---------: | ---------: |
| AI Generated | **98.22%** | **97.83%** | **98.03%** |
| Real         | **99.31%** | **99.43%** | **99.37%** |

### Spectre V2

| Class        |  Precision |     Recall |   F1-Score |
| ------------ | ---------: | ---------: | ---------: |
| AI Generated | **98.59%** | **96.06%** | **97.31%** |
| Real         | **98.75%** | **99.56%** | **99.16%** |

---

## Image Preprocessing

* Resize to **224 × 224**
* Convert image to tensor
* Normalize using ImageNet statistics

```python
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Inference Output

The inference script provides:

* Predicted class
* Confidence score
* AI-generated probability
* Real-image probability
* Inference time
* Probability bar chart
* Grad-CAM visualization

---

## Models

### Spectre V1

| Component            | Specification                 |
| -------------------- | ----------------------------- |
| CNN Backbone         | ResNet50                      |
| Transformer Backbone | Vision Transformer (ViT-B/16) |
| Framework            | PyTorch                       |
| Library              | timm                          |
| Task                 | Binary Image Classification   |

### Spectre V2

| Component    | Specification               |
| ------------ | --------------------------- |
| CNN Backbone | ResNet50                    |
| Framework    | PyTorch                     |
| Library      | timm                        |
| Task         | Binary Image Classification |

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* timm
* NumPy
* Pillow
* Matplotlib
* CUDA
* Jupyter Notebook

---

## License

This project is intended for educational and research purposes.

---

## Author

**Debangan Makhal**


Computer Vision • Deep Learning • AI Research
