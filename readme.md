# Spectre – AI vs Real Image Detection

Spectre is a deep learning-based image forensics model that classifies an image as **AI-Generated** or **Real**. The model combines the feature extraction capabilities of **ResNet50** with the global contextual understanding of a **Vision Transformer (ViT-B/16)** for robust binary image classification.

---

## Features

- AI vs Real image classification
- Hybrid ResNet50 + Vision Transformer architecture
- CUDA-enabled inference
- Confidence score and probability visualization
- Fast single-image inference
- PyTorch + timm implementation

---

## Model Architecture

```
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
├── inference.py
├── Spectre.pth
├── image.jpg
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/Spectre.git
cd Spectre

pip install -r requirements.txt
```

or

```bash
pip install torch torchvision timm matplotlib pillow
```

---

## Usage

1. Place the trained model (`Spectre.pth`) in the project folder.
2. Place the input image (`image.jpg`) in the project folder.
3. Run

```bash
cd code && python model.py
```

---

## Performance

| Metric | Score |
|--------|------:|
| Accuracy | **99.05%** |
| Macro Precision | **98.77%** |
| Macro Recall | **98.63%** |
| Macro F1-Score | **98.70%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|----------:|-------:|---------:|
| AI Generated | **98.22%** | **97.83%** | **98.03%** |
| Real | **99.31%** | **99.43%** | **99.37%** |

**Test Samples:** 2,100

---

## Tech Stack

- Python
- PyTorch
- timm
- Torchvision
- Pillow
- Matplotlib

---

## Image Preprocessing

- Resize to **224 × 224**
- Convert to Tensor
- Normalize using ImageNet statistics

```python
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Output

The inference script provides:

- Predicted class
- Confidence score
- AI probability
- Real probability
- Inference time
- Probability bar chart

---

## Model

- **CNN Backbone:** ResNet50
- **Transformer Backbone:** ViT Base Patch16 224
- **Framework:** PyTorch
- **Library:** timm
- **Task:** Binary Image Classification

---

## License

This project is intended for educational and research purposes.

---

## Author

**Debangan Makhal**

Computer Vision • Deep Learning • AI Research