# 🎭 Spectre – AI vs Real Image Detection

Spectre is a deep learning-based image forensics system that detects whether an image is **AI-generated** or **authentic (real)**. The model combines the strengths of a **ResNet50** convolutional backbone and a **Vision Transformer (ViT-B/16)** to capture both local texture artifacts and global semantic inconsistencies.

---

## 📌 Features

- 🔍 Binary classification (AI Generated vs Real)
- 🧠 Hybrid ResNet50 + Vision Transformer architecture
- ⚡ GPU (CUDA) accelerated inference
- 📊 Confidence score visualization
- 📈 Prediction probability graph
- 🖼️ Supports inference on any RGB image
- 💾 PyTorch checkpoint loading

---

## 🏗️ Model Architecture

```
                Input Image (224 × 224)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
      ResNet50                 Vision Transformer
        │                               │
        └───────────────┬───────────────┘
                        │
              Feature Concatenation
                        │
                Fully Connected Layers
                        │
                 Binary Classification
                        │
            AI Generated / Real Image
```

---

## 🛠️ Tech Stack

- Python
- PyTorch
- timm
- Torchvision
- Pillow
- Matplotlib

---

## 📂 Project Structure

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

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Spectre.git
cd Spectre
```

Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually

```bash
pip install torch torchvision timm matplotlib pillow
```

---

## 🚀 Running Inference

1. Place your trained model

```
Spectre.pth
```

inside the project directory.

2. Place the test image

```
image.jpg
```

inside the project directory.

3. Run

```bash
python inference.py
```

---

## 📊 Output

The program displays

- Uploaded image
- Predicted class
- Confidence score
- AI probability
- Real probability
- Inference time
- Probability bar chart

Example

```
Prediction      : AI GENERATED IMAGE

Confidence      : 99.32%

AI Probability  : 99.32%

Real Probability: 0.68%

Inference Time  : 0.042 sec
```

---

## 🧠 Model Details

| Component | Backbone |
|-----------|----------|
| CNN | ResNet50 |
| Transformer | ViT Base Patch16 224 |
| Framework | PyTorch |
| Library | timm |
| Task | Binary Image Classification |

---

## 📈 Image Preprocessing

- Resize → 224 × 224
- Convert to Tensor
- Normalize using ImageNet statistics

```python
Mean = [0.485, 0.456, 0.406]

Std = [0.229, 0.224, 0.225]
```

---

## 📚 Dependencies

```
torch
torchvision
timm
matplotlib
Pillow
```

---

## ⚠️ Notes

- CUDA is automatically used if available.
- Images must be RGB.
- Model weights should be stored as `Spectre.pth`.
- The inference model architecture must exactly match the architecture used during training.

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Debangan Makhal**

Deep Learning • Computer Vision • AI Research