import time
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import timm
from PIL import Image

# -------------------------------------------------------------
# Hybrid ResNet50 + Vision Transformer Model Architecture
# -------------------------------------------------------------
class HybridClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # ResNet50 Backbone
        # -------------------------
        self.resnet = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0      # Remove classifier
        )

        # Output Feature Size = 2048
        self.resnet_dim = self.resnet.num_features

        # -------------------------
        # Vision Transformer
        # -------------------------
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        # Output Feature Size = 768
        self.vit_dim = self.vit.num_features

        # -------------------------
        # Feature Fusion
        # -------------------------
        fusion_dim = self.resnet_dim + self.vit_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Binary Classifier
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        cnn_features = self.resnet(x)
        vit_features = self.vit(x)
        features = torch.cat(
            (cnn_features, vit_features),
            dim=1
        )
        fused = self.fusion(features)
        output = self.classifier(fused)
        return output

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Model
# -------------------------------
model = HybridClassifier().to(device)

model.load_state_dict(
    torch.load(
        "weights\Spectre.pth",
        map_location=device
    )
)

model.eval()

print("✅ Model Loaded Successfully")

# -------------------------------
# Local Image Selection
# -------------------------------
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "D:\Completed-Projects\Spectre-V1\images\kag6.jpeg"

if not os.path.exists(image_path):
    print(f"❌ Error: Image file '{image_path}' not found.")
    image_path = input("Please enter the path to an image file: ").strip()
    image_path = image_path.strip('\'"')

if not os.path.exists(image_path):
    print(f"❌ Error: File '{image_path}' still not found. Exiting.")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")

# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

input_tensor = transform(image).unsqueeze(0).to(device)

# -------------------------------
# Prediction
# -------------------------------
start_time = time.time()

with torch.no_grad():

    output = model(input_tensor)

    probability_real = torch.sigmoid(output).item()

end_time = time.time()

# Binary Prediction
if probability_real >= 0.5:
    prediction = "REAL IMAGE"
else:
    prediction = "AI GENERATED IMAGE"

confidence = max(probability_real, 1 - probability_real)

prob_ai = (1 - probability_real) * 100
prob_real = probability_real * 100

# -------------------------------
# Display Uploaded Image
# -------------------------------
plt.figure(figsize=(7,7))
plt.imshow(image)
plt.axis("off")
plt.title(
    f"{prediction}\nConfidence : {confidence*100:.2f}%",
    fontsize=15,
    fontweight="bold"
)
plt.show()

# -------------------------------
# Professional Inference Report
# -------------------------------
print("="*60)
print("        AI vs REAL IMAGE DETECTION REPORT")
print("="*60)

print(f"Image Name      : {image_path}")
print(f"Prediction      : {prediction}")
print(f"Confidence      : {confidence*100:.2f}%")
print(f"AI Probability  : {prob_ai:.2f}%")
print(f"Real Probability: {prob_real:.2f}%")
print(f"Inference Time  : {(end_time-start_time):.4f} sec")
print(f"Device          : {device}")
print(f"Model           : Hybrid ResNet50 + Vision Transformer")

print("="*60)

# -------------------------------
# Probability Graph
# -------------------------------
plt.figure(figsize=(6,4))

classes = ["AI Generated","Real"]

scores = [prob_ai, prob_real]

bars = plt.bar(classes, scores)

plt.ylim(0,100)

plt.ylabel("Probability (%)")

plt.title("Prediction Confidence")

for bar in bars:

    y = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        y+1,
        f"{y:.2f}%",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.grid(axis="y", alpha=0.3)

plt.show()

# -------------------------------
# Final Decision
# -------------------------------
print()

if prediction == "REAL IMAGE":
    print("✅ Final Decision : The uploaded image is classified as a REAL IMAGE.")
else:
    print("🚨 Final Decision : The uploaded image is classified as an AI GENERATED IMAGE.")