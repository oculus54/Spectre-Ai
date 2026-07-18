# ============================================================
# AI vs Real Image Detection - Local Inference + Grad-CAM
# ============================================================

import os
import time
import argparse
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from PIL import Image

# ------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="AI vs Real Image Detection")
parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to input image"
)

parser.add_argument(
    "--weights",
    type=str,
    default="weights\SpectreV2.pth",
    help="Path to trained model"
)

args = parser.parse_args()

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device : {device}")

# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = timm.create_model(
            "resnet50",
            pretrained=False,
            num_classes=0
        )

        self.resnet_dim = self.resnet.num_features

        self.fusion = nn.Sequential(
            nn.Linear(self.resnet_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        features = self.resnet(x)
        fused = self.fusion(features)
        return self.classifier(fused)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
model = ResNetClassifier().to(device)

checkpoint = torch.load(args.weights, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model Loaded Successfully")

# ------------------------------------------------------------
# Grad-CAM Hooks
# ------------------------------------------------------------
gradients = {}
activations = {}

def save_gradient(name):
    def hook(grad):
        gradients[name] = grad
    return hook

def forward_hook(module, input, output):
    activations["layer4"] = output
    output.register_hook(save_gradient("layer4"))

target_layer = model.resnet.layer4
handle = target_layer.register_forward_hook(forward_hook)

# ------------------------------------------------------------
# Load Image
# ------------------------------------------------------------
image = Image.open(args.image).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

input_tensor = transform(image).unsqueeze(0).to(device)
input_tensor.requires_grad_()

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
start_time = time.time()

output = model(input_tensor)

probability_real = torch.sigmoid(output).item()

model.zero_grad()

output.backward()

end_time = time.time()

# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------
grads = gradients["layer4"]
acts = activations["layer4"]

weights = grads.mean(dim=(2,3), keepdim=True)

cam = (weights * acts).sum(dim=1, keepdim=True)

cam = F.relu(cam)

cam = F.interpolate(
    cam,
    size=(224,224),
    mode="bilinear",
    align_corners=False
)

cam = cam.squeeze().detach().cpu().numpy()

cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)

handle.remove()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if probability_real >= 0.5:
    prediction = "REAL IMAGE"
else:
    prediction = "AI GENERATED IMAGE"

confidence = max(probability_real,1-probability_real)

prob_real = probability_real*100
prob_ai = (1-probability_real)*100

# ------------------------------------------------------------
# Create Results Folder
# ------------------------------------------------------------
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
img_resized = image.resize((224,224))
img_np = np.array(img_resized)/255.0

fig, axes = plt.subplots(1,3,figsize=(16,6))

axes[0].imshow(img_np)
axes[0].set_title(
    f"{prediction}\nConfidence : {confidence*100:.2f}%"
)
axes[0].axis("off")

axes[1].imshow(cam,cmap="jet")
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(img_np)
axes[2].imshow(cam,cmap="jet",alpha=0.5)
axes[2].set_title("Grad-CAM Overlay")
axes[2].axis("off")

plt.tight_layout()

save_path = os.path.join("results","gradcam_result.png")
plt.savefig(save_path,dpi=300)

plt.show()

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
print("="*65)
print("              AI vs REAL IMAGE DETECTION REPORT")
print("="*65)

print(f"Image Name       : {os.path.basename(args.image)}")
print(f"Prediction       : {prediction}")
print(f"Confidence       : {confidence*100:.2f}%")
print(f"AI Probability   : {prob_ai:.2f}%")
print(f"Real Probability : {prob_real:.2f}%")
print(f"Inference Time   : {(end_time-start_time):.4f} sec")
print(f"Device           : {device}")
print(f"Model            : ResNet50")
print(f"Output Saved     : {save_path}")

print("="*65)

# ------------------------------------------------------------
# Probability Plot
# ------------------------------------------------------------
plt.figure(figsize=(6,4))

classes = ["AI Generated","Real"]

scores = [prob_ai,prob_real]

bars = plt.bar(classes,scores)

plt.ylim(0,100)

plt.ylabel("Probability (%)")

plt.title("Prediction Confidence")

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        y+1,
        f"{y:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.grid(axis="y",alpha=0.3)

plt.savefig("results/probability_graph.png",dpi=300)

plt.show()

# ------------------------------------------------------------
# Final Decision
# ------------------------------------------------------------
print()

if prediction=="REAL IMAGE":
    print("✅ Final Decision : The uploaded image is classified as a REAL IMAGE.")
else:
    print("🚨 Final Decision : The uploaded image is classified as an AI GENERATED IMAGE.")