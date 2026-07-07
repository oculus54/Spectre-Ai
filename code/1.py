import torch

ckpt = torch.load("Spectre.pth", map_location="cpu")

print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    print("\nFirst 10 keys:")
    for i, k in enumerate(ckpt.keys()):
        print(k)
        if i == 9:
            break