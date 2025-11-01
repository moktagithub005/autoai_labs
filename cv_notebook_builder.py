# cv_notebook_builder.py
from __future__ import annotations
import nbformat as nbf
from typing import Optional

HEADER = """# UNISOLE Auto-AI — Image Classification (Education Notebook)

This notebook was auto-generated. It shows:
- How to load a **folder-structured** dataset (class subfolders)
- Build **PyTorch** DataLoaders
- Train a **pretrained CNN** (ResNet/EfficientNet/MobileNet)
- View loss/accuracy curves
- Save and reuse the model for inference

> Tip: Run cells in order. Explore & edit freely.
"""

def _md(x): return nbf.v4.new_markdown_cell(x.strip("\n"))
def _py(x): return nbf.v4.new_code_cell(x.strip("\n"))

def build_cv_notebook_bytes(
    data_dir: str,
    model_name: str = "resnet18",
    img_size: int = 224,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    kernel_name: str = "auto_ml",
) -> bytes:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": kernel_name, "language": "python", "name": kernel_name}
    cells = []

    cells += [
        _md(HEADER),
        _md("## 1) Setup"),
        _py("""
import os, json, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
device
"""),
        _md("## 2) Paths & Hyperparameters"),
        _py(f'''
DATA_DIR = r"{data_dir}"
MODEL_NAME = "{model_name}"
IMG_SIZE = {img_size}
BATCH_SIZE = {batch_size}
EPOCHS = {epochs}
LR = {lr}
VAL_SPLIT = 0.2
AUGMENT = True
'''),
        _md("## 3) Transforms & Datasets"),
        _py("""
mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
train_tf = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
if AUGMENT:
    train_tf += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(0.15,0.15,0.15)]
train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
val_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean, std)])
full_ds = datasets.ImageFolder(DATA_DIR, transform=transforms.Compose(train_tf))

n_total = len(full_ds); n_val = max(1, int(n_total*VAL_SPLIT)); n_train = n_total - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val])
val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)
classes = full_ds.classes; len(classes), classes[:10]
"""),
        _md("## 4) DataLoaders"),
        _py("""
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=max(1,BATCH_SIZE*2), shuffle=False)
len(train_loader), len(val_loader)
"""),
        _md("## 5) Build Pretrained Model"),
        _py("""
if MODEL_NAME == "resnet18":
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_f = net.fc.in_features; net.fc = nn.Linear(in_f, len(classes))
elif MODEL_NAME == "efficientnet_b0":
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_f = net.classifier[1].in_features; net.classifier[1] = nn.Linear(in_f, len(classes))
elif MODEL_NAME == "mobilenet_v3_small":
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = net.classifier[3].in_features; net.classifier[3] = nn.Linear(in_f, len(classes))
else:
    raise ValueError("Unknown model")
net = net.to(device)
"""),
        _md("## 6) Train"),
        _py("""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS+1):
    net.train()
    run = 0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(); o = net(x); loss = criterion(o,y); loss.backward(); optimizer.step()
        run += loss.item()*x.size(0)
    tl = run/len(train_loader.dataset)

    net.eval(); vl=0.0; corr=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            o = net(x); loss = criterion(o,y)
            vl += loss.item()*x.size(0)
            corr += (o.argmax(1)==y).sum().item()
    vl /= len(val_loader.dataset); va = corr/len(val_loader.dataset)
    history["train_loss"].append(tl); history["val_loss"].append(vl); history["val_acc"].append(va)
    if va>best_acc: best_acc=va; best_state={k:v.cpu() for k,v in net.state_dict().items()}
    print(f"Epoch {epoch}/{EPOCHS}  TL={tl:.4f}  VL={vl:.4f}  VA={va:.4f}")

if best_state: net.load_state_dict(best_state)
"""),
        _md("## 7) Curves"),
        _py("""
plt.figure(); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val"); plt.title("Loss"); plt.legend(); plt.show()
plt.figure(); plt.plot(history["val_acc"], label="val_acc"); plt.title("Val Acc"); plt.legend(); plt.show()
"""),
        _md("## 8) Save & Reuse"),
        _py("""
stamp=str(int(time.time()))
mpath=f"cv_model_{stamp}.pt"; lpath=f"labels_{stamp}.json"
import json, torch
torch.save(net.state_dict(), mpath)
with open(lpath, "w") as f: json.dump({"classes": classes}, f, indent=2)
mpath, lpath
"""),
        _md("## 9) Inference Demo"),
        _py("""
from PIL import Image
def predict(path):
    img=Image.open(path).convert("RGB")
    x=val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        o=net(x); p=torch.softmax(o,1)[0].cpu().numpy()
    idx=int(o.argmax(1)); return classes[idx], float(p[idx])
# Example: change to your file
# predict("/path/to/image.jpg")
"""),
    ]
    nb["cells"] = cells
    return nbf.writes(nb).encode("utf-8")
