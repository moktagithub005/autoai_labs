# cv_trainer.py
# UNISOLE Auto-AI | PyTorch Image Classification Trainer (CPU/GPU/MPS Compatible)

from __future__ import annotations
import os
import json
import time
import zipfile
import shutil
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# ---------------------- CONFIG ----------------------

@dataclass
class CVTrainConfig:
    model_name: str = "resnet18"                # resnet18 | efficientnet_b0 | mobilenet_v3_small
    img_size: int = 224
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 5
    val_split: float = 0.2
    augment: bool = True
    augment_strength: str = "auto"             # auto | heavy
    seed: int = 42
    device: str = "auto"                       # auto | cpu | cuda | mps (apple)


# ---------------------- UTILITIES ----------------------

def best_device(pref: str = "auto") -> torch.device:
    """Choose best device available."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # Auto fallback logic
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_dataset_from_zip(zip_bytes: bytes, out_dir: str = "uploads") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ds_id = f"dataset_{int(time.time())}"
    ds_dir = os.path.join(out_dir, ds_id)
    os.makedirs(ds_dir, exist_ok=True)
    zip_path = os.path.join(ds_dir, "raw.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ds_dir)
    os.remove(zip_path)
    return ds_dir


# ---------------------- TRANSFORMS ----------------------

def make_transforms(img_size: int, augment: bool, strength: str = "auto"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = [transforms.Resize((img_size, img_size))]
    if augment:
        aug_light = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.15, 0.15, 0.15),
        ]
        aug_heavy = aug_light + [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        train_tf += aug_heavy if strength == "heavy" else aug_light

    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transforms.Compose(train_tf), val_tf


# ---------------------- DATA LOADERS ----------------------

def build_loaders(data_dir: str, cfg: CVTrainConfig):
    train_tf, val_tf = make_transforms(cfg.img_size, cfg.augment, cfg.augment_strength)
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tf)

    n_total = len(full_ds)
    n_val = max(1, int(cfg.val_split * n_total))
    n_train = max(1, n_total - n_val)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    # Apply validation transform
    val_ds.dataset = datasets.ImageFolder(root=data_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, cfg.batch_size * 2), shuffle=False)

    return train_loader, val_loader, full_ds.classes


# ---------------------- MODELS ----------------------

def build_model(model_name: str, num_classes: int, device: torch.device):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


# ---------------------- TRAIN LOOP ----------------------

def train_loop(model, train_loader, val_loader, cfg: CVTrainConfig, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        total_vl = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_vl += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()

        avg_val_loss = total_vl / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

        print(f"Epoch {epoch}/{cfg.epochs}: train_loss={avg_train_loss:.4f}, val_acc={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, history


# ---------------------- PLOTS ----------------------

def plot_history(history: Dict[str, List[float]]):
    fig1 = plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    fig2 = plt.figure(figsize=(6, 4))
    plt.plot(history["val_acc"], label="val_acc", color="green")
    plt.title("Validation Accuracy")
    plt.legend()

    return [fig1, fig2]


# ---------------------- SAVE & LOAD ----------------------

def save_model(model, classes, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    ts = str(int(time.time()))
    model_path = os.path.join(out_dir, f"cv_model_{ts}.pt")
    labels_path = os.path.join(out_dir, f"labels_{ts}.json")
    torch.save(model.state_dict(), model_path)
    with open(labels_path, "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    return model_path, labels_path


def load_model_for_infer(model_path, model_name, labels_json, device):
    with open(labels_json) as f:
        classes = json.load(f)["classes"]
    model = build_model(model_name, len(classes), device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, classes


# ---------------------- PREDICT ----------------------

def predict_image(model, img: Image.Image, img_size, device, classes):
    _, val_tf = make_transforms(img_size, False)
    x = val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = probs.argmax()
    return classes[idx], float(probs[idx]), probs, classes
