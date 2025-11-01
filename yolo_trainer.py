# yolo_trainer.py
from __future__ import annotations
import os, io, time, zipfile, shutil
from typing import Dict, Any, Tuple, List, Optional

from ultralytics import YOLO
from PIL import Image
import numpy as np


# -------- Zip helpers --------
def extract_zip_to_dir(zip_bytes: bytes, base_dir: str = "uploads") -> str:
    os.makedirs(base_dir, exist_ok=True)
    ds_dir = os.path.join(base_dir, f"yolo_{int(time.time())}")
    os.makedirs(ds_dir, exist_ok=True)
    zip_path = os.path.join(ds_dir, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ds_dir)
    os.remove(zip_path)
    return ds_dir


def find_data_yaml(root: str) -> Optional[str]:
    # Walk and return the first data.yaml we find (Roboflow/YOLO exports include this).
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower() == "data.yaml":
                return os.path.join(r, fn)
    return None


# -------- Train --------
def train_yolo(
    model_name: str,
    data_yaml: str,
    imgsz: int = 640,
    epochs: int = 50,
    batch: int = 16,
    device: str = "",     # "", "cpu", "cuda", "mps"
    project: str = "runs/detect",
    name: str = "exp",
    patience: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a YOLO model. Returns dict with paths and basic info.
    model_name examples: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov11n.pt", etc.
    """
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device or None,
        project=project,
        name=name,
        patience=patience,
        seed=seed,
        verbose=True,
    )
    # Ultralytics writes best.pt here:
    save_dir = results.save_dir  # e.g., runs/detect/exp
    best_path = os.path.join(save_dir, "weights", "best.pt")
    last_path = os.path.join(save_dir, "weights", "last.pt")
    # Load best back for inference summary
    model_best = YOLO(best_path)
    names = model_best.names  # {class_id: name}
    return {
        "save_dir": save_dir,
        "best": best_path,
        "last": last_path,
        "names": names
    }


# -------- Inference --------
def infer_image_bytes(
    weights_path: str,
    image: Image.Image,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = ""
) -> Tuple[Image.Image, Dict[str, int]]:
    """
    Run detection on a PIL image. Returns annotated PIL image and per-class counts.
    """
    model = YOLO(weights_path)
    res = model.predict(
        source=np.array(image),  # accepts numpy array
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device or None,
        verbose=False,
    )
    r = res[0]
    # Render annotated image
    annotated = Image.fromarray(r.plot())
    # Count classes
    counts: Dict[str, int] = {}
    names = model.names
    if r.boxes is not None and r.boxes.cls is not None:
        for cls_id in r.boxes.cls.cpu().numpy().astype(int):
            cls_name = names.get(cls_id, str(cls_id))
            counts[cls_name] = counts.get(cls_name, 0) + 1
    return annotated, counts


def infer_video_file(
    weights_path: str,
    video_bytes: bytes,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = ""
) -> Tuple[List[Image.Image], List[Dict[str, int]]]:
    """
    Simple video inference: samples frames (every n frames) and returns list of annotated frames + counts.
    (Streamlit can display these as a gallery.)
    """
    import cv2
    import tempfile
    frames_annotated: List[Image.Image] = []
    frames_counts: List[Dict[str, int]] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        path = tmp.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], []

    model = YOLO(weights_path)
    idx = 0
    step = 10  # sample every N frames to keep things light
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            res = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device or None,
                verbose=False,
            )[0]
            annotated = Image.fromarray(res.plot())
            counts: Dict[str, int] = {}
            names = model.names
            if res.boxes is not None and res.boxes.cls is not None:
                for cls_id in res.boxes.cls.cpu().numpy().astype(int):
                    cls_name = names.get(cls_id, str(cls_id))
                    counts[cls_name] = counts.get(cls_name, 0) + 1
            frames_annotated.append(annotated)
            frames_counts.append(counts)
        idx += 1

    cap.release()
    try:
        os.remove(path)
    except Exception:
        pass

    return frames_annotated, frames_counts
