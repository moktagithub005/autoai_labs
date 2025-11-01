# cv_data_prep.py
from __future__ import annotations
import os, re, io, time, zipfile, shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

@dataclass
class ImportReport:
    total_images: int
    labeled: int
    unknown: int
    class_counts: Dict[str, int]
    organized_dir: Optional[str] = None

def _is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in IMG_EXTS

def _extract_zip_to_tmp(zip_bytes: bytes, base_dir: str = "uploads") -> str:
    os.makedirs(base_dir, exist_ok=True)
    tmp_dir = os.path.join(base_dir, f"raw_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)
    zpath = os.path.join(tmp_dir, "upload.zip")
    with open(zpath, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(tmp_dir)
    os.remove(zpath)
    return tmp_dir

def _scan_images(root: str) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(r, fn)
            if _is_image(p):
                paths.append(p)
    return sorted(paths)

def _heuristic_label_from_filename(fname: str) -> Optional[str]:
    """
    Heuristic: use the most 'wordy' token (letters only) from the filename.
    Examples:
      red_apple_001.jpg -> red
      dog-12.png -> dog
      class=cat img.png -> cat
    """
    base = os.path.splitext(os.path.basename(fname))[0].lower()
    base = base.replace("=", " ").replace("-", " ").replace("_", " ")
    tokens = re.split(r"[^a-z]+", base)
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    # prefer the longest alpha token (often the class)
    tokens.sort(key=len, reverse=True)
    return tokens[0]

def infer_labels_auto(img_paths: List[str]) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for p in img_paths:
        mapping[p] = _heuristic_label_from_filename(os.path.basename(p))
    return mapping

def apply_csv_labels(img_paths: List[str], csv_text: str) -> Dict[str, Optional[str]]:
    """
    csv_text format: filename,label  (header optional)
    Supports just basenames in first column.
    """
    mapping: Dict[str, Optional[str]] = {p: None for p in img_paths}
    # normalize
    table = {}
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    for ln in lines:
        if "," not in ln:
            continue
        a, b = ln.split(",", 1)
        a, b = a.strip(), b.strip()
        if a.lower() == "filename" and b.lower() == "label":
            continue
        table[a.lower()] = b
    for p in img_paths:
        bn = os.path.basename(p).lower()
        if bn in table:
            mapping[p] = table[bn]
    return mapping

def build_org_structure(mapping: Dict[str, Optional[str]], out_base: str = "uploads") -> Tuple[str, ImportReport, List[str]]:
    """
    Organize images to out_dir/<label>/filename.
    Returns: (organized_dir, report, unknown_list)
    """
    out_dir = os.path.join(out_base, f"organized_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    unknowns: List[str] = []
    class_counts: Dict[str, int] = {}
    labeled = 0

    for src, label in mapping.items():
        if not _is_image(src):
            continue
        if label is None or label.strip() == "":
            unknowns.append(src)
            continue
        label_s = re.sub(r"[^a-z0-9_\-]+", "_", label.lower())
        dst_dir = os.path.join(out_dir, label_s)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src))
        # if name collision, add counter
        if os.path.exists(dst_path):
            stem, ext = os.path.splitext(os.path.basename(src))
            k = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_dir, f"{stem}_{k}{ext}")
                k += 1
        shutil.copy2(src, dst_path)
        class_counts[label_s] = class_counts.get(label_s, 0) + 1
        labeled += 1

    report = ImportReport(
        total_images=len(mapping),
        labeled=labeled,
        unknown=len(unknowns),
        class_counts=class_counts,
        organized_dir=out_dir
    )
    return out_dir, report, unknowns

def add_manual_labels(mapping: Dict[str, Optional[str]], user_labels: Dict[str, str]) -> Dict[str, Optional[str]]:
    out = dict(mapping)
    for p, lab in user_labels.items():
        out[p] = lab if lab else out.get(p)
    return out

def zip_directory(dir_path: str) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, files in os.walk(dir_path):
            for fn in files:
                fp = os.path.join(r, fn)
                z.write(fp, arcname=os.path.relpath(fp, dir_path))
    bio.seek(0)
    return bio.getvalue()

def small_dataset_hint(class_counts: Dict[str, int], min_per_class: int = 50) -> Dict[str, int]:
    """Return how many images each class is short of the target, to guide augmentation."""
    need = {}
    for c, n in class_counts.items():
        if n < min_per_class:
            need[c] = max(0, min_per_class - n)
    return need
