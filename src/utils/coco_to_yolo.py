"""
COCO → YOLO format converter.

BlenderProc2 outputs COCO JSON annotations. YOLOv8 expects per-image .txt
label files in normalised [class cx cy w h] format. This module handles
the conversion and validates the output before training begins.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def convert_coco_to_yolo(
    coco_json: str,
    output_dir: str,
    class_names: Optional[list] = None,
    val_split: float = 0.15,
) -> str:
    """
    Convert a COCO annotation file to YOLO dataset format.

    Args:
        coco_json:    Path to the COCO annotations JSON.
        output_dir:   Root directory for the YOLO dataset.
        class_names:  List of class names. Defaults to ["TargetSKU"].
        val_split:    Fraction of images to use as validation set.

    Returns:
        Path to the generated dataset.yaml (passed directly to YOLO.train).
    """
    if class_names is None:
        class_names = ["TargetSKU"]

    with open(coco_json, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    annotations = data.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    if not images:
        raise ValueError(f"COCO JSON contains no images: {coco_json}")

    # Build per-image annotation lookup
    img_to_anns: dict = {img_id: [] for img_id in images}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id in img_to_anns:
            img_to_anns[img_id].append(ann)

    # Split into train/val
    all_ids = list(images.keys())
    n_val = max(1, int(len(all_ids) * val_split))
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    coco_dir = Path(coco_json).parent

    for split, id_set in [("train", train_ids), ("val", val_ids)]:
        img_dir = Path(output_dir) / "images" / split
        lbl_dir = Path(output_dir) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in id_set:
            img_meta = images[img_id]
            img_filename = img_meta["file_name"]
            W = img_meta["width"]
            H = img_meta["height"]

            # Copy image
            src = coco_dir / img_filename
            dst = img_dir / Path(img_filename).name
            if src.exists():
                shutil.copy(src, dst)
            else:
                logger.warning(f"Image not found: {src}")

            # Write label file
            stem = Path(img_filename).stem
            label_path = lbl_dir / f"{stem}.txt"
            with open(label_path, "w") as lf:
                for ann in img_to_anns.get(img_id, []):
                    cat_id = ann.get("category_id", 1)
                    # Map COCO category IDs to zero-indexed YOLO class IDs
                    cls_idx = list(categories.keys()).index(cat_id) if cat_id in categories else 0
                    x, y, w, h = ann["bbox"]
                    # YOLO normalised centre-x, centre-y, width, height
                    xc = (x + w / 2) / W
                    yc = (y + h / 2) / H
                    wn = w / W
                    hn = h / H
                    # Clamp to [0, 1] to guard against BlenderProc edge cases
                    xc, yc, wn, hn = (
                        max(0.0, min(1.0, xc)),
                        max(0.0, min(1.0, yc)),
                        max(0.001, min(1.0, wn)),
                        max(0.001, min(1.0, hn)),
                    )
                    lf.write(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    # Write dataset.yaml
    yaml_path = Path(output_dir) / "dataset.yaml"
    nc = len(class_names)
    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_content = (
        f"path: {Path(output_dir).resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names:\n{names_str}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    n_train = len(train_ids)
    n_val_actual = len(val_ids)
    logger.info(
        f"[COCO→YOLO] Converted {len(images)} images "
        f"(train={n_train}, val={n_val_actual}) → {yaml_path}"
    )
    return str(yaml_path)
