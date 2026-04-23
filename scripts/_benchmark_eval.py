"""
Benchmark-specific evaluator.

Evaluates YOLO weights on the val split of a COCO-annotated synthetic dataset
using pycocotools for real mAP. Only called by benchmark_ewc.py.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict


def evaluate_on_split(
    weights_path: str,
    coco_json: str,
    split: str = "val",
) -> Dict[str, float]:
    """
    Run YOLO inference on the `split` subset of `coco_json` and compute real mAP.

    Args:
        weights_path: Path to fine-tuned .pt weights.
        coco_json:    Path to COCO JSON with 'split' field on each image entry
                      (written by stage_generate after Task 10).
        split:        Which split to evaluate. Default: 'val'.

    Returns:
        {"map50": float, "map50_95": float, "eval_mode": "real_map", "n_images": int}
    """
    from ultralytics import YOLO
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.metrics import compute_map, yolo_results_to_coco_predictions

    with open(coco_json) as f:
        data = json.load(f)

    val_images = [img for img in data["images"] if img.get("split", "train") == split]
    if not val_images:
        raise ValueError(
            f"No images with split='{split}' in {coco_json}. "
            "Run stage_generate with Task 10 changes applied."
        )

    val_image_ids = {img["id"] for img in val_images}
    val_annotations = [a for a in data["annotations"] if a["image_id"] in val_image_ids]

    val_gt = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": data["categories"],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(val_gt, f)
        val_gt_path = f.name

    try:
        coco_dir = Path(coco_json).parent
        val_dir = coco_dir / "images" / split
        if not val_dir.is_dir() or not list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")):
            raise FileNotFoundError(
                f"Val images directory not found or empty: {val_dir}. "
                "Ensure stage_generate created images/val/ via Task 10."
            )

        model = YOLO(weights_path)
        results = model(str(val_dir), conf=0.25, verbose=False)

        image_id_map = {Path(img["file_name"]).stem: img["id"] for img in val_images}
        predictions = yolo_results_to_coco_predictions(results, image_id_map)
        metrics = compute_map(predictions, ground_truth_coco=val_gt_path)
    finally:
        os.unlink(val_gt_path)

    return {
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "eval_mode": "real_map",
        "n_images": len(val_images),
    }
