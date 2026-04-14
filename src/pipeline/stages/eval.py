"""
Stage 4: Sim2Real Validation

Runs the fine-tuned YOLOv8n on real shelf images and computes mAP@50
using COCO-standard evaluation (pycocotools) or the NumPy fallback.

The goal is to quantify the Sim2Real transfer gap — how well synthetic
training priors generalise to physical retail shelf conditions including:
  - Harsh specular reflections on metallic packaging
  - Partial hand occlusions during pick events
  - Out-of-distribution ambient lighting (LED, fluorescent, mixed)
  - SKU rotation and tilt on curved pegboard shelves
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def stage_eval(
    real_images_dir: str,
    weights_path: str,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    coco_gt_path: Optional[str] = None,
    dry_run: bool = False,
    job_id: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """
    Evaluate trained weights on real images and return metric dict.

    Args:
        real_images_dir: Directory of real product images for validation.
        weights_path:    Path to fine-tuned .pt weights.
        config:          Loaded config.yaml.
        coco_gt_path:    Optional COCO GT JSON for proper mAP (required for
                         pycocotools). When None, uses score-based proxy.
        dry_run:         Return mock metrics without running inference.
        job_id:          Run namespace.
        status_callback: API progress hook.

    Returns:
        Dict: {"map50": float, "map50_95": float, "n_images": int,
               "n_detections": int, "mean_confidence": float}
    """
    tag = f"[Stage 4: Eval{f'/{job_id}' if job_id else ''}]"

    if status_callback:
        status_callback("eval")

    if dry_run:
        mock = {"map50": 0.0, "map50_95": 0.0, "n_images": 0,
                "n_detections": 0, "mean_confidence": 0.0, "dry_run": True}
        logger.info(f"{tag} Dry-run — returning mock metrics.")
        return mock

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(f"{tag} ultralytics not installed: {e}") from e

    from ...utils.metrics import compute_map, yolo_results_to_coco_predictions

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{tag} Weights not found: {weights_path}. Run Stage 3.")

    real_dir = Path(real_images_dir)
    if not real_dir.is_dir():
        raise FileNotFoundError(f"{tag} Real images directory not found: {real_images_dir}")

    image_files = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    if not image_files:
        logger.warning(f"{tag} No images found in {real_images_dir}.")
        return {"map50": 0.0, "map50_95": 0.0, "n_images": 0,
                "n_detections": 0, "mean_confidence": 0.0}

    logger.info(f"{tag} Running inference on {len(image_files)} images...")
    model = YOLO(weights_path)
    conf_threshold = config.get("eval_confidence_threshold", 0.25)
    results = model(str(real_images_dir), conf=conf_threshold, verbose=False)

    # ---- Aggregate raw detection stats ----------------------------------------
    total_detections = sum(len(r.boxes) for r in results)
    all_confs = [
        box.conf.item()
        for r in results
        for box in r.boxes
    ]
    mean_conf = float(sum(all_confs) / len(all_confs)) if all_confs else 0.0

    logger.info(
        f"{tag} Inference complete — "
        f"{total_detections} detections across {len(image_files)} images "
        f"(mean conf: {mean_conf:.3f})"
    )

    # Per-image detection breakdown
    for result in results:
        img_name = Path(result.path).name
        n = len(result.boxes)
        confs = [f"{b.conf.item():.2f}" for b in result.boxes]
        logger.info(f"  {img_name}: {n} detection(s) — conf={confs}")

    # ---- mAP computation -------------------------------------------------------
    if coco_gt_path and os.path.exists(coco_gt_path):
        with open(coco_gt_path) as f:
            gt_data = json.load(f)
        image_id_map = {
            Path(img["file_name"]).stem: img["id"]
            for img in gt_data.get("images", [])
        }
        predictions = yolo_results_to_coco_predictions(results, image_id_map)
        map_metrics = compute_map(predictions, ground_truth_coco=coco_gt_path)
        logger.info(
            f"{tag} mAP@50={map_metrics['map50']:.3f}  "
            f"mAP@50:95={map_metrics['map50_95']:.3f}"
        )
    else:
        # Score-based proxy mAP when no ground-truth annotations exist
        # (common scenario: user has real images but no bounding box labels)
        predictions = [
            {"image_id": i, "category_id": 1,
             "bbox": box.xyxy[0].tolist(), "score": box.conf.item()}
            for i, result in enumerate(results)
            for box in result.boxes
        ]
        map_metrics = compute_map(predictions, ground_truth_coco=None)
        logger.info(
            f"{tag} Proxy mAP@50={map_metrics['map50']:.3f} "
            f"(no GT annotations — for ground-truth mAP pass --gt_coco)"
        )

    metrics = {
        "map50": map_metrics["map50"],
        "map50_95": map_metrics["map50_95"],
        "n_images": len(image_files),
        "n_detections": total_detections,
        "mean_confidence": mean_conf,
    }

    # Persist metrics alongside weights
    metrics_path = os.path.join(checkpoint_dir, f"{job_id or 'eval'}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"{tag} Metrics saved: {metrics_path}")

    return metrics
