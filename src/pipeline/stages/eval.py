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

Failure Gallery
---------------
After inference, the 10 images on which the model was least confident
(lowest max detection score, or zero detections) are copied to:

    <checkpoint_dir>/failure_gallery/<job_id>/

alongside a machine-readable JSON summary:

    <checkpoint_dir>/failure_gallery/<job_id>/gallery_summary.json

This makes iterative debugging concrete — instead of staring at aggregate
mAP numbers, you look at exactly which real images confused the model and
why (occlusion? lighting? packaging variant? out-of-distribution angle?).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _build_failure_gallery(
    results: list,
    image_files: List[Path],
    checkpoint_dir: str,
    job_id: Optional[str],
    n_failures: int = 10,
) -> str:
    """
    Identify the N lowest-confidence images and copy them to a labelled
    failure gallery directory for manual inspection.

    Confidence score used: max detection confidence per image.
    Images with zero detections are treated as confidence = 0.0 and
    ranked first (they are the hardest cases for the model).

    Args:
        results:        List of Ultralytics Results objects from model().
        image_files:    Ordered list of image paths (same order as results).
        checkpoint_dir: Root checkpoint directory.
        job_id:         Run identifier (used as sub-directory name).
        n_failures:     Number of worst images to include. Default: 10.

    Returns:
        Path to the gallery directory.
    """
    # --- Score each image by its maximum detection confidence --------------------
    scored: List[Tuple[float, Path, int]] = []   # (max_conf, path, n_detections)
    for result, img_path in zip(results, image_files):
        boxes = result.boxes
        if len(boxes) == 0:
            max_conf = 0.0
        else:
            max_conf = float(max(b.conf.item() for b in boxes))
        scored.append((max_conf, img_path, len(boxes)))

    # Sort ascending — lowest confidence first
    scored.sort(key=lambda x: x[0])
    worst_n = scored[:n_failures]

    # --- Create gallery directory -------------------------------------------------
    gallery_dir = Path(checkpoint_dir) / "failure_gallery" / (job_id or "default")
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy images and build summary JSON --------------------------------------
    summary = {
        "job_id":         job_id,
        "n_total_images": len(image_files),
        "n_gallery":      len(worst_n),
        "description": (
            "Images ranked by lowest model confidence. "
            "Zero-detection images appear first. "
            "Use this gallery to diagnose: occlusion patterns, lighting outliers, "
            "packaging variants, or out-of-distribution angles."
        ),
        "failures": [],
    }

    for rank, (conf, src_path, n_dets) in enumerate(worst_n, start=1):
        dest_name = f"{rank:02d}_conf{conf:.3f}_{src_path.name}"
        dest_path = gallery_dir / dest_name
        shutil.copy2(src_path, dest_path)

        failure_entry = {
            "rank":          rank,
            "original_path": str(src_path),
            "gallery_path":  str(dest_path),
            "max_confidence": round(conf, 4),
            "n_detections":  n_dets,
            "diagnosis_hint": (
                "No detections — possible full occlusion, extreme angle, or "
                "severe domain shift."
                if n_dets == 0 else
                f"Low confidence ({conf:.3f}) — model uncertain. "
                "Check for partial occlusion, specular highlights, or "
                "packaging variant not seen in synthetic training data."
            ),
        }
        summary["failures"].append(failure_entry)

    # --- Persist summary ---------------------------------------------------------
    summary_path = gallery_dir / "gallery_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[Failure Gallery] {len(worst_n)} worst images copied to {gallery_dir}"
    )
    logger.info(f"[Failure Gallery] Summary: {summary_path}")

    # --- Log the top-3 to console for immediate visibility -----------------------
    for entry in summary["failures"][:3]:
        logger.warning(
            f"  Rank {entry['rank']:02d} | conf={entry['max_confidence']:.3f} | "
            f"{Path(entry['original_path']).name}  — {entry['diagnosis_hint'][:60]}..."
        )

    return str(gallery_dir)


def stage_eval(
    real_images_dir: str,
    weights_path: str,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    coco_gt_path: Optional[str] = None,
    dry_run: bool = False,
    job_id: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    n_failure_gallery: int = 10,
) -> Dict[str, float]:
    """
    Evaluate trained weights on real images and return metric dict.

    Args:
        real_images_dir: Directory of real product images for validation.
        weights_path:    Path to fine-tuned .pt weights.
        config:          Loaded config.yaml.
        coco_gt_path:    Optional COCO GT JSON for proper mAP (required for
                         pycocotools). When None, uses score-based proxy.
        dry_run:            Return mock metrics without running inference.
        job_id:             Run namespace.
        status_callback:    API progress hook.
        n_failure_gallery:  Number of lowest-confidence images to write to
                            the failure gallery. Set 0 to disable. Default: 10.

    Returns:
        Dict: {"map50": float, "map50_95": float, "n_images": int,
               "n_detections": int, "mean_confidence": float,
               "failure_gallery_dir": str}
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

    # ---- Automated Failure Gallery ---------------------------------------------
    gallery_dir = ""
    if n_failure_gallery > 0:
        try:
            gallery_dir = _build_failure_gallery(
                results       = results,
                image_files   = image_files,
                checkpoint_dir= checkpoint_dir,
                job_id        = job_id,
                n_failures    = n_failure_gallery,
            )
        except Exception as exc:
            logger.warning(f"{tag} Failure gallery generation failed (non-fatal): {exc}")

    metrics = {
        "map50":               map_metrics["map50"],
        "map50_95":            map_metrics["map50_95"],
        "n_images":            len(image_files),
        "n_detections":        total_detections,
        "mean_confidence":     mean_conf,
        "failure_gallery_dir": gallery_dir,
    }

    # Persist metrics alongside weights
    metrics_path = os.path.join(checkpoint_dir, f"{job_id or 'eval'}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"{tag} Metrics saved: {metrics_path}")

    if gallery_dir:
        logger.info(f"{tag} Failure gallery: {gallery_dir}")

    return metrics
