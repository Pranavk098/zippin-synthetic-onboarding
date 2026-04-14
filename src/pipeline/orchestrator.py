"""
Pipeline Orchestrator — unified CLI + programmatic entry point.

Can be called:
  1. As a CLI:  python -m src.pipeline.orchestrator --stage all --image product.jpg
  2. As a lib:  from src.pipeline import run_pipeline; run_pipeline(...)
  3. Via API:   POST /onboard  (FastAPI server wraps this in a background task)

All three paths converge here, enabling the same pipeline logic whether
you're testing locally on Windows, benchmarking on Jetson Orin, or serving
production onboarding requests through the REST layer.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Callable, Dict, Optional

# Ensure the project root is importable when running as __main__
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except OSError:
        logger.warning(f"config.yaml not found at {config_path} — using defaults.")
        return {}


def run_pipeline(
    sku_name: str,
    image_path: str,
    stages: tuple = ("extract", "generate", "train", "eval"),
    real_dir: str = "real_validation/",
    coco_gt_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    dry_run: bool = False,
    job_id: Optional[str] = None,
    config_path: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Execute the full zero-shot SKU onboarding pipeline.

    Args:
        sku_name:        Human-readable product name.
        image_path:      Path to the reference product image.
        stages:          Tuple of stage names to execute (ordered).
        real_dir:        Directory of real images for Stage 4 eval.
        coco_gt_path:    Optional GT COCO JSON for real mAP computation.
        checkpoint_dir:  All intermediate artifacts land here.
        dry_run:         Skip heavy compute; verify wiring only.
        job_id:          Unique run identifier (UUID-short from API layer).
        config_path:     Override default config.yaml location.
        status_callback: Called with stage name as each stage begins.

    Returns:
        Dict with keys: weights_path, ewc_path, metrics, stages_run
    """
    config = load_config(config_path)
    job_tag = job_id or "local"
    result: Dict = {"stages_run": [], "weights_path": None, "ewc_path": None, "metrics": None}

    # ---- Stage 1: Extract -------------------------------------------------------
    if "extract" in stages:
        from .stages.extract import stage_extract
        logger.info(f"=== Stage 1/4: Semantic Extraction [{job_tag}] ===")
        attrs = stage_extract(
            image_path=image_path,
            config=config,
            checkpoint_dir=checkpoint_dir,
            dry_run=dry_run,
            job_id=job_id,
        )
        result["stages_run"].append("extract")
        result["sku_attributes"] = attrs

    # ---- Stage 2: Generate -------------------------------------------------------
    if "generate" in stages:
        from .stages.generate import stage_generate
        logger.info(f"=== Stage 2/4: Synthetic Generation [{job_tag}] ===")

        features_name = f"{job_id}_sku_features" if job_id else "sku_features"
        features_path = os.path.join(checkpoint_dir, f"{features_name}.json")

        coco_json = stage_generate(
            features_path=features_path,
            config=config,
            checkpoint_dir=checkpoint_dir,
            dry_run=dry_run,
            job_id=job_id,
            status_callback=status_callback,
        )
        result["stages_run"].append("generate")
        result["coco_json"] = coco_json

    # ---- Stage 3: Train -----------------------------------------------------------
    if "train" in stages:
        from .stages.train import stage_train
        logger.info(f"=== Stage 3/4: YOLOv8n + EWC Training [{job_tag}] ===")

        coco_json = result.get("coco_json") or os.path.join(
            checkpoint_dir, job_id or "synthetic_dataset", "coco_annotations.json"
        )
        weights_path = stage_train(
            coco_json=coco_json,
            config=config,
            checkpoint_dir=checkpoint_dir,
            dry_run=dry_run,
            job_id=job_id,
            status_callback=status_callback,
        )
        result["stages_run"].append("train")
        result["weights_path"] = weights_path
        result["ewc_path"] = os.path.join(checkpoint_dir, "ewc_state.pt")

    # ---- Stage 4: Eval -----------------------------------------------------------
    if "eval" in stages:
        from .stages.eval import stage_eval
        logger.info(f"=== Stage 4/4: Sim2Real Evaluation [{job_tag}] ===")

        weights_path = result.get("weights_path") or os.path.join(
            checkpoint_dir, f"{job_id or 'new_sku'}_weights.pt"
        )
        metrics = stage_eval(
            real_images_dir=real_dir,
            weights_path=weights_path,
            config=config,
            checkpoint_dir=checkpoint_dir,
            coco_gt_path=coco_gt_path,
            dry_run=dry_run,
            job_id=job_id,
            status_callback=status_callback,
        )
        result["stages_run"].append("eval")
        result["metrics"] = metrics

    logger.info(f"Pipeline complete [{job_tag}]. Stages run: {result['stages_run']}")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline.orchestrator",
        description="Zippin Zero-Shot SKU Onboarding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a new product image:
  python -m src.pipeline.orchestrator --stage all --image product.jpg --sku_name "RedBull250ml"

  # Run only extraction + generation:
  python -m src.pipeline.orchestrator --stage extract generate --image product.jpg

  # Dry-run to verify wiring without GPU:
  python -m src.pipeline.orchestrator --stage all --image product.jpg --dry-run

  # Evaluate against real images with GT annotations:
  python -m src.pipeline.orchestrator --stage eval --real_dir cvs_occlusion_photos/ --gt_coco cvs_gt.json
        """,
    )
    p.add_argument("--stage", nargs="+",
                   choices=["extract", "generate", "train", "eval", "all"],
                   required=True, help="Pipeline stage(s) to run.")
    p.add_argument("--image", type=str, help="Reference product image path.")
    p.add_argument("--sku_name", type=str, default="UnknownSKU")
    p.add_argument("--real_dir", type=str, default="real_validation/")
    p.add_argument("--gt_coco", type=str, default=None,
                   help="COCO GT JSON for ground-truth mAP (optional).")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--config", type=str, default=None,
                   help="Override config.yaml path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip heavy computation — verify config/wiring only.")
    p.add_argument("--job_id", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    stages = args.stage
    if "all" in stages:
        stages = ["extract", "generate", "train", "eval"]

    if "extract" in stages and not args.image and not args.dry_run:
        print("Error: --image is required for the 'extract' stage.")
        sys.exit(1)

    run_pipeline(
        sku_name=args.sku_name,
        image_path=args.image or "",
        stages=tuple(stages),
        real_dir=args.real_dir,
        coco_gt_path=args.gt_coco,
        checkpoint_dir=args.checkpoint_dir,
        dry_run=args.dry_run,
        job_id=args.job_id,
        config_path=args.config,
    )
