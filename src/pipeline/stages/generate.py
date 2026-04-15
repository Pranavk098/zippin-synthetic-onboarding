"""
Stage 2: Procedural Synthetic Dataset Generation (BlenderProc2)

Orchestrates the BlenderProc2 subprocess that renders physics-based synthetic
images with randomized lighting, camera poses, and occlusion objects — then
outputs fully validated COCO annotations ready for YOLO conversion.

Design decision — separate subprocess:
  BlenderProc2 ships with its own embedded Python interpreter (Blender's bpy).
  It cannot be imported directly into a standard Python process. We invoke it
  via `blenderproc run src/rendering/bproc_generator.py <features_path>`.
  The features_path checkpoint acts as the data contract between stages.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def stage_generate(
    features_path: str,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    dry_run: bool = False,
    job_id: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Invoke BlenderProc2 to render synthetic training images.

    Args:
        features_path:   Path to the Stage 1 SKU features JSON checkpoint.
        config:          Loaded config.yaml dict.
        checkpoint_dir:  Root output directory.
        dry_run:         Skip actual render; log intent only.
        job_id:          Optional ID for namespacing outputs.
        status_callback: Callable receiving stage name updates for the API.

    Returns:
        Path to the COCO annotations JSON file.
    """
    tag = f"[Stage 2: Generate{f'/{job_id}' if job_id else ''}]"

    if status_callback:
        status_callback("generate")

    output_dir = os.path.join(checkpoint_dir, job_id or "synthetic_dataset")

    if dry_run:
        logger.info(f"{tag} Dry-run — skipping BlenderProc2 render.")
        _mock_coco_output(output_dir)
        return os.path.join(output_dir, "coco_annotations.json")

    # ---- Clean output dir before render -----------------------------------------
    # BlenderProc2 appends to coco_annotations.json rather than replacing it.
    # Wiping the images/ folder and the JSON ensures each run starts fresh and
    # never accumulates stale image references from previous sessions.
    images_dir = os.path.join(output_dir, "images")
    coco_json  = os.path.join(output_dir, "coco_annotations.json")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
        logger.info(f"{tag} Cleared stale renders: {images_dir}")
    if os.path.exists(coco_json):
        os.remove(coco_json)
        logger.info(f"{tag} Cleared stale COCO JSON: {coco_json}")

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"{tag} Features checkpoint missing: {features_path}. Run Stage 1 first."
        )

    blenderproc_bin = shutil.which("blenderproc")
    if not blenderproc_bin:
        raise EnvironmentError(
            f"{tag} `blenderproc` not found in PATH. "
            "Install with: pip install blenderproc"
        )

    script_path = Path(__file__).parent.parent.parent / "rendering" / "bproc_generator.py"
    if not script_path.exists():
        raise FileNotFoundError(f"{tag} Renderer script not found: {script_path}")

    render_count = config.get("render_count", 50)
    resolution = config.get("image_resolution", [640, 640])

    logger.info(
        f"{tag} Launching BlenderProc2 — "
        f"{render_count} frames @ {resolution[0]}x{resolution[1]}"
    )

    env = os.environ.copy()
    env["BPROC_OUTPUT_DIR"] = output_dir
    env["BPROC_RENDER_COUNT"] = str(render_count)
    env["BPROC_RESOLUTION_W"] = str(resolution[0])
    env["BPROC_RESOLUTION_H"] = str(resolution[1])

    result = subprocess.run(
        [blenderproc_bin, "run", str(script_path), features_path],
        env=env,
        capture_output=False,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{tag} BlenderProc2 exited with code {result.returncode}. "
            "Check stdout above for details."
        )

    coco_path = os.path.join(output_dir, "coco_annotations.json")
    if not os.path.exists(coco_path):
        # BlenderProc2 may nest inside a datestamped subdirectory
        candidates = list(Path(output_dir).rglob("coco_annotations.json"))
        if candidates:
            coco_path = str(candidates[0])
        else:
            raise FileNotFoundError(
                f"{tag} COCO annotations not found under {output_dir}. "
                "Check BlenderProc2 output."
            )

    logger.info(f"{tag} COCO annotations written: {coco_path}")
    return coco_path


def _mock_coco_output(output_dir: str) -> None:
    """
    Write a minimal valid COCO JSON so downstream stages don't crash in dry-run.
    """
    import json
    os.makedirs(output_dir, exist_ok=True)
    coco = {
        "images": [
            {"id": i, "file_name": f"syn_{i:04d}.jpg", "width": 640, "height": 640}
            for i in range(5)
        ],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1,
             "bbox": [100, 100, 200, 300], "area": 60000, "iscrowd": 0}
            for i in range(5)
        ],
        "categories": [{"id": 1, "name": "TargetSKU", "supercategory": "product"}],
    }
    with open(os.path.join(output_dir, "coco_annotations.json"), "w") as f:
        json.dump(coco, f, indent=2)
