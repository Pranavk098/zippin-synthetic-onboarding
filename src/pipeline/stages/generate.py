"""
Stage 2: Synthetic Dataset Generation (BlenderProc2 + optional TripoSR mesh)

Orchestrates the full Stage 2 pipeline:

  mesh_source = "reconstructed" (config.yaml default after TripoSR integration):
    2a. mesh_generator.py  — TripoSR: product.jpg → raw/mesh.glb (cached)
    2b. mesh_normalizer.py — Blender headless cleanup → processed/mesh.glb
    2c. mesh_critic.py     — Quality gate (geometry + VLM check)
    2d. bproc_generator.py — Domain randomisation render (loads processed GLB)

  mesh_source = "procedural" (fallback / legacy):
    2a–2c skipped.
    2d. bproc_generator.py — Builds can/cube primitive, renders as before.

Design decision — BlenderProc2 subprocess:
  BlenderProc2 ships with its own embedded Python interpreter (Blender's bpy).
  It cannot be imported directly into a standard Python process. We invoke it
  via `blenderproc run src/rendering/bproc_generator.py <features_path>`.
  The features_path checkpoint acts as the data contract between stages.

  The reconstructed GLB path is passed to BlenderProc2 via BPROC_PROCESSED_GLB
  and BPROC_MESH_SOURCE environment variables — the same injection pattern used
  for all other rendering parameters.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _apply_deterministic_split(coco_json_path: str, val_fraction: float = 0.2) -> None:
    """
    Assign 'split' field to each image in the COCO JSON and copy val images
    to images/val/. Split is deterministic: last floor(n * val_fraction) images
    when sorted by file_name are validation; the rest are training.
    Modifies coco_json_path in-place.
    """
    import json as _json

    with open(coco_json_path) as f:
        data = _json.load(f)

    images = sorted(data["images"], key=lambda x: x["file_name"])
    if not images:
        logger.warning("[Generate] _apply_deterministic_split: no images in COCO JSON, skipping split.")
        return
    n_val = max(0, int(len(images) * val_fraction))
    val_ids = {img["id"] for img in images[-n_val:]}

    for img in data["images"]:
        img["split"] = "val" if img["id"] in val_ids else "train"

    # Copy val images to images/val/
    coco_dir = Path(coco_json_path).parent
    val_dir = coco_dir / "images" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    for img in data["images"]:
        if img["split"] == "val":
            src = coco_dir / "images" / "train" / Path(img["file_name"]).name
            if src.exists():
                shutil.copy2(str(src), str(val_dir / src.name))

    with open(coco_json_path, "w") as f:
        _json.dump(data, f, indent=2)

    logger.info(
        f"[Generate] Split applied: {len(images) - n_val} train / {n_val} val "
        f"(val_fraction={val_fraction:.2f})"
    )


def stage_generate(
    features_path: str,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    dry_run: bool = False,
    job_id: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    product_image_path: Optional[str] = None,
) -> str:
    """
    Invoke BlenderProc2 to render synthetic training images, optionally preceded
    by the TripoSR mesh reconstruction stage.

    Args:
        features_path:       Path to the Stage 1 SKU features JSON checkpoint.
        config:              Loaded config.yaml dict.
        checkpoint_dir:      Root output directory.
        dry_run:             Skip actual render; log intent only.
        job_id:              Optional ID for namespacing outputs.
        status_callback:     Callable receiving stage name updates for the API.
        product_image_path:  Path to the original product.jpg — required when
                             mesh_source == "reconstructed".

    Returns:
        Path to the COCO annotations JSON file.
    """
    import json as _json

    tag = f"[Stage 2: Generate{f'/{job_id}' if job_id else ''}]"

    if status_callback:
        status_callback("generate")

    output_dir = os.path.join(checkpoint_dir, job_id or "synthetic_dataset")

    if dry_run:
        logger.info(f"{tag} Dry-run — skipping BlenderProc2 render.")
        _mock_coco_output(output_dir)
        return os.path.join(output_dir, "coco_annotations.json")

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"{tag} Features checkpoint missing: {features_path}. Run Stage 1 first."
        )

    # ---- Derive SKU ID from job_id or features filename -----------------------
    sku_id = job_id or Path(features_path).stem

    # ---- Resolve mesh_source from config -------------------------------------
    mesh_source   = config.get("mesh_source", "procedural")
    processed_glb = ""   # filled in below for "reconstructed" path

    if mesh_source == "reconstructed":
        processed_glb = _run_reconstruction_stages(
            features_path=features_path,
            product_image_path=product_image_path,
            sku_id=sku_id,
            config=config,
            checkpoint_dir=checkpoint_dir,
            tag=tag,
        )
        if not processed_glb:
            # mesh_critic triggered procedural fallback
            logger.warning(f"{tag} Mesh reconstruction failed quality gates — "
                           "falling back to procedural mesh.")
            mesh_source = "procedural"

    # ---- Clean output dir before render -----------------------------------------
    images_dir = os.path.join(output_dir, "images")
    coco_json  = os.path.join(output_dir, "coco_annotations.json")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
        logger.info(f"{tag} Cleared stale renders: {images_dir}")
    if os.path.exists(coco_json):
        os.remove(coco_json)
        logger.info(f"{tag} Cleared stale COCO JSON: {coco_json}")

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
    resolution   = config.get("image_resolution", [640, 640])

    logger.info(
        f"{tag} Launching BlenderProc2 — mesh_source={mesh_source}, "
        f"{render_count} frames @ {resolution[0]}x{resolution[1]}"
    )

    env = os.environ.copy()
    env["BPROC_OUTPUT_DIR"]    = output_dir
    env["BPROC_RENDER_COUNT"]  = str(render_count)
    env["BPROC_RESOLUTION_W"]  = str(resolution[0])
    env["BPROC_RESOLUTION_H"]  = str(resolution[1])
    env["BPROC_MESH_SOURCE"]   = mesh_source
    env["BPROC_PROCESSED_GLB"] = processed_glb   # empty string when procedural

    if product_image_path:
        env["BPROC_PRODUCT_IMAGE"] = str(Path(product_image_path).resolve())

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
        candidates = list(Path(output_dir).rglob("coco_annotations.json"))
        if candidates:
            coco_path = str(candidates[0])
        else:
            raise FileNotFoundError(
                f"{tag} COCO annotations not found under {output_dir}. "
                "Check BlenderProc2 output."
            )

    logger.info(f"{tag} COCO annotations written: {coco_path}")

    val_fraction = config.get("val_fraction", 0.2)
    _apply_deterministic_split(coco_path, val_fraction=val_fraction)

    return coco_path


def _run_reconstruction_stages(
    features_path: str,
    product_image_path: Optional[str],
    sku_id: str,
    config: dict,
    checkpoint_dir: str,
    tag: str,
) -> str:
    """
    Run Stages 2a–2c: TripoSR generation → Blender normalisation → critic gate.

    Returns the path to the validated processed GLB, or an empty string if the
    mesh failed quality gates and the caller should fall back to procedural.
    """
    from ...rendering.mesh_generator import generate_mesh
    from ...rendering.mesh_critic import validate_mesh

    assets_root = os.path.join(checkpoint_dir, "assets")

    # Resolve product image — check env var then features directory
    if not product_image_path:
        product_image_path = os.environ.get("BPROC_PRODUCT_IMAGE", "")
    if not product_image_path or not os.path.exists(product_image_path):
        # Search relative to features checkpoint
        for candidate in [
            Path(features_path).parent / "product.jpg",
            Path(features_path).parent.parent / "product.jpg",
        ]:
            if candidate.exists():
                product_image_path = str(candidate)
                break

    if not product_image_path or not os.path.exists(product_image_path):
        logger.warning(
            f"{tag} product.jpg not found — cannot run TripoSR reconstruction. "
            "Falling back to procedural."
        )
        return ""

    # ---- Stage 2a: TripoSR mesh generation -----------------------------------
    logger.info(f"{tag} [2a] Running TripoSR mesh generation for SKU '{sku_id}'...")
    try:
        raw_glb = generate_mesh(
            image_path=product_image_path,
            sku_id=sku_id,
            assets_root=assets_root,
            backend=config.get("mesh_backend", "triposr_local"),
            resolution=config.get("mesh_resolution", 256),
            vram_budget_gb=config.get("mesh_vram_budget_gb", 6.0),
        )
        logger.info(f"{tag} [2a] Raw GLB: {raw_glb}")
    except Exception as e:
        logger.warning(f"{tag} [2a] TripoSR generation failed ({e}) — falling back to procedural.")
        return ""

    # ---- Stage 2b: Blender mesh normalisation --------------------------------
    logger.info(f"{tag} [2b] Running Blender mesh normalization...")
    processed_glb = _run_mesh_normalizer(
        raw_glb=raw_glb,
        sku_id=sku_id,
        assets_root=assets_root,
        tag=tag,
    )
    if not processed_glb:
        return ""

    # ---- Stage 2c: Quality gate ---------------------------------------------
    logger.info(f"{tag} [2c] Running mesh critic quality gate...")
    critic_result = validate_mesh(
        glb_path=processed_glb,
        product_image_path=product_image_path,
        sku_id=sku_id,
        ollama_url=config.get("ollama_url", "http://localhost:11434/api/generate"),
        vlm_model=config.get("vlm_model", "llava:7b"),
        max_retries=config.get("critic_max_retries", 2),
    )

    for w in critic_result.get("warnings", []):
        logger.warning(f"{tag} [2c] Critic warning: {w}")

    if critic_result.get("use_procedural"):
        logger.warning(f"{tag} [2c] Critic: procedural fallback triggered.")
        return ""

    logger.info(
        f"{tag} [2c] Critic passed — "
        f"layer1={critic_result['layer1_ok']}, layer2={critic_result['layer2_ok']}"
    )
    return processed_glb


def _run_mesh_normalizer(
    raw_glb: str,
    sku_id: str,
    assets_root: str,
    tag: str,
) -> str:
    """
    Invoke mesh_normalizer.py via headless Blender to produce processed/mesh.glb.
    Returns the processed GLB path, or empty string on failure.
    """
    blender_bin = shutil.which("blender")
    if not blender_bin:
        logger.warning(
            f"{tag} [2b] `blender` not found in PATH — mesh normalization skipped. "
            "The raw GLB will be used directly (may have sub-optimal geometry)."
        )
        # Use raw GLB directly — mesh_critic Layer 1 will still validate it
        return raw_glb

    normalizer_script = (
        Path(__file__).parent.parent.parent / "rendering" / "mesh_normalizer.py"
    )
    if not normalizer_script.exists():
        logger.warning(f"{tag} [2b] mesh_normalizer.py not found at {normalizer_script}.")
        return raw_glb

    processed_dir = Path(assets_root) / sku_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_glb = str(processed_dir / "mesh.glb")

    result = subprocess.run(
        [
            blender_bin,
            "--background",
            "--python", str(normalizer_script),
            "--",
            "--input",  raw_glb,
            "--output", processed_glb,
            "--height", "1.0",
        ],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        logger.warning(
            f"{tag} [2b] Blender normalizer exited {result.returncode}. "
            f"Stderr: {result.stderr[-300:]}. Using raw GLB as fallback."
        )
        return raw_glb

    if not os.path.exists(processed_glb):
        logger.warning(f"{tag} [2b] Processed GLB not found after normalizer run. Using raw.")
        return raw_glb

    logger.info(f"{tag} [2b] Normalised GLB: {processed_glb}")
    return processed_glb


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
