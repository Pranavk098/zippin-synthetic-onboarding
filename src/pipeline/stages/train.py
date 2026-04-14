"""
Stage 3: YOLOv8n Fine-Tuning with Elastic Weight Consolidation

Trains YOLOv8 Nano on the BlenderProc2-generated synthetic dataset while
applying EWC regularization to prevent catastrophic forgetting of prior SKUs.

The EWC state (Fisher matrix + parameter means) is persisted alongside the
weights so it can be restored for the next SKU without re-computation.

Jetson Orin compatibility:
  - YOLOv8n runs at ~60 FPS on Jetson Orin NX in FP16 mode.
  - EWC overhead is a single forward pass per batch — negligible at edge.
  - INT8 TensorRT export is handled in a separate export step (see README).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def stage_train(
    coco_json: str,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    dry_run: bool = False,
    job_id: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Fine-tune YOLOv8n on synthetic data with EWC continual learning.

    Args:
        coco_json:       Path to the Stage 2 COCO annotations.
        config:          Loaded config.yaml dict.
        checkpoint_dir:  Root checkpoint directory.
        dry_run:         Skip training; log intent only.
        job_id:          Optional run namespace.
        status_callback: API progress hook.

    Returns:
        Path to the fine-tuned weights (.pt).
    """
    tag = f"[Stage 3: Train{f'/{job_id}' if job_id else ''}]"

    if status_callback:
        status_callback("train")

    if dry_run:
        logger.info(f"{tag} Dry-run — skipping training.")
        return os.path.join(checkpoint_dir, "new_sku_weights.pt")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            f"{tag} Missing dependency: {e}. Run: pip install torch ultralytics"
        ) from e

    from ...utils.coco_to_yolo import convert_coco_to_yolo
    from ...continual_learning import EWC
    from ...continual_learning.ewc_trainer import build_ewc_trainer

    # ---- 1. Convert COCO → YOLO dataset format --------------------------------
    run_dir = os.path.join(checkpoint_dir, job_id or "yolo_dataset")
    dataset_yaml = convert_coco_to_yolo(
        coco_json,
        output_dir=run_dir,
        class_names=["TargetSKU"],
    )

    # ---- 2. Load base model ---------------------------------------------------
    model_name = config.get("yolo_model", "yolov8n.pt")
    logger.info(f"{tag} Loading base model: {model_name}")
    model = YOLO(model_name)

    # ---- 3. EWC initialisation -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = config.get("ewc_lambda", 5000)
    ewc_state_path = os.path.join(checkpoint_dir, "ewc_state.pt")

    if os.path.exists(ewc_state_path):
        logger.info(f"{tag} Restoring EWC state from {ewc_state_path}")
        ewc = EWC.load(ewc_state_path, device)
    else:
        logger.info(
            f"{tag} No prior EWC state found — initialising with uniform Fisher "
            "(first SKU, no prior tasks to protect)."
        )
        ewc = EWC(model.model, dataloader=None, device=device, lam=lam)

    initial_penalty = ewc.penalty(model.model).item()
    logger.info(f"{tag} EWC initial penalty: {initial_penalty:.6f}  (λ={lam})")

    # ---- 4. Build EWC-aware trainer and fine-tune ----------------------------
    # This is the critical step: build_ewc_trainer returns a DetectionTrainer
    # subclass that overrides criterion() to inject the EWC penalty into every
    # gradient update — not just report it before/after training.
    EWCTrainerClass = build_ewc_trainer(ewc)

    epochs = config.get("train_epochs", 10)
    imgsz = config.get("image_resolution", [640, 640])[0]
    train_run_name = f"sku_run_{job_id or 'latest'}"

    train_kwargs = {
        "data": dataset_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "project": checkpoint_dir,
        "name": train_run_name,
        "exist_ok": True,
        "verbose": False,
        "plots": False,
        "trainer": EWCTrainerClass,   # ← EWC penalty injected per-batch
    }

    logger.info(f"{tag} Training YOLOv8n — epochs={epochs}, device={train_kwargs['device']}")
    model.train(**train_kwargs)

    post_penalty = ewc.penalty(model.model).item()
    logger.info(
        f"{tag} EWC post-training penalty: {post_penalty:.6f}  "
        f"(delta={post_penalty - initial_penalty:+.6f})"
    )

    # ---- 5. Export weights ---------------------------------------------------
    weights_src = os.path.join(checkpoint_dir, train_run_name, "weights", "best.pt")
    weights_dst = os.path.join(checkpoint_dir, f"{job_id or 'new_sku'}_weights.pt")

    if os.path.exists(weights_src):
        shutil.copy(weights_src, weights_dst)
    else:
        model.save(weights_dst)

    logger.info(f"{tag} Weights exported: {weights_dst}")

    # ---- 6. Consolidate EWC for next SKU onboarding -------------------------
    _consolidate_ewc(ewc, model, run_dir, device, ewc_state_path, tag)

    return weights_dst


def _consolidate_ewc(
    ewc, model, yolo_dir: str, device, ewc_state_path: str, tag: str
) -> None:
    """
    After training, build a DataLoader over the synthetic images and run
    EWC consolidation so the next SKU fine-tune is protected by the current
    Fisher matrix.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from pathlib import Path
    from PIL import Image
    import numpy as np

    img_dir = Path(yolo_dir) / "images" / "train"
    if not img_dir.exists():
        logger.warning(f"{tag} Consolidation skipped — image dir not found: {img_dir}")
        return

    class _SyntheticDS(Dataset):
        def __init__(self, folder, imgsz=640):
            self.paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            self.imgsz = imgsz

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB").resize(
                (self.imgsz, self.imgsz)
            )
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)
            return torch.from_numpy(arr)

    dataset = _SyntheticDS(img_dir)
    if len(dataset) == 0:
        logger.warning(f"{tag} No images in {img_dir} — EWC consolidation skipped.")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    ewc.consolidate(model.model, loader)
    ewc.save(ewc_state_path)
    logger.info(f"{tag} EWC consolidated and saved: {ewc_state_path}")
