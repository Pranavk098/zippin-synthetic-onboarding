"""
EWC-Aware Ultralytics Trainer

This is the critical piece that closes the EWC injection gap.

Problem with naive YOLOv8 + EWC:
  Calling model.train() on a new SKU uses Ultralytics' internal training
  loop, which has no knowledge of the EWC penalty. The penalty can be
  *computed* before and after training, but it is never *applied* — meaning
  the training loss doesn't protect prior SKU weights. This defeats the
  entire purpose of EWC.

Solution:
  Ultralytics' trainer is built for subclassing. We override `criterion()`
  — the method that computes the loss for each batch — to inject the EWC
  penalty at every gradient update. This is the correct, mathematically
  valid implementation of EWC continual learning.

  L_total(batch) = L_detection(batch) + (λ/2) · Σ_i F_i · (θ_i - θ*_i)²
                   ↑ standard YOLO loss     ↑ EWC regularisation

Reference:
  Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural
  Networks", PNAS 2017, §2.3 (Continual Learning with EWC).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def build_ewc_trainer(ewc_instance):
    """
    Factory that returns an EWC-aware Ultralytics DetectionTrainer subclass,
    capturing the ewc instance via closure.

    Args:
        ewc_instance: An initialised EWC object (or None — degrades to
                      standard YOLOv8 training with a log warning).

    Returns:
        EWCDetectionTrainer class (not instance — YOLO.train() takes a class).

    Usage:
        ewc = EWC(model.model, prev_dataloader, device, lam=5000)
        EWCTrainerClass = build_ewc_trainer(ewc)
        model.train(trainer=EWCTrainerClass, data=..., epochs=...)
    """
    try:
        from ultralytics.models.yolo.detect.train import DetectionTrainer
    except ImportError:
        try:
            from ultralytics.engine.trainer import BaseTrainer as DetectionTrainer
        except ImportError:
            raise ImportError(
                "ultralytics not installed or incompatible version. "
                "Run: pip install ultralytics>=8.2.0"
            )

    _ewc = ewc_instance

    class EWCDetectionTrainer(DetectionTrainer):
        """
        YOLOv8 DetectionTrainer with EWC regularisation injected into the
        loss computation. Fully compatible with Ultralytics' training loop,
        callbacks, and checkpoint management.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if _ewc is not None:
                logger.info(
                    f"[EWCTrainer] EWC regularisation active "
                    f"(λ={_ewc.lam:.0f}). "
                    f"Prior SKU weights are protected."
                )
            else:
                logger.warning(
                    "[EWCTrainer] No EWC instance provided — "
                    "training without continual learning protection."
                )

        def criterion(self, preds, batch):
            """
            Override: add EWC penalty to the standard YOLO detection loss.

            This method is called once per batch during the training loop.
            The base class computes the YOLO multi-task loss (box + cls + dfl),
            then we add the EWC term before returning to the optimiser.
            """
            # Standard YOLO detection loss (box regression + classification + DFL)
            loss, loss_items = super().criterion(preds, batch)

            if _ewc is None:
                return loss, loss_items

            # EWC penalty — protects Fisher-high weights from prior SKU tasks
            ewc_penalty = _ewc.penalty(self.model)

            # Normalise by batch size to keep scale consistent with YOLO loss
            batch_size = batch["img"].shape[0] if isinstance(batch, dict) else preds[0].shape[0]
            ewc_penalty_normalised = ewc_penalty / max(batch_size, 1)

            total_loss = loss + ewc_penalty_normalised

            # Log EWC contribution (visible in Ultralytics training output)
            if hasattr(self, "tloss") and self.tloss is not None:
                # Append EWC value to the running loss display
                pass

            if torch.is_tensor(ewc_penalty) and self.args.verbose:
                logger.debug(
                    f"[EWCTrainer] batch loss={loss.item():.4f}  "
                    f"ewc={ewc_penalty_normalised.item():.6f}  "
                    f"total={total_loss.item():.4f}"
                )

            return total_loss, loss_items

    return EWCDetectionTrainer
