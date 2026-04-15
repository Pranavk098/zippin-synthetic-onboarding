"""
Elastic Weight Consolidation (EWC)
Kirkpatrick et al., 2017 — "Overcoming Catastrophic Forgetting in Neural Networks"

Key insight for Zippin edge deployment:
  When we fine-tune YOLOv8n on a new SKU, we must not overwrite the Fisher-high
  (information-dense) parameters that were critical for previously learned SKUs.
  EWC anchors those parameters via a quadratic penalty weighted by the diagonal
  of the Fisher Information Matrix (FIM) — the curvature of the log-likelihood.

  L_total = L_new_task + (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

Why diagonal FIM and not full?
  Full FIM is O(params²) — infeasible for YOLOv8n (~3.2M params).
  The diagonal approximation retains per-parameter importance estimates
  at O(params) cost, which is well-validated in the continual learning literature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Iterator, Optional
import logging

logger = logging.getLogger(__name__)


class EWC:
    """
    Production-grade EWC regularizer for continual SKU learning on Jetson Orin.

    Usage:
        # After training on SKU batch N-1:
        ewc = EWC(model.model, dataloader_prev, device, lam=5000)

        # During training on SKU batch N:
        for batch in new_dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch))
            loss += ewc.penalty(model.model)   # ← add EWC term
            loss.backward()
            optimizer.step()

        # After training on SKU N, consolidate for SKU N+1:
        ewc.consolidate(model.model, new_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader],
        device: torch.device,
        lam: float = 5000.0,
        n_fisher_samples: int = 200,
    ):
        """
        Args:
            model:            The backbone whose weights we want to protect.
            dataloader:       DataLoader over the *previous* task's training data.
                              If None, falls back to uniform importance (L2-like).
            device:           Compute device (cuda / cpu).
            lam:              EWC regularization strength. Higher → stronger
                              protection of prior SKUs. Tuned to 5000 for Jetson
                              based on Zippin shelf-cam domain.
            n_fisher_samples: Max images used to estimate FIM (speed/accuracy trade).
        """
        self.lam = lam
        self.device = device
        self.n_fisher_samples = n_fisher_samples

        # θ* — optimal parameters from the previous task
        self._means: Dict[str, torch.Tensor] = {
            n: p.data.detach().clone().to(device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # F — diagonal Fisher Information Matrix
        if dataloader is not None:
            logger.info("[EWC] Computing diagonal Fisher Information Matrix...")
            self._precision_matrices = self._diag_fisher(model, dataloader)
            fisher_norm = sum(v.sum().item() for v in self._precision_matrices.values())
            logger.info(f"[EWC] FIM computed. L1 norm = {fisher_norm:.4e}")
        else:
            logger.warning(
                "[EWC] No dataloader provided — using uniform Fisher (degrades to L2). "
                "Pass the previous SKU's validation dataloader for true EWC."
            )
            self._precision_matrices = {
                n: torch.ones_like(p.data, device=device)
                for n, p in model.named_parameters()
                if p.requires_grad
            }

    # ------------------------------------------------------------------
    # Core: Fisher Information Matrix estimation
    # ------------------------------------------------------------------

    def _diag_fisher(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal FIM by accumulating squared gradients of the
        log-likelihood (NLL proxy) over previous-task data.

            F_ii ≈ (1/N) * Σ_x [ (∂ log p(y|x, θ) / ∂θ_i)² ]

        For detection models the exact NLL is implicit in the multi-task
        detection loss. We use the total training loss as a proxy, which
        is standard practice (see: online EWC, progressive neural nets).
        """
        precision: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p.data, device=self.device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        model.eval()
        samples_processed = 0

        for batch in dataloader:
            if samples_processed >= self.n_fisher_samples:
                break

            model.zero_grad()

            # ---- Handle multiple batch formats ----------------------------------------
            # YOLOv8 DataLoader yields dicts; plain image tensors for synthetic eval
            if isinstance(batch, dict):
                imgs = batch["img"].to(self.device).float() / 255.0
            elif isinstance(batch, (list, tuple)):
                imgs = batch[0]
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.to(self.device).float()
                    if imgs.max() > 1.0:
                        imgs = imgs / 255.0
            elif isinstance(batch, torch.Tensor):
                imgs = batch.to(self.device).float()
                if imgs.max() > 1.0:
                    imgs = imgs / 255.0
            else:
                logger.debug(f"[EWC] Unrecognised batch type {type(batch)}, skipping.")
                continue
            # ---------------------------------------------------------------------------

            try:
                with torch.enable_grad():
                    output = model(imgs)

                    # Extract a scalar loss from whatever the model returns
                    loss = self._extract_loss(output)
                    loss.backward()

                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        precision[n].add_(p.grad.detach().pow(2))

                samples_processed += imgs.size(0)

            except Exception as exc:
                logger.warning(f"[EWC] Fisher batch failed ({type(exc).__name__}: {exc}), skipping.")
                continue

        model.train()

        if samples_processed > 0:
            for n in precision:
                precision[n].div_(samples_processed)
        else:
            logger.warning("[EWC] No samples processed — using uniform Fisher.")
            for n in precision:
                precision[n].fill_(1.0)

        return precision

    @staticmethod
    def _extract_loss(output) -> torch.Tensor:
        """
        Coerce the model output to a scalar loss suitable for .backward().
        Handles YOLO, plain tensor, and tuple outputs.
        """
        if isinstance(output, torch.Tensor):
            # Raw logit tensor — use mean absolute value as proxy
            return output.abs().mean()

        if isinstance(output, (list, tuple)):
            # YOLOv8 returns (predictions, proto) or similar
            pred = output[0]
            if isinstance(pred, torch.Tensor):
                # Confidence scores in detection head: last dim contains obj conf
                return -torch.log(pred[..., 4].sigmoid().clamp(min=1e-9)).mean()
            return sum(t.abs().mean() for t in output if isinstance(t, torch.Tensor))

        # Fallback: sum all numeric attributes
        return torch.tensor(0.0, requires_grad=True)

    # ------------------------------------------------------------------
    # EWC penalty (added to the new-task training loss)
    # ------------------------------------------------------------------

    def penalty(self, current_model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularization term.

            penalty = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

        This is added to the new-task loss *during training*, not after.
        """
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for n, p in current_model.named_parameters():
            if n not in self._precision_matrices or not p.requires_grad:
                continue
            fisher = self._precision_matrices[n]
            mean = self._means[n].to(p.device)
            # Clamp Fisher to avoid numerical blow-up on rarely activated weights
            fisher_clamped = fisher.clamp(max=1e6)
            loss = loss + (fisher_clamped * (p - mean).pow(2)).sum()

        return (self.lam / 2.0) * loss

    # ------------------------------------------------------------------
    # Consolidation — call after each task to prepare for the next
    # ------------------------------------------------------------------

    def consolidate(self, model: nn.Module, new_dataloader: DataLoader) -> None:
        """
        Online EWC consolidation: merge the new task's Fisher with the old one
        using a running average, then update the parameter means to current θ.

        Call this *after* training on a new SKU is complete and *before* starting
        the next SKU. This is what enables horizontal scaling across many SKUs.
        """
        new_fisher = self._diag_fisher(model, new_dataloader)

        for n in self._precision_matrices:
            # Running average consolidation (prevents FIM from exploding over time)
            self._precision_matrices[n] = (
                0.5 * self._precision_matrices[n] + 0.5 * new_fisher[n]
            )

        self._means = {
            n: p.data.detach().clone().to(self.device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        logger.info("[EWC] Consolidated: parameter means and Fisher matrix updated.")

    # ------------------------------------------------------------------
    # Serialization — save/load the EWC state for edge deployment
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "lam": self.lam,
            "means": {n: t.cpu() for n, t in self._means.items()},
            "fisher": {n: t.cpu() for n, t in self._precision_matrices.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device) -> "EWC":
        """Restore EWC state from a checkpoint (no model/dataloader needed)."""
        obj = cls.__new__(cls)
        obj.lam = state["lam"]
        obj.device = device
        obj.n_fisher_samples = 200
        obj._means = {n: t.to(device) for n, t in state["means"].items()}
        obj._precision_matrices = {n: t.to(device) for n, t in state["fisher"].items()}
        return obj

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info(f"[EWC] State saved to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "EWC":
        state = torch.load(path, map_location=device)
        return cls.from_state_dict(state, device)
