"""
Domain Invariance Score (DIS) — Sim2Real Feature Alignment Metric

Quantifies how well the synthetic render distribution approximates the visual
feature space of the original product photograph.  A low DIS means the
renderer is producing images that look nothing like the real product —
the Sim2Real gap is wide and the detector will struggle on shelf cameras.
A high DIS means synthetic and real images live in the same feature
neighbourhood — the trained weights will transfer.

Method
------
1. Load the clean product image (anchor) and all synthetic renders (queries).
2. Extract L2-normalised embeddings from each using CLIP ViT-B/32.
3. Compute pairwise cosine similarity: anchor · queryᵢ  (= dot product after
   L2-norm, range [−1, 1]).
4. Report the mean, std, min, p10, p25, p75, p90 of the similarity
   distribution plus flag any renders falling below the FAIL_THRESHOLD.

Empirical calibration
---------------------
  DIS ≥ 0.80  →  PASS  — tight Sim2Real alignment
  DIS 0.70–0.80 →  WARN  — acceptable; consider more DR passes
  DIS < 0.70  →  FAIL  — renders likely mis-coloured or mis-shaped;
                          check VLM attribute extraction (Stage 1)

Usage
-----
  python scripts/compute_dis.py \\
      --original  product.jpg \\
      --synthetic checkpoints/synthetic_dataset/images/ \\
      [--encoder clip-vit-b32 | resnet50] \\
      [--output   checkpoints/dis_report.json] \\
      [--top_k    5]          # Save top-K most-similar and bottom-K least-similar

Requirements
------------
  pip install transformers torch torchvision Pillow tqdm
  (CLIP encoder — downloads ~350 MB ViT-B/32 weights on first run)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Encoder back-ends
# ---------------------------------------------------------------------------

def _load_clip_encoder(device: str):
    """Load CLIP ViT-B/32 via HuggingFace transformers."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch

        print("[DIS] Loading CLIP ViT-B/32 (HuggingFace)...")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        def encode(image_paths: List[str]) -> np.ndarray:
            from PIL import Image
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()

        return encode

    except ImportError as e:
        raise ImportError(
            "transformers not installed. Run: pip install transformers"
        ) from e


def _load_resnet_encoder(device: str):
    """Fallback: ResNet-50 penultimate layer (no external model download)."""
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        print("[DIS] Loading ResNet-50 encoder (torchvision fallback)...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the classification head — use avg-pool features (2048-d)
        encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        encoder = encoder.to(device).eval()

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        def encode(image_paths: List[str]) -> np.ndarray:
            from PIL import Image
            import torch
            tensors = torch.stack([transform(Image.open(p).convert("RGB"))
                                   for p in image_paths]).to(device)
            with torch.no_grad():
                feats = encoder(tensors).squeeze(-1).squeeze(-1)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()

        return encode

    except ImportError as e:
        raise ImportError(
            "torchvision not installed. Run: pip install torchvision"
        ) from e


# ---------------------------------------------------------------------------
# Core DIS computation
# ---------------------------------------------------------------------------

def compute_dis(
    original_path: str,
    synthetic_dir: str,
    encoder_name: str = "clip-vit-b32",
    output_json: str | None = None,
    top_k: int = 5,
    batch_size: int = 16,
) -> Dict:
    """
    Compute the Domain Invariance Score for a synthetic render suite.

    Args:
        original_path:  Path to the clean product photo (anchor image).
        synthetic_dir:  Directory containing synthetic renders (.jpg/.png).
        encoder_name:   "clip-vit-b32" (recommended) or "resnet50" (fallback).
        output_json:    Optional path to write the full JSON report.
        top_k:          Number of best/worst renders to include in the report.
        batch_size:     Encoder batch size (reduce if OOM).

    Returns:
        dict with keys: dis_score, std, min, max, p10, p25, p75, p90,
                        status, n_renders, worst_renders, best_renders
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DIS] Device: {device}")

    # --- Encoder selection -------------------------------------------------------
    if encoder_name == "clip-vit-b32":
        encode = _load_clip_encoder(device)
    elif encoder_name == "resnet50":
        encode = _load_resnet_encoder(device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose clip-vit-b32 or resnet50.")

    # --- Validate inputs ---------------------------------------------------------
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original image not found: {original_path}")

    syn_dir = Path(synthetic_dir)
    if not syn_dir.is_dir():
        raise FileNotFoundError(f"Synthetic directory not found: {synthetic_dir}")

    render_paths = sorted(
        list(syn_dir.glob("*.jpg")) + list(syn_dir.glob("*.png"))
    )
    if not render_paths:
        raise ValueError(f"No .jpg/.png images found in {synthetic_dir}")

    n_renders = len(render_paths)
    print(f"[DIS] Anchor : {original_path}")
    print(f"[DIS] Renders: {n_renders} images in {synthetic_dir}")
    print(f"[DIS] Encoder: {encoder_name}")

    # --- Anchor embedding --------------------------------------------------------
    anchor_emb = encode([original_path])   # (1, D)

    # --- Batched render embeddings -----------------------------------------------
    all_embs: List[np.ndarray] = []
    render_str = [str(p) for p in render_paths]

    try:
        from tqdm import tqdm
        _range = lambda items, desc: tqdm(items, desc=desc)
    except ImportError:
        _range = lambda items, desc: items

    batches = [render_str[i:i + batch_size]
               for i in range(0, len(render_str), batch_size)]

    for batch in _range(batches, "Encoding renders"):
        embs = encode(batch)    # (B, D)
        all_embs.append(embs)

    render_embs = np.concatenate(all_embs, axis=0)   # (N, D)

    # --- Cosine similarities (anchor already L2-normed, renders too) -------------
    similarities = (render_embs @ anchor_emb.T).squeeze()   # (N,)

    # --- Statistics --------------------------------------------------------------
    dis_mean = float(np.mean(similarities))
    dis_std  = float(np.std(similarities))
    dis_min  = float(np.min(similarities))
    dis_max  = float(np.max(similarities))

    percentiles = np.percentile(similarities, [10, 25, 75, 90])

    if dis_mean >= 0.80:
        status = "PASS"
        status_msg = "Tight Sim2Real alignment — synthetic distribution closely matches real product."
    elif dis_mean >= 0.70:
        status = "WARN"
        status_msg = ("Moderate alignment. Consider increasing render count or "
                      "improving texture projection from Stage 1 VLM attributes.")
    else:
        status = "FAIL"
        status_msg = ("Poor Sim2Real alignment. Check VLM attribute extraction — "
                      "colours/shape may be incorrect. Inspect bottom-K renders below.")

    # --- Top-K / Bottom-K render paths -------------------------------------------
    sorted_idx = np.argsort(similarities)
    worst_k = [{"path": str(render_paths[i]), "similarity": float(similarities[i])}
               for i in sorted_idx[:top_k]]
    best_k  = [{"path": str(render_paths[i]), "similarity": float(similarities[i])}
               for i in sorted_idx[-top_k:][::-1]]

    # --- n_below_threshold -------------------------------------------------------
    n_below_07 = int(np.sum(similarities < 0.70))
    n_below_08 = int(np.sum(similarities < 0.80))

    report = {
        "dis_score":        round(dis_mean, 4),
        "std":              round(dis_std,  4),
        "min":              round(dis_min,  4),
        "max":              round(dis_max,  4),
        "p10":              round(float(percentiles[0]), 4),
        "p25":              round(float(percentiles[1]), 4),
        "p75":              round(float(percentiles[2]), 4),
        "p90":              round(float(percentiles[3]), 4),
        "status":           status,
        "status_message":   status_msg,
        "n_renders":        n_renders,
        "n_below_0.70":     n_below_07,
        "n_below_0.80":     n_below_08,
        "encoder":          encoder_name,
        "anchor_image":     original_path,
        "worst_renders":    worst_k,
        "best_renders":     best_k,
    }

    # --- Print summary -----------------------------------------------------------
    print(f"\n{'=' * 56}")
    print(f"  Domain Invariance Score (DIS)  —  {encoder_name}")
    print(f"{'=' * 56}")
    print(f"  DIS (mean cosine sim) : {dis_mean:.4f}   [{status}]")
    print(f"  Std deviation         : {dis_std:.4f}")
    print(f"  Range                 : [{dis_min:.4f}, {dis_max:.4f}]")
    print(f"  p10 / p25 / p75 / p90 : "
          f"{percentiles[0]:.3f} / {percentiles[1]:.3f} / "
          f"{percentiles[2]:.3f} / {percentiles[3]:.3f}")
    print(f"  n_renders             : {n_renders}")
    print(f"  n < 0.70 (FAIL band)  : {n_below_07}")
    print(f"  n < 0.80 (WARN band)  : {n_below_08}")
    print(f"\n  Status: {status_msg}")
    print(f"\n  Worst {top_k} renders (lowest similarity):")
    for item in worst_k:
        print(f"    {item['similarity']:.4f}  {item['path']}")
    print(f"\n  Best {top_k} renders (highest similarity):")
    for item in best_k:
        print(f"    {item['similarity']:.4f}  {item['path']}")
    print(f"{'=' * 56}\n")

    # --- Persist report ----------------------------------------------------------
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[DIS] Full report saved: {output_json}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Domain Invariance Score (DIS) for a synthetic render suite."
    )
    parser.add_argument(
        "--original", default="product.jpg",
        help="Path to the clean product image (anchor). Default: product.jpg",
    )
    parser.add_argument(
        "--synthetic", default="checkpoints/synthetic_dataset/images/",
        help="Directory containing synthetic renders.",
    )
    parser.add_argument(
        "--encoder", default="clip-vit-b32",
        choices=["clip-vit-b32", "resnet50"],
        help="Feature encoder backend. clip-vit-b32 recommended (default).",
    )
    parser.add_argument(
        "--output", default="checkpoints/dis_report.json",
        help="Output path for JSON report.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of best/worst renders to include in report.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Encoder batch size (reduce to 4–8 if VRAM is limited).",
    )

    args = parser.parse_args()

    report = compute_dis(
        original_path = args.original,
        synthetic_dir = args.synthetic,
        encoder_name  = args.encoder,
        output_json   = args.output,
        top_k         = args.top_k,
        batch_size    = args.batch_size,
    )

    # Exit with non-zero code if DIS fails — enables CI gate
    if report["status"] == "FAIL":
        sys.exit(1)
