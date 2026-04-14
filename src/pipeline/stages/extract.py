"""
Stage 1: Semantic Extraction via VLM (Ollama + LLaVA)

Connects to a locally running Ollama inference server, sends the product
image as base64, and extracts structured JSON attributes used to procedurally
configure the BlenderProc2 scene (shape, material, colors, dimensions).

Why Ollama over vLLM?
  - Ollama's 4-bit GGUF quantization fits under the 8GB VRAM ceiling on
    NVIDIA Jetson Orin NX without OOM errors.
  - Single-command pull model, no custom CUDA kernels to compile.
  - Natively supports Windows + Linux dev environments with identical API.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FALLBACK_ATTRS = {
    "shape": "cylinder",
    "material": "glossy aluminum",
    "primary_colors": ["red", "silver"],
    "dimensions_estimate": {"height": "12cm", "width": "6cm", "depth": "6cm"},
}

_EXTRACTION_PROMPT = """
Analyze this retail product image carefully.
Return ONLY a valid JSON object with these exact keys — no markdown, no prose:
{
  "shape": "<cylinder|box|bag|bottle|other>",
  "material": "<e.g. glossy aluminum, matte cardboard, clear plastic>",
  "primary_colors": ["<color1>", "<color2>"],
  "dimensions_estimate": {
    "height": "<estimate with unit>",
    "width": "<estimate with unit>",
    "depth": "<estimate with unit>"
  }
}
""".strip()


def stage_extract(
    image_path: str,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    dry_run: bool = False,
    job_id: Optional[str] = None,
) -> dict:
    """
    Extract semantic attributes from a product image via LLaVA.

    Returns the parsed attribute dict and persists it to
    {checkpoint_dir}/{job_id or 'sku_features'}.json.
    """
    tag = f"[Stage 1: Extract{f'/{job_id}' if job_id else ''}]"

    if dry_run:
        logger.info(f"{tag} Dry-run mode — using fallback attributes.")
        _save_checkpoint(checkpoint_dir, job_id, _FALLBACK_ATTRS)
        return _FALLBACK_ATTRS

    try:
        import httpx
    except ImportError:
        logger.error(f"{tag} httpx not installed. Run: pip install httpx")
        raise

    try:
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"{tag} Cannot read image: {image_path}") from exc

    model_name = config.get("vlm_model", "llava:7b")
    ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")
    timeout = config.get("vlm_timeout_secs", 90.0)

    logger.info(f"{tag} Querying {model_name} at {ollama_url} ...")

    try:
        resp = httpx.post(
            ollama_url,
            json={"model": model_name, "prompt": _EXTRACTION_PROMPT,
                  "images": [b64_image], "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["response"]

        # Strip markdown code fences if the model wraps the JSON
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        attrs = json.loads(raw)

        logger.info(f"{tag} Extracted attributes: {attrs}")

    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
        logger.warning(f"{tag} Ollama call failed ({exc}). Using fallback attributes.")
        attrs = _FALLBACK_ATTRS

    _save_checkpoint(checkpoint_dir, job_id, attrs)
    return attrs


def _save_checkpoint(checkpoint_dir: str, job_id: Optional[str], attrs: dict) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    name = f"{job_id}_sku_features" if job_id else "sku_features"
    path = os.path.join(checkpoint_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(attrs, f, indent=2)
    logger.info(f"[Stage 1: Extract] Checkpoint saved: {path}")
    return path
