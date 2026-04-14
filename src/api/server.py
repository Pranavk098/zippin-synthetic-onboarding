"""
Zippin Zero-Shot SKU Onboarding — REST API

FastAPI server that wraps the pipeline as an async background job queue.
Designed for corporate deployment where:
  - Retail ops team uploads product images via POST /onboard
  - ML platform polls GET /jobs/{id} for async completion
  - Edge deployment team fetches GET /skus/{id}/metrics for SLA validation
  - Monitoring hits GET /health for Kubernetes liveness probes

Horizontal scaling:
  In production, pin the SKU registry to a shared persistent volume (NFS or
  S3-backed) and run N replicas behind a load balancer. The registry is
  thread-safe and uses file-based locking adequate for PoC and small clusters.
  For high-throughput production, replace with Redis or PostgreSQL backend.

Start with:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .schemas import EvalResult, HealthResponse, OnboardStatus, SKUInfo
from ..utils.sku_registry import SKURegistry

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Zippin Zero-Shot SKU Onboarding API",
    description=(
        "End-to-end pipeline: single product image → "
        "VLM semantic extraction → physics-based synthetic data generation → "
        "YOLOv8n + EWC continual learning → Jetson-ready edge weights."
    ),
    version="2.0.0",
    docs_url="/",       # Swagger UI at root
    redoc_url="/redoc",
)

_CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
_registry = SKURegistry(os.path.join(_CHECKPOINT_DIR, "sku_registry.json"))


# ---------------------------------------------------------------------------
# Onboarding endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/onboard",
    response_model=OnboardStatus,
    status_code=202,
    summary="Upload a product image and trigger the onboarding pipeline",
    tags=["Pipeline"],
)
async def onboard_sku(
    background_tasks: BackgroundTasks,
    sku_name: str,
    image: UploadFile = File(..., description="JPEG/PNG reference product image"),
):
    """
    **Zero-shot onboarding flow (async)**:
    1. Persist the uploaded image.
    2. Register the job in the SKU registry.
    3. Kick off the pipeline in a background thread.
    4. Return 202 Accepted with a `job_id` for polling.

    Typical end-to-end time: **2–8 minutes** depending on render count and GPU.
    """
    job_id = uuid.uuid4().hex[:8]

    upload_dir = Path(_CHECKPOINT_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Persist upload
    safe_filename = Path(image.filename).name if image.filename else "product.jpg"
    image_path = upload_dir / safe_filename
    content = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)

    # Register
    _registry.register(sku_name=sku_name, sku_id=job_id)

    # Async pipeline
    background_tasks.add_task(
        _run_pipeline_task, job_id=job_id, sku_name=sku_name, image_path=str(image_path)
    )

    logger.info(f"[API] Job {job_id} queued for SKU '{sku_name}'")
    return OnboardStatus(job_id=job_id, sku_name=sku_name, status="queued")


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get(
    "/jobs/{job_id}",
    response_model=OnboardStatus,
    summary="Poll pipeline status",
    tags=["Pipeline"],
)
async def get_job(job_id: str):
    entry = _registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return OnboardStatus(**_entry_to_status(entry))


# ---------------------------------------------------------------------------
# SKU catalogue
# ---------------------------------------------------------------------------

@app.get(
    "/skus",
    response_model=List[SKUInfo],
    summary="List all onboarded SKUs",
    tags=["SKU Catalogue"],
)
async def list_skus():
    return [SKUInfo(**e) for e in _registry.list_all()]


@app.get(
    "/skus/{sku_id}/metrics",
    response_model=EvalResult,
    summary="Get Sim2Real evaluation metrics for a SKU",
    tags=["SKU Catalogue"],
)
async def get_metrics(sku_id: str):
    entry = _registry.get(sku_id)
    if entry is None:
        raise HTTPException(404, f"SKU '{sku_id}' not found.")
    metrics = entry.get("metrics")
    if not metrics:
        raise HTTPException(404, f"No metrics yet for SKU '{sku_id}'. Pipeline may still be running.")
    return EvalResult(**metrics)


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Kubernetes liveness / readiness probe",
    tags=["Ops"],
)
async def health():
    return HealthResponse(
        status="ok",
        device="cuda" if torch.cuda.is_available() else "cpu",
        skus_registered=len(_registry.list_all()),
        ewc_active=os.path.exists(os.path.join(_CHECKPOINT_DIR, "ewc_state.pt")),
    )


@app.delete(
    "/jobs/{job_id}",
    status_code=204,
    summary="Delete a job entry from the registry",
    tags=["Ops"],
)
async def delete_job(job_id: str):
    entry = _registry.get(job_id)
    if entry is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    # Mark as deleted rather than removing; preserves EWC consolidation history
    _registry.update(job_id, status="deleted")
    return JSONResponse(status_code=204, content=None)


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

async def _run_pipeline_task(job_id: str, sku_name: str, image_path: str) -> None:
    """
    Runs the pipeline in a thread pool to avoid blocking the event loop.
    Updates registry at each stage transition.
    """
    from ..pipeline.orchestrator import run_pipeline, load_config

    def _status_hook(stage: str):
        _registry.update(job_id, status="running", stage=stage)

    _registry.update(job_id, status="running", stage="extract")

    try:
        config = load_config()
        result = await asyncio.to_thread(
            run_pipeline,
            sku_name=sku_name,
            image_path=image_path,
            stages=("extract", "generate", "train", "eval"),
            checkpoint_dir=_CHECKPOINT_DIR,
            job_id=job_id,
            status_callback=_status_hook,
        )
        _registry.update(
            job_id,
            status="complete",
            stage=None,
            weights_path=result.get("weights_path"),
            ewc_path=result.get("ewc_path"),
            metrics=result.get("metrics"),
        )
        logger.info(f"[API] Job {job_id} completed successfully.")

    except Exception as exc:
        logger.exception(f"[API] Job {job_id} failed: {exc}")
        _registry.update(job_id, status="failed", stage=None, error=str(exc))


def _entry_to_status(entry: dict) -> dict:
    return {
        "job_id": entry["sku_id"],
        "sku_name": entry["sku_name"],
        "status": entry["status"],
        "stage": entry.get("stage"),
        "error": entry.get("error"),
        "weights_path": entry.get("weights_path"),
        "metrics": entry.get("metrics"),
    }
