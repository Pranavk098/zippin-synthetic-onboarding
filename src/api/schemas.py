"""Pydantic schemas for the FastAPI onboarding service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OnboardStatus(BaseModel):
    job_id: str
    sku_name: str
    status: str = Field(..., description="queued | running | complete | failed")
    stage: Optional[str] = Field(None, description="Current pipeline stage")
    error: Optional[str] = None
    weights_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class SKUInfo(BaseModel):
    sku_id: str
    sku_name: str
    status: str
    created_at: str
    updated_at: str
    weights_path: Optional[str] = None
    ewc_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class EvalResult(BaseModel):
    map50: float = Field(..., description="COCO mAP @ IoU=0.50")
    map50_95: float = Field(..., description="COCO mAP @ IoU=0.50:0.95")
    n_images: int
    n_detections: int
    mean_confidence: float


class HealthResponse(BaseModel):
    status: str
    device: str
    skus_registered: int
    ewc_active: bool
