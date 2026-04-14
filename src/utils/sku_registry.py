"""
SKU Registry — persistent state management for multi-SKU continual learning.

Tracks every onboarded SKU, its associated weights, Fisher state, and eval
metrics so the pipeline can be extended horizontally across thousands of SKUs
without re-training from scratch.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class SKURegistry:
    """
    Thread-safe JSON-backed registry for onboarded SKUs.

    Schema per entry:
        {
            "sku_id":      str,          # UUID-based job ID
            "sku_name":    str,          # Human-readable product name
            "status":      str,          # queued | running | complete | failed
            "stage":       str | null,   # Current pipeline stage
            "created_at":  ISO-8601,
            "updated_at":  ISO-8601,
            "weights_path": str | null,  # Path to fine-tuned .pt file
            "ewc_path":    str | null,   # Path to EWC state dict
            "metrics":     dict | null,  # Eval results from stage_eval
            "error":       str | null    # Error message on failure
        }
    """

    def __init__(self, registry_path: str = "checkpoints/sku_registry.json"):
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, sku_name: str, sku_id: str) -> dict:
        entry = {
            "sku_id": sku_id,
            "sku_name": sku_name,
            "status": "queued",
            "stage": None,
            "created_at": _now(),
            "updated_at": _now(),
            "weights_path": None,
            "ewc_path": None,
            "metrics": None,
            "error": None,
        }
        with self._lock:
            self._data[sku_id] = entry
            self._persist()
        return entry

    def update(self, sku_id: str, **kwargs) -> Optional[dict]:
        with self._lock:
            if sku_id not in self._data:
                return None
            self._data[sku_id].update(kwargs)
            self._data[sku_id]["updated_at"] = _now()
            self._persist()
            return self._data[sku_id].copy()

    def get(self, sku_id: str) -> Optional[dict]:
        with self._lock:
            return self._data.get(sku_id)

    def list_all(self) -> List[dict]:
        with self._lock:
            return list(self._data.values())

    def get_metrics(self, sku_id: str) -> Optional[dict]:
        entry = self.get(sku_id)
        return entry.get("metrics") if entry else None

    def find_by_name(self, sku_name: str) -> Optional[dict]:
        with self._lock:
            for entry in self._data.values():
                if entry["sku_name"] == sku_name:
                    return entry
        return None

    def complete_skus(self) -> List[dict]:
        return [e for e in self.list_all() if e["status"] == "complete"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, dict]:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _persist(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
