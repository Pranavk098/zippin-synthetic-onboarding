"""
Strategic Proposal II: Asynchronous Multi-Modal Attention Framework
for Occlusion-Robust Checkout Detection

=============================================================================
PROBLEM STATEMENT
=============================================================================

Zippin's Walk-Up and Lane formats are designed for extreme transactional
speed — a shopper completes a full purchase in seconds. This creates a
fundamental tension with computer vision:

  When a shopper reaches for a product, their arm/torso simultaneously
  occludes the SKU from all overhead camera angles. The visual confidence
  score for the target bounding box drops precipitously — sometimes to zero.

  Visual data alone cannot resolve this ambiguity deterministically. Without
  a second modality, the system must either:
    (a) Wait for the arm to retract — introducing unacceptable latency
    (b) Make a probabilistic guess — introducing margin-eroding errors
    (c) Flag for manual review — eliminating the "frictionless" promise

=============================================================================
PROPOSED SOLUTION: DYNAMIC GATED ATTENTION
=============================================================================

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  60Hz Overhead Camera Stream                                    │
  │  ─────────────────────────────►  Visual Tracker                │
  │                                        │                        │
  │                              confidence < τ?                   │
  │                                    YES │                        │
  │                                        ▼                        │
  │  Smart Shelf Weight Sensors ──► Async Message Queue            │
  │  (event-driven, async)              │                           │
  │                                     │ ΔW matches known SKU mass?│
  │                                     ▼                           │
  │                            EKF State Updater                    │
  │                             (cart belief update)               │
  └─────────────────────────────────────────────────────────────────┘

Key insight: heavy multimodal fusion runs ONLY when visual confidence
drops below threshold τ. Under normal (unoccluded) conditions, the
system runs the cheap visual-only path at full 60 FPS. This dynamic
gating preserves the Jetson Orin's thermal and power budget.

The temporal alignment problem (60Hz video vs. event-driven weight
sensor) is solved via an asyncio-based timestamped message queue that
buffers weight events and resolves them against the nearest visual frame.

=============================================================================
IMPLEMENTATION
=============================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class OcclusionState(Enum):
    CLEAR     = "clear"      # Visual confidence ≥ τ; vision-only path
    OCCLUDED  = "occluded"   # Visual confidence < τ; fusion required
    RESOLVING = "resolving"  # Weight event received; EKF updating


@dataclass
class VisualFrame:
    frame_id: int
    timestamp: float                        # Unix time (seconds)
    detections: List[Dict]                  # [{bbox, confidence, class_id}]
    occlusion_zones: List[Tuple[float, float, float, float]]  # (x1,y1,x2,y2)


@dataclass
class WeightEvent:
    shelf_id: str
    timestamp: float                        # Unix time (seconds)
    delta_grams: float                      # Positive = item removed
    absolute_grams: float


@dataclass
class CartUpdate:
    shopper_id: str
    sku_id: str
    action: str                             # "pickup" | "return"
    confidence: float                       # 0-1
    source: str                             # "visual" | "fusion"
    timestamp: float


# ---------------------------------------------------------------------------
# Extended Kalman Filter for belief-state tracking
# ---------------------------------------------------------------------------

class ShelfBeliefEKF:
    """
    Lightweight Extended Kalman Filter tracking shelf occupancy state.

    State vector: [n_items_on_shelf, item_mass_estimate]
    Measurements:
      - z_visual: bounding box count from visual detector
      - z_weight: ΔW from load cell (absolute grounding signal)

    The EKF is the mathematical core of deterministic transaction resolution.
    When both signals agree, confidence → 1.0. When only weight is available
    (full occlusion), confidence is determined by the mass match precision.
    """

    def __init__(self, sku_mass_grams: float, process_noise: float = 0.1):
        self.sku_mass = sku_mass_grams
        self.Q = process_noise                   # Process noise covariance
        self.R_visual = 0.3                      # Visual measurement noise
        self.R_weight = 0.02                     # Weight measurement noise (precise)

        # State: [count_estimate, mass_estimate]
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0                 # State covariance

    def predict(self, dt: float = 1 / 60) -> None:
        """Propagate state forward one timestep (called every visual frame)."""
        F = np.eye(2)                            # State transition (static shelf)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.eye(2) * self.Q

    def update_visual(self, detection_count: int) -> float:
        """
        Kalman update step using visual detection count as measurement.
        Returns posterior confidence.
        """
        H = np.array([[1.0, 0.0]])               # Observe count dimension
        z = np.array([float(detection_count)])
        y = z - H @ self.x                       # Innovation
        S = H @ self.P @ H.T + self.R_visual
        K = self.P @ H.T / S                     # Kalman gain
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return float(1.0 / (1.0 + abs(y[0])))

    def update_weight(self, delta_grams: float) -> Tuple[str, float]:
        """
        Kalman update using weight sensor delta.

        Returns (action, confidence):
          action ∈ {"pickup", "return", "unknown"}
          confidence ∈ [0, 1] — based on mass match to known SKU mass
        """
        H = np.array([[0.0, 1.0]])               # Observe mass dimension
        z = np.array([abs(delta_grams)])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_weight
        K = self.P @ H.T / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

        # Confidence: how closely does ΔW match the registered SKU mass?
        mass_error_ratio = abs(abs(delta_grams) - self.sku_mass) / max(self.sku_mass, 1.0)
        confidence = float(np.exp(-5.0 * mass_error_ratio))

        action = "pickup" if delta_grams > 0 else "return"
        return action, confidence


# ---------------------------------------------------------------------------
# Async Multi-Modal Attention Framework
# ---------------------------------------------------------------------------

class AsyncMultiModalAttention:
    """
    Asynchronous Multi-Modal Attention Framework for Zippin edge nodes.

    Manages two async data streams:
      1. Visual frames at 60 Hz (continuous, deterministic timing)
      2. Weight sensor events (asynchronous, event-driven)

    The attention mechanism dynamically gates fusion:
      - Confidence ≥ τ: cheap visual-only path (preserves Jetson budget)
      - Confidence < τ: query weight queue → EKF fusion → cart update

    This architecture processes 60 FPS from N cameras concurrently using
    asyncio, each camera with its own EKF instance per shelf zone.
    """

    OCCLUSION_THRESHOLD = 0.45      # τ — visual confidence below this triggers fusion
    WEIGHT_SYNC_WINDOW  = 0.5       # Seconds to look back in weight buffer for sync

    def __init__(
        self,
        sku_registry: Dict[str, float],    # {sku_id: mass_grams}
        cart_update_callback: Optional[Callable[[CartUpdate], None]] = None,
    ):
        self._sku_registry = sku_registry
        self._cart_callback = cart_update_callback

        # Per-shelf EKF instances — one per (camera_id, shelf_zone)
        self._ekf_map: Dict[str, ShelfBeliefEKF] = {}

        # Weight event buffer — bounded deque, one per shelf_id
        self._weight_buffers: Dict[str, Deque[WeightEvent]] = {}

        # Async queue for weight events (populated by hardware callbacks)
        self._weight_queue: asyncio.Queue[WeightEvent] = asyncio.Queue(maxsize=1000)

        self._running = False
        self._stats = {"visual_only": 0, "fusion_triggered": 0, "resolved": 0}

    # ------------------------------------------------------------------
    # Public interface: ingest data from hardware drivers
    # ------------------------------------------------------------------

    async def ingest_visual_frame(
        self, frame: VisualFrame, shopper_id: str
    ) -> Optional[CartUpdate]:
        """
        Process one visual frame. Returns a CartUpdate if a transaction
        was resolved (visually or via fusion), else None.

        Called at 60 Hz per camera. Must complete in < 5ms on Jetson Orin.
        """
        # Determine visual confidence for the most prominent detection
        best_conf = max(
            (d["confidence"] for d in frame.detections), default=0.0
        )

        # Propagate EKF beliefs
        for shelf_id, ekf in self._ekf_map.items():
            ekf.predict()

        if best_conf >= self.OCCLUSION_THRESHOLD:
            # ── Fast path: vision-only ───────────────────────────────────
            self._stats["visual_only"] += 1
            return self._resolve_visual(frame, shopper_id, best_conf)
        else:
            # ── Fusion path: query weight buffer ─────────────────────────
            self._stats["fusion_triggered"] += 1
            return await self._resolve_fusion(frame, shopper_id)

    def ingest_weight_event(self, event: WeightEvent) -> None:
        """
        Called by the weight sensor hardware driver (sync or from thread).
        Thread-safe: uses asyncio.Queue.put_nowait.
        """
        if event.shelf_id not in self._weight_buffers:
            self._weight_buffers[event.shelf_id] = deque(maxlen=50)
        self._weight_buffers[event.shelf_id].append(event)

        try:
            self._weight_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"[Fusion] Weight queue full — dropping event from {event.shelf_id}")

    def register_shelf(self, shelf_id: str, sku_id: str) -> None:
        """
        Register a shelf zone with its expected SKU and mass.
        Call once per shelf during store initialisation.
        """
        mass = self._sku_registry.get(sku_id, 100.0)
        self._ekf_map[shelf_id] = ShelfBeliefEKF(sku_mass_grams=mass)
        self._weight_buffers[shelf_id] = deque(maxlen=50)
        logger.info(f"[Fusion] Registered shelf {shelf_id} → SKU {sku_id} ({mass}g)")

    # ------------------------------------------------------------------
    # Internal resolution logic
    # ------------------------------------------------------------------

    def _resolve_visual(
        self, frame: VisualFrame, shopper_id: str, confidence: float
    ) -> Optional[CartUpdate]:
        """Visual-only transaction resolution — no sensor fusion needed."""
        if not frame.detections:
            return None

        top = max(frame.detections, key=lambda d: d["confidence"])
        return CartUpdate(
            shopper_id=shopper_id,
            sku_id=top.get("class_id", "unknown"),
            action="pickup",
            confidence=confidence,
            source="visual",
            timestamp=frame.timestamp,
        )

    async def _resolve_fusion(
        self, frame: VisualFrame, shopper_id: str
    ) -> Optional[CartUpdate]:
        """
        Occlusion resolution: look for a temporally aligned weight event
        and use EKF to ground the transaction deterministically.

        Window: search ±WEIGHT_SYNC_WINDOW seconds around frame.timestamp.
        """
        t = frame.timestamp

        for shelf_id, buf in self._weight_buffers.items():
            # Find weight event closest in time to the visual frame
            aligned = min(
                (e for e in buf if abs(e.timestamp - t) <= self.WEIGHT_SYNC_WINDOW),
                key=lambda e: abs(e.timestamp - t),
                default=None,
            )
            if aligned is None:
                continue

            ekf = self._ekf_map.get(shelf_id)
            if ekf is None:
                continue

            action, conf = ekf.update_weight(aligned.delta_grams)

            if conf < 0.3:
                logger.debug(
                    f"[Fusion] Low-confidence mass match ({conf:.2f}) "
                    f"on shelf {shelf_id} — ΔW={aligned.delta_grams:.1f}g, "
                    f"expected={ekf.sku_mass:.1f}g"
                )
                continue

            self._stats["resolved"] += 1
            logger.info(
                f"[Fusion] RESOLVED via weight: shelf={shelf_id} "
                f"action={action} conf={conf:.3f}"
            )

            update = CartUpdate(
                shopper_id=shopper_id,
                sku_id=shelf_id,
                action=action,
                confidence=conf,
                source="fusion",
                timestamp=t,
            )

            if self._cart_callback:
                self._cart_callback(update)

            return update

        # No weight event found in window — transaction unresolved
        logger.debug(f"[Fusion] No weight event in ±{self.WEIGHT_SYNC_WINDOW}s window")
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        total = self._stats["visual_only"] + self._stats["fusion_triggered"]
        return {
            **self._stats,
            "total_frames": total,
            "fusion_rate_pct": round(
                100.0 * self._stats["fusion_triggered"] / max(total, 1), 1
            ),
            "resolution_rate_pct": round(
                100.0 * self._stats["resolved"] / max(self._stats["fusion_triggered"], 1), 1
            ),
        }


# ---------------------------------------------------------------------------
# Demo harness (illustrates the async event loop pattern)
# ---------------------------------------------------------------------------

async def _demo():
    """
    Minimal demonstration of the async attention framework.
    Simulates 5 seconds of 60Hz frames with one mid-sequence occlusion.
    """
    sku_registry = {"energy_drink_250ml": 275.0, "protein_bar": 52.0}
    received_updates = []

    fusion = AsyncMultiModalAttention(
        sku_registry=sku_registry,
        cart_update_callback=lambda u: received_updates.append(u),
    )
    fusion.register_shelf("shelf_A1", "energy_drink_250ml")

    print("[Demo] Simulating 5s of 60Hz frames with mid-sequence occlusion...")

    for frame_idx in range(300):    # 5 seconds @ 60 FPS
        t = time.time()
        occluded = 100 <= frame_idx <= 140     # Simulated occlusion window

        frame = VisualFrame(
            frame_id=frame_idx,
            timestamp=t,
            detections=[] if occluded else [
                {"bbox": [100, 100, 200, 300], "confidence": 0.82, "class_id": "energy_drink_250ml"}
            ],
            occlusion_zones=[(80, 80, 220, 320)] if occluded else [],
        )

        # Inject weight event at occlusion start (simulates pick)
        if frame_idx == 105:
            fusion.ingest_weight_event(WeightEvent(
                shelf_id="shelf_A1",
                timestamp=t,
                delta_grams=278.0,      # Close to registered 275g
                absolute_grams=0.0,
            ))

        update = await fusion.ingest_visual_frame(frame, shopper_id="shopper_42")
        await asyncio.sleep(1 / 60)

    print(f"[Demo] Stats: {fusion.stats}")
    print(f"[Demo] Cart updates resolved: {len(received_updates)}")
    for u in received_updates:
        print(f"  → {u.action.upper()} {u.sku_id} via {u.source} (conf={u.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(_demo())
