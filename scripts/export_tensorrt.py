"""
TensorRT Export Script
Converts fine-tuned YOLOv8n weights to a TensorRT FP16 engine
for deployment on NVIDIA Jetson Orin NX.

Target performance: ~10-15ms per frame (60+ FPS) on Orin NX
vs ~35ms baseline (28 FPS) with native PyTorch.

Usage:
  python scripts/export_tensorrt.py --weights checkpoints/new_sku_weights.pt
  python scripts/export_tensorrt.py --weights checkpoints/new_sku_weights.pt --int8

Ref: https://docs.ultralytics.com/modes/export/#arguments
"""

import argparse
import os
import time

import torch


def export(weights_path: str, use_int8: bool = False, batch_size: int = 1) -> str:
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if not torch.cuda.is_available():
        print("[Export] WARNING: No CUDA GPU detected. TensorRT export requires CUDA.")
        print("[Export] On Jetson Orin, ensure JetPack 5.x is installed.")
        return ""

    model = YOLO(weights_path)

    precision = "int8" if use_int8 else "fp16"
    print(f"[Export] Converting to TensorRT {precision.upper()} engine...")
    print(f"[Export] Batch size: {batch_size} | Device: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    export_path = model.export(
        format="engine",
        half=(not use_int8),
        int8=use_int8,
        batch=batch_size,
        workspace=4,          # GB — respect Jetson Orin 8GB ceiling
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n[Export] Engine written: {export_path}")
    print(f"[Export] Export time: {elapsed:.1f}s")

    # Benchmark the exported engine
    _benchmark(export_path)
    return export_path


def _benchmark(engine_path: str, n_warmup: int = 5, n_runs: int = 100) -> None:
    """Measure FPS and latency of the exported TensorRT engine."""
    from ultralytics import YOLO
    import numpy as np

    print(f"\n[Benchmark] Warming up ({n_warmup} runs)...")
    model = YOLO(engine_path)
    dummy_img = "product.jpg" if os.path.exists("product.jpg") else None

    if dummy_img is None:
        print("[Benchmark] No product.jpg found — skipping latency benchmark.")
        return

    # Warmup
    for _ in range(n_warmup):
        model(dummy_img, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t = time.perf_counter()
        model(dummy_img, verbose=False)
        latencies.append((time.perf_counter() - t) * 1000)

    p50  = float(np.percentile(latencies, 50))
    p95  = float(np.percentile(latencies, 95))
    p99  = float(np.percentile(latencies, 99))
    fps  = 1000.0 / p50

    print(f"\n{'='*50}")
    print(f"  TensorRT Inference Benchmark ({n_runs} runs)")
    print(f"{'='*50}")
    print(f"  p50 latency : {p50:.2f}ms  ({fps:.1f} FPS)")
    print(f"  p95 latency : {p95:.2f}ms")
    print(f"  p99 latency : {p99:.2f}ms")
    print(f"  Target      : <17ms (60 FPS SLA)")
    print(f"  Status      : {'PASS' if p50 < 17 else 'FAIL — investigate layer fusion'}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8n to TensorRT")
    parser.add_argument("--weights", default="checkpoints/new_sku_weights.pt")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantization (requires calibration data)")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    export(args.weights, use_int8=args.int8, batch_size=args.batch)
