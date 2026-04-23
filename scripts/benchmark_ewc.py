"""
EWC vs Naive Fine-tuning Benchmark — 3-SKU sequential onboarding.

Usage:
    python scripts/benchmark_ewc.py

Requires:
    - skus/sku1.jpg, skus/sku2.jpg, skus/sku3.jpg
    - BlenderProc2 installed and on PATH
    - Ollama + LLaVA-7B running (for Stage 1 extraction)
    - GPU with sufficient VRAM for YOLOv8n training

Output:
    docs/benchmark_results/ewc_vs_naive.csv
    docs/benchmark_results/retention_curve.png
    docs/benchmark_results/benchmark_config.yaml
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

SKUS = [
    {"name": "SKU1_Can",    "image": "skus/sku1.jpg"},
    {"name": "SKU2_Bottle", "image": "skus/sku2.jpg"},
    {"name": "SKU3_Box",    "image": "skus/sku3.jpg"},
]

SEEDS = [42, 123, 456]
RESULTS_DIR = Path("docs/benchmark_results")
BENCHMARK_DIR = Path("checkpoints/benchmark")

CSV_FIELDNAMES = [
    "method", "seed", "step", "current_sku",
    "sku1_map50", "sku2_map50", "sku3_map50",
    "avg_seen_map50", "current_sku_map50",
]


def _load_base_config() -> dict:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["val_fraction"] = 0.2
    return cfg


def _generate_sku_data(sku: dict, shared_dir: str, config: dict) -> str:
    """Generate synthetic data for one SKU if not already generated. Returns coco_json path."""
    coco_json = os.path.join(shared_dir, "coco_annotations.json")
    if os.path.exists(coco_json):
        print(f"  [Generate] Reusing existing data: {shared_dir}")
        return coco_json

    from src.pipeline.stages.extract import stage_extract
    from src.pipeline.stages.generate import stage_generate

    features_path = os.path.join(shared_dir, "sku_features.json")
    stage_extract(
        image_path=sku["image"],
        config=config,
        checkpoint_dir=shared_dir,
        dry_run=False,
    )
    coco_json = stage_generate(
        features_path=features_path,
        config=config,
        checkpoint_dir=shared_dir,
        dry_run=False,
        product_image_path=sku["image"],
    )
    print(f"  [Generate] Done: {coco_json}")
    return coco_json


def run_experiment(method: str, ewc_lambda: int, seed: int, coco_jsons: dict) -> list:
    """
    Run one 3-SKU sequential onboarding experiment.

    Args:
        method:     "ewc" or "naive"
        ewc_lambda: EWC regularisation strength (0 = naive)
        seed:       Random seed for reproducibility
        coco_jsons: dict mapping SKU name to coco_json path (pre-generated, shared)

    Returns:
        List of result row dicts for CSV output.
    """
    import torch
    torch.manual_seed(seed)

    from src.pipeline.stages.train import stage_train
    from scripts._benchmark_eval import evaluate_on_split

    exp_dir = str(BENCHMARK_DIR / f"{method}_seed{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    config = _load_base_config()
    config["ewc_lambda"] = ewc_lambda
    config["train_epochs"] = 10   # Fewer epochs for benchmark speed; tune as needed

    rows = []
    current_weights = None

    for step, sku in enumerate(SKUS, start=1):
        sku_name = sku["name"]
        step_dir = os.path.join(exp_dir, f"step{step}_{sku_name}")
        job_id = f"{method}_s{seed}_step{step}"

        print(f"\n[{method}/seed={seed}] Step {step}: training on {sku_name}")
        current_weights = stage_train(
            coco_json=coco_jsons[sku_name],
            config=config,
            checkpoint_dir=step_dir,
            dry_run=False,
            job_id=job_id,
        )

        # Carry EWC state forward to next step
        ewc_src = os.path.join(step_dir, job_id, "ewc_state_snapshot.pt")
        ewc_canonical = os.path.join(exp_dir, "ewc_state.pt")
        if os.path.exists(ewc_src):
            shutil.copy(ewc_src, ewc_canonical)

        # Evaluate on val splits of all seen SKUs
        row: dict = {
            "method": method, "seed": seed, "step": step,
            "current_sku": sku_name,
            "sku1_map50": None, "sku2_map50": None, "sku3_map50": None,
        }

        for eval_idx, eval_sku in enumerate(SKUS[:step], start=1):
            print(f"  Evaluating on {eval_sku['name']} val split...")
            try:
                m = evaluate_on_split(
                    weights_path=current_weights,
                    coco_json=coco_jsons[eval_sku["name"]],
                    split="val",
                )
                row[f"sku{eval_idx}_map50"] = round(m["map50"], 4)
            except Exception as exc:
                print(f"  [WARN] Eval failed for {eval_sku['name']}: {exc}")
                row[f"sku{eval_idx}_map50"] = None

        seen = [row[f"sku{i+1}_map50"] for i in range(step) if row[f"sku{i+1}_map50"] is not None]
        row["avg_seen_map50"] = round(sum(seen) / len(seen), 4) if seen else 0.0
        row["current_sku_map50"] = row[f"sku{step}_map50"]

        print(f"  avg_seen={row['avg_seen_map50']:.4f}  current={row['current_sku_map50']}")
        rows.append(row)

    return rows


def plot_retention_curve(csv_path: Path, out_path: Path) -> None:
    import csv as _csv
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))

    methods = ["ewc", "naive"]
    colors = {"ewc": "#1f77b4", "naive": "#d62728"}
    steps = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(7, 4))

    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        by_step: dict = {s: [] for s in steps}
        for r in method_rows:
            s = int(r["step"])
            v = r["avg_seen_map50"]
            if v not in (None, "", "None"):
                by_step[s].append(float(v))

        means = [np.mean(by_step[s]) if by_step[s] else 0.0 for s in steps]
        stds  = [np.std(by_step[s])  if by_step[s] else 0.0 for s in steps]

        ax.errorbar(
            steps, means, yerr=stds,
            label=method.upper(), color=colors[method],
            marker="o", linewidth=2, capsize=4,
        )

    sku_labels = [s["name"] for s in SKUS]
    ax.set_xticks(steps)
    ax.set_xticklabels([f"After {n}" for n in sku_labels], rotation=15, ha="right")
    ax.set_ylabel("Avg seen-SKU mAP@50")
    ax.set_title("EWC vs Naive: Retention Across Sequential SKU Onboarding")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[Benchmark] Retention curve saved: {out_path}")


def main() -> None:
    for sku in SKUS:
        if not os.path.exists(sku["image"]):
            raise FileNotFoundError(
                f"SKU image not found: {sku['image']}. "
                "Source images per Task 13 before running the benchmark."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    config = _load_base_config()

    # Generate synthetic data once per SKU (shared between EWC and naive runs)
    print("\n=== Generating synthetic data for all 3 SKUs ===")
    coco_jsons = {}
    for sku in SKUS:
        shared_dir = str(BENCHMARK_DIR / f"generated_{sku['name']}")
        coco_jsons[sku["name"]] = _generate_sku_data(sku, shared_dir, config)

    csv_path = RESULTS_DIR / "ewc_vs_naive.csv"
    all_rows: list = []

    for seed in SEEDS:
        for method, lam in [("ewc", 5000), ("naive", 0)]:
            print(f"\n=== Experiment: {method.upper()} | seed={seed} ===")
            rows = run_experiment(method, lam, seed, coco_jsons)
            all_rows.extend(rows)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[Benchmark] Results saved: {csv_path}")

    plot_retention_curve(csv_path, RESULTS_DIR / "retention_curve.png")

    # Save benchmark config for reproducibility
    benchmark_cfg = {
        "skus": SKUS,
        "seeds": SEEDS,
        "ewc_lambda": 5000,
        "naive_lambda": 0,
        "val_fraction": config["val_fraction"],
        "train_epochs": 10,
    }
    with open(RESULTS_DIR / "benchmark_config.yaml", "w") as f:
        yaml.dump(benchmark_cfg, f, default_flow_style=False)
    print(f"[Benchmark] Config saved: {RESULTS_DIR / 'benchmark_config.yaml'}")


if __name__ == "__main__":
    main()
