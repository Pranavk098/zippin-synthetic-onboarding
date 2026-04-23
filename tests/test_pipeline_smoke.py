"""
Smoke tests for the pipeline. All run in dry_run mode — no GPU, Blender, or Ollama needed.
"""
import pytest
from pathlib import Path


def test_dry_run_produces_run_report(tmp_path):
    from src.pipeline.orchestrator import run_pipeline

    result = run_pipeline(
        sku_name="SmokeSKU",
        image_path="product.jpg",
        stages=("extract", "generate", "train", "eval"),
        checkpoint_dir=str(tmp_path),
        dry_run=True,
        job_id="smoke",
    )

    report = tmp_path / "smoke" / "run_report.md"
    assert report.exists(), f"run_report.md not found at {report}"
    assert result["report_path"] is not None


def test_no_gt_eval_returns_proxy_mode(tmp_path):
    from src.pipeline.stages.eval import stage_eval

    weights = tmp_path / "weights.pt"
    weights.touch()

    # dry_run=True gives us eval_mode without needing YOLO
    metrics = stage_eval(
        real_images_dir=str(tmp_path / "real"),
        weights_path=str(weights),
        config={},
        checkpoint_dir=str(tmp_path),
        coco_gt_path=None,
        dry_run=True,
        job_id="smoke_proxy",
    )

    assert metrics["eval_mode"] == "dry_run"


def test_missing_real_dir_returns_skipped(tmp_path):
    from src.pipeline.stages.eval import stage_eval

    weights = tmp_path / "weights.pt"
    weights.touch()

    metrics = stage_eval(
        real_images_dir=str(tmp_path / "nonexistent_real"),
        weights_path=str(weights),
        config={},
        checkpoint_dir=str(tmp_path),
        dry_run=False,
        job_id="smoke_skip",
    )

    assert metrics["eval_mode"] == "skipped"


def test_run_report_contains_eval_mode_label(tmp_path):
    from src.pipeline.orchestrator import run_pipeline

    run_pipeline(
        sku_name="SmokeSKU",
        image_path="product.jpg",
        stages=("extract", "generate", "train", "eval"),
        checkpoint_dir=str(tmp_path),
        dry_run=True,
        job_id="smoke_label",
    )

    report_text = (tmp_path / "smoke_label" / "run_report.md").read_text()
    assert "Eval Mode:" in report_text, (
        "run_report.md does not contain 'Eval Mode:' label. "
        "reporting.py must surface eval_mode prominently."
    )
