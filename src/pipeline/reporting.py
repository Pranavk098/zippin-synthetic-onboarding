"""
Run reporting helpers.

Produces a compact Markdown + JSON artifact after each pipeline run so the
project yields an immediately shareable output instead of only scattered logs.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _copy_render_thumbnails(run_dir: Path, coco_json: Optional[str], n: int = 6) -> list:
    """Copy first n renders (sorted by filename) from the synthetic dataset into run_dir/renders/.
    Returns list of relative paths like ['renders/render_001.jpg', ...].
    If no renders exist, returns [] and logs a warning.
    """
    if not coco_json:
        return []
    renders_src = Path(coco_json).parent / "images" / "train"
    if not renders_src.exists():
        logger.warning(f"[Report] Renders dir not found: {renders_src} — skipping thumbnails.")
        return []

    all_images = sorted(
        list(renders_src.glob("*.jpg")) + list(renders_src.glob("*.png")),
        key=lambda p: p.name,
    )
    if not all_images:
        logger.warning(f"[Report] No render images found in {renders_src}.")
        return []

    renders_dst = run_dir / "renders"
    renders_dst.mkdir(exist_ok=True)

    copied = []
    for i, src in enumerate(all_images[:n], start=1):
        dst = renders_dst / f"render_{i:03d}{src.suffix}"
        shutil.copy2(src, dst)
        copied.append(f"renders/render_{i:03d}{src.suffix}")

    return copied


def write_run_report(
    result: Dict,
    sku_name: str,
    image_path: str,
    checkpoint_dir: str,
    job_id: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    run_id = job_id or "local"
    report_dir = Path(checkpoint_dir) / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    thumbnails = _copy_render_thumbnails(report_dir, result.get("coco_json"))

    payload = {
        "run_id": run_id,
        "sku_name": sku_name,
        "image_path": image_path,
        "dry_run": dry_run,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages_run": result.get("stages_run", []),
        "artifacts": {
            "features": result.get("sku_attributes"),
            "coco_json": result.get("coco_json"),
            "weights_path": result.get("weights_path"),
            "ewc_path": result.get("ewc_path"),
            "ewc_snapshot_path": str(report_dir / "ewc_state_snapshot.pt")
                                 if (report_dir / "ewc_state_snapshot.pt").exists() else None,
            "metrics": result.get("metrics"),
        },
    }

    json_path = report_dir / "run_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metrics = result.get("metrics") or {}
    md_lines = [
        f"# Run Report: {sku_name}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Dry run: `{dry_run}`",
        f"- Source image: `{image_path or 'N/A'}`",
        f"- Stages run: `{', '.join(payload['stages_run']) or 'none'}`",
        "",
        "## Output Summary",
        "",
        f"- COCO annotations: `{result.get('coco_json') or 'not produced'}`",
        f"- Weights: `{result.get('weights_path') or 'not produced'}`",
        f"- EWC state: `{result.get('ewc_path') or 'not produced'}`",
        f"- Metrics file data: `{json_path.name}`",
        "",
        "## Evaluation",
        "",
        f"**Eval Mode: {metrics.get('eval_mode', 'unknown')}** — {metrics.get('eval_mode_note', '')}",
        "",
        f"- mAP@50: `{metrics.get('map50')}`",
        f"- mAP@50:95: `{metrics.get('map50_95')}`",
        f"- Images evaluated: `{metrics.get('n_images')}`",
        f"- Detections: `{metrics.get('n_detections')}`",
        f"- Mean confidence: `{metrics.get('mean_confidence')}`",
        f"- Failure gallery: `{metrics.get('failure_gallery_dir') or 'not produced'}`",
    ]

    if thumbnails:
        md_lines += [
            "",
            "## Synthetic Renders (sample)",
            "",
        ]
        for rel_path in thumbnails:
            md_lines.append(f"![{rel_path}]({rel_path})")

    md_path = report_dir / "run_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return str(md_path)
