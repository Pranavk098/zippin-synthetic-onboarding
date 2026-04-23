# Showcase Conversion Design — Zippin Synthetic SKU Onboarding PoC

**Date:** 2026-04-22
**Approach:** Parallel Track B — Hiring Panel + ML Engineer
**Status:** Approved, ready for implementation

---

## Problem Statement

The repository contains a genuinely strong PoC with real EWC implementation, a working 4-stage pipeline, and a failure gallery. The problem is product truthfulness:

- The README claims outcomes ("production-grade", "under 10 minutes") that are not yet proven with artifacts.
- Evaluation semantics mix real mAP, proxy confidence metrics, and skipped states without labeling.
- Run output is artifact-scattered — a reviewer cannot find the run report, renders, weights, and failure gallery without reading the code.
- The central EWC claim ("continual learning prevents catastrophic forgetting") is code only — no benchmark result exists.

The goal of this design is to fix those gaps with minimal changes to the existing pipeline code.

---

## Approach: Two Parallel Tracks

### Track 1 — Hiring Panel Track

**Audience:** A non-specialist reviewer who has 5 minutes to understand the project.
**Deliverable:** One command → one run folder → reviewer can understand the project without reading code.
**Blocking dependency:** None. Starts with the existing `product.jpg`.

### Track 2 — ML Engineer Track

**Audience:** A senior ML engineer who will scrutinize the EWC claim.
**Deliverable:** A retention matrix and curve showing EWC vs naive fine-tuning across 3 sequential SKUs.
**Blocking dependency:** 2 additional product images must be sourced before Track 2 can run.

Both tracks converge into a single updated README.

---

## Track 1 Design

### 1. Default Config — Procedural Mesh

Change `config.yaml`:
```yaml
mesh_source: "procedural"   # was "reconstructed"
```

Create `config_reconstructed.yaml` as a separate file with:
```yaml
mesh_source: "reconstructed"
mesh_backend: "triposr_local"   # NOT hf_space — rate-limited and unreliable for demo
mesh_resolution: 256            # safe for RTX 5070 8GB VRAM (~5GB used)
```

Add a note in `config_reconstructed.yaml` header:
```
# Requires: pip install tsr rembg
# Environment: GPU with 6+ GB VRAM. Tested on RTX 5070 (8GB) at resolution=256.
# NOT the default because it requires additional setup. Use procedural mode for
# first-run demos and CI.
```

**Rationale:** Procedural mode has no external dependencies beyond BlenderProc2 and always
produces a valid COCO dataset. Reconstructed mode is a legitimate enhancement but must not
be presented as "easy" when it requires TripoSR installed and 6GB VRAM free.

### 2. `eval_mode` Label in Metrics and Report

`stage_eval` must add `"eval_mode"` to its return dict:

| Condition | `eval_mode` value | Human note |
|---|---|---|
| `coco_gt_path` provided and file exists | `"real_map"` | "Ground-truth mAP computed with pycocotools." |
| No GT path — proxy confidence used | `"proxy_only"` | "No GT annotations — confidence-based proxy. Pass --gt_coco for real mAP." |
| `real_images_dir` missing or empty | `"skipped"` | "No real images found — eval skipped. Pass --real_dir to enable." |
| `dry_run=True` | `"dry_run"` | "Dry-run — mock metrics only." |

`reporting.py` must surface `eval_mode` prominently in the markdown:

```markdown
## Evaluation

**Eval Mode: proxy_only** — No GT annotations. Confidence-based proxy metrics shown.
Pass `--gt_coco <path>` for ground-truth mAP.

- mAP@50: 0.61  (proxy)
- ...
```

The label must be bold and on its own line. Reviewers must not need to infer what the numbers mean.

### 3. Render Thumbnails in Run Report

`reporting.py` is extended to:

1. Glob for images in `<checkpoint_dir>/<job_id or 'synthetic_dataset'>/images/train/`
2. Sort by filename (deterministic — not by mtime)
3. Copy the first 6 into `<run_dir>/renders/render_001.jpg` ... `render_006.jpg`
4. Reference them as markdown image links in `run_report.md`:

```markdown
## Synthetic Renders (sample)

![render_001](renders/render_001.jpg) ![render_002](renders/render_002.jpg) ...
```

If fewer than 6 renders exist, copy all. If the renders dir doesn't exist (dry-run), skip silently with a log warning — do not crash.

### 4. Standardized Artifact Layout

Every run must produce exactly this folder structure. The spec is the contract:

```
checkpoints/<run_id>/
├── run_report.md                  ← human-readable summary
├── run_report.json                ← machine-readable (API consumers)
├── renders/                       ← 6 deterministically selected synthetic renders
│   ├── render_001.jpg
│   ├── render_002.jpg
│   └── ... (up to 6)
├── coco_annotations.json          ← Stage 2 COCO output (symlink or copy)
├── <run_id>_weights.pt            ← fine-tuned YOLO weights
├── ewc_state_snapshot.pt          ← SNAPSHOT of ewc_state.pt after this run (see below)
├── <run_id>_metrics.json          ← eval metrics dict including eval_mode
└── failure_gallery/               ← always nested under run_id, never at checkpoint root
    ├── <rank>_conf<score>_<name>.jpg
    └── gallery_summary.json
```

**`ewc_state.pt` vs `ewc_state_snapshot.pt`:**
- `checkpoints/ewc_state.pt` is the shared canonical state used by the next onboarding run. It is updated in-place after each SKU training.
- After updating the canonical state, `stage_train` copies it to `checkpoints/<run_id>/ewc_state_snapshot.pt`. This makes the run folder self-describing and reproducible: you can restore exactly what EWC state existed after this run.
- The `run_report.json` records `"ewc_snapshot_path"` pointing to the snapshot.

**`failure_gallery/` path fix:**
Currently `stage_eval` passes `checkpoint_dir` to `_build_failure_gallery`, which creates `<checkpoint_dir>/failure_gallery/<job_id>/`. This must be changed to write to `<checkpoint_dir>/<job_id>/failure_gallery/` so the gallery is always inside the run folder. The `stage_eval` call must pass `run_dir = os.path.join(checkpoint_dir, job_id or "local")` as the base.

### 5. Smoke Tests

**`tests/test_pipeline_smoke.py`:**

```python
def test_dry_run_produces_run_report():
    # run full pipeline in dry_run=True, assert run_report.md exists

def test_no_gt_eval_returns_proxy_mode():
    # call stage_eval with no coco_gt_path, assert metrics["eval_mode"] == "proxy_only"

def test_missing_real_dir_returns_skipped():
    # call stage_eval with nonexistent real_dir, assert metrics["eval_mode"] == "skipped"

def test_run_report_contains_eval_mode_label():
    # dry-run pipeline, read run_report.md, assert "Eval Mode:" appears in content
```

**`tests/test_api_lifecycle.py`:**

```python
def test_api_onboard_lifecycle(monkeypatch):
    # monkeypatch run_pipeline to return a mock result dict immediately
    # POST /onboard → assert 202 + job_id
    # GET /status/<job_id> → assert status in {pending, running, done}
    # GET /metrics/<job_id> → assert response matches OnboardingResult schema
    # Does NOT invoke Blender, YOLO, or Ollama
```

The API test mocks `run_pipeline` at the import level. It validates the lifecycle schema and HTTP contract, not the pipeline itself.

### 6. Golden Demo Commands (for README)

**Start here (no GPU required):**
```bash
python -m src.pipeline.orchestrator \
  --stage all --image product.jpg \
  --sku_name "RedBull250ml" --dry-run
```
Expected output: `checkpoints/local/run_report.md`

**GPU-backed run (BlenderProc2 + YOLO training):**
```bash
python -m src.pipeline.orchestrator \
  --stage all --image product.jpg \
  --sku_name "RedBull250ml"
```
Expected output: `checkpoints/local/` with full artifact layout above.

---

## Track 2 Design

### 1. Source Product Images

Acquire 3 visually distinct retail product reference images:
- **SKU 1:** A can (cylindrical, metallic, label-forward)
- **SKU 2:** A bottle (glass or plastic, taller aspect ratio)
- **SKU 3:** A box/carton (rectangular, flat faces)

Sources: Open Images Dataset v7 (filter: "Tin can", "Bottle", "Box") or real product photos.

**Preprocessing requirements (apply to all 3):**
- Crop to product only — no busy backgrounds
- Resize longest edge to 640px
- Save as `skus/sku1.jpg`, `skus/sku2.jpg`, `skus/sku3.jpg`

Visual diversity requirement: the 3 SKUs must differ in shape category, not just packaging art. A can, bottle, and box achieve this. Three different soda cans do not.

### 2. Deterministic Train/Val Split at Generation Time

`stage_generate` (via `bproc_generator.py`) currently writes all renders to `images/train/`. Extend it to:

1. Accept a `val_fraction: float = 0.2` parameter (passed from config)
2. After rendering, deterministically assign renders to train vs val using `sorted(glob(...))` — no shuffle — then take last `int(n * val_fraction)` as val
3. Write train images to `images/train/` and val images to `images/val/`
4. The COCO annotations JSON already contains all images — add a `"split"` field to each image entry: `"train"` or `"val"`

This split is deterministic (sorted filenames, fixed fraction) so both EWC and naive runs use the same val set per SKU.

### 3. Benchmark Script — `scripts/benchmark_ewc.py`

Runs two 3-SKU sequential onboarding experiments:

**EWC run:** `ewc_lambda: 5000`
**Naive run:** `ewc_lambda: 0` (passed via config override)

For each experiment, for each SKU step N (1→2→3):
1. Run `stage_train` on SKU N's synthetic training data
2. After training, evaluate the current weights on the held-out val sets of ALL seen SKUs (SKU 1…N)
3. Record per-SKU mAP@50 using the benchmark evaluator (not `stage_eval`)

**Retention matrix columns:**
- `sku1_map50`, `sku2_map50`, `sku3_map50` — per-SKU retention
- `current_sku_map50` — performance on the just-trained SKU (forward transfer / acquisition)
- `avg_seen_map50` — mean across all SKUs onboarded so far

The `avg_seen_map50` column prevents EWC from "winning" by freezing too much and learning new SKUs poorly.

**Seeds:** Run with 3 random seeds. Record seed in CSV. Report mean ± std in README table.

**Output:**
```
docs/benchmark_results/
├── ewc_vs_naive.csv           ← full matrix (run, seed, step, sku1_map50, ..., avg_seen_map50)
├── retention_curve.png        ← matplotlib: avg_seen_map50 vs onboarding step, EWC vs naive
└── benchmark_config.yaml      ← exact config used (seeds, lambda, image paths, val_fraction)
```

### 4. Benchmark-Specific Evaluator

Add `scripts/_benchmark_eval.py` (private to the benchmark script, not part of the main pipeline):

```python
def evaluate_on_split(weights_path, coco_json, split="val") -> dict:
    """
    Load weights, run YOLO inference on the val split images from coco_json,
    compute mAP@50 with pycocotools using the GT from coco_json.
    Returns {"map50": float, "map50_95": float, "eval_mode": "real_map"}.
    """
```

This function reads the `"split"` field from the COCO JSON, filters to val images only, runs inference, and computes real mAP. It does not touch `stage_eval` or any pipeline stage — the benchmark has its own evaluation path.

**Eval mode for benchmark:** Always `"real_map"`. The benchmark note in README:
> "Retention measured on held-out synthetic validation renders with ground-truth labels generated by BlenderProc2."

### 5. Precise Benchmark Claim

The README headline claim is:

> **"Under sequential synthetic SKU onboarding, EWC preserves prior-SKU validation mAP better than naive fine-tuning (mean ± std across 3 seeds)."**

A separate honest note:
> "Real shelf-image generalization (Sim2Real transfer) is a separate validation step requiring labeled shelf photos. See Roadmap."

---

## Convergence — README Structure

After both tracks complete, the README is rewritten around this structure:

### 1. What This Demonstrates
Three outcome-based bullets:
- Single-image SKU onboarding pipeline (VLM extraction → synthetic data → YOLO fine-tune → evaluation)
- Sequential onboarding benchmark with EWC retention across 3 SKUs
- Reproducible per-run artifacts: report, renders, weights, failure gallery

### 2. Quickstart
Dry-run command first. GPU-backed run second. Expected artifact layout shown.

### 3. Continual Learning Results
Retention table (mean ± std, 3 seeds). Retention curve image. Precise claim. Honest eval note.

### 4. Architecture
Existing mermaid diagram, trimmed to remove unvalidated components (TensorRT export, DIS score, edge profiler). Those move to Roadmap.

### 5. Artifact Layout
The standardized folder structure from Track 1 Section 4, verbatim.

### 6. Setup
```
pip install -r requirements.txt
# Procedural mode: no additional deps
# Reconstructed mode (optional): pip install tsr rembg  [requires 6GB+ VRAM]
```

### 7. Roadmap
Honest list of what is next:
- Labeled real shelf images for Sim2Real mAP validation
- TripoSR reconstructed mesh quality benchmark
- TensorRT FP16/INT8 export and edge profiling on Jetson Orin NX
- Occlusion stress-test suite

**Removed from README:** "production-grade", "under 10 minutes", "full stack", "showcase demo output", and any claim not backed by a result artifact in this repository.

---

## What Is NOT In Scope

The following are explicitly out of scope for this design:

- Real shelf image collection or labeling
- TensorRT export or Jetson profiling
- Mesh critic retry logic improvements
- Analytics RAG or sensor fusion proposals
- Any restructuring of the existing pipeline module layout

These remain in the codebase and are acknowledged in the Roadmap. They are not touched by this implementation.

---

## Success Criteria

**Track 1 complete when:**
- [ ] `python -m src.pipeline.orchestrator --stage all --image product.jpg --sku_name X --dry-run` produces `checkpoints/local/run_report.md` with `Eval Mode:` label visible
- [ ] All 4 smoke tests pass
- [ ] API lifecycle test passes with monkeypatched pipeline
- [ ] Run folder contains renders/, failure_gallery/ nested under run_id/, ewc_state_snapshot.pt

**Track 2 complete when:**
- [ ] `docs/benchmark_results/ewc_vs_naive.csv` exists with 3 seeds × 3 steps × 2 methods
- [ ] `docs/benchmark_results/retention_curve.png` exists
- [ ] README "Continual Learning Results" section shows the retention table with mean ± std

**Both tracks complete when:**
- [ ] README contains no unproven claims
- [ ] A reviewer can run one command and find all artifacts in the documented layout
- [ ] A senior ML engineer can read the benchmark section and reproduce the result
