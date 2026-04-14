# Pranav Koduru — Engineering Intelligence Brief
### For: Motilal Agrawal, Chief Scientist & Abhinav Katiyar, SVP of Engineering, Zippin

---

This repository is my application to join Zippin.

Not a resume. Not a cover letter template. A working proof — built on my own time — that I understand your core engineering bottleneck deeply enough to have already started solving it.

I want to be part of the team building the infrastructure that makes frictionless retail real at scale. This is what I've invested before the first conversation. What comes after that conversation is the part I'm asking for the chance to show you.

---

## The Thesis Behind the Build

I spent a significant amount of time studying Zippin before writing a single line of code.  
What I found was three compounding constraints that, individually, are solvable.  
Together, they form the actual problem:

**Constraint 1: Velocity of new SKU introductions**  
Stadium concession menus rotate by game, by season, by visiting team. Convenience stores cycle through limited-edition packaging quarterly. Every new SKU requires annotated training data, a model update, and a deployment. At current scale, that process is weeks long. At the scale Zippin is targeting — tens of thousands of stores — it becomes physically impossible.

**Constraint 2: The physics of edge hardware**  
NVIDIA Jetson Orin NX has 8GB of unified memory shared between CPU and GPU. Large VLMs, full model fine-tunes, and high-throughput video inference cannot coexist in that budget simultaneously. Every architectural decision must be made *against* this ceiling, not around it.

**Constraint 3: Catastrophic forgetting**  
When you fine-tune a detection model on new SKU data, it overwrites the parameter space that encoded knowledge of previous SKUs. This is not a minor inconvenience — it is the reason you cannot simply push a model update each time a new product appears. Without a mathematically principled solution to continual learning, every new SKU degrades the whole product catalogue. At 10 SKUs this is tolerable. At 10,000 it is catastrophic.

My thesis: **these three constraints are not independent problems — they are one problem**.  
And the solution is a single, disciplined pipeline that addresses all three simultaneously.

> **The detail that separates this from a paper re-implementation:** Most public EWC code computes the Fisher matrix and stops — it measures importance without ever injecting the penalty into training. This codebase overrides `criterion()` in a custom `DetectionTrainer` subclass so the EWC penalty lands on every gradient update, not just around it. The math is real. The forgetting protection is real. Full breakdown in Decision 2 below.

---

## The Architecture Decisions — And Why I Made Them

### Decision 1: Ollama + LLaVA over vLLM

The original design document for this PoC specified vLLM for VLM inference. I changed it.

vLLM is an excellent inference server for cloud deployments — but it requires compiled CUDA extensions, a Linux-native Python environment, and a separate installation path from the rest of the stack. On Windows development machines (where Zippin engineers work), and on the path toward Jetson Orin cross-compilation, this introduces friction at every environment transition.

Ollama serves GGUF-quantized models behind a simple HTTP API. LLaVA-7B in 4-bit GGUF runs cleanly in 4.2GB of VRAM — well under the 8GB ceiling. The API is identical across Windows, Linux, and macOS. It pulls and runs with a single command. When your edge constraint is memory, you choose the tool that respects the budget by design, not by careful configuration.

This is not a shortcut. It is a hardware-aware decision.

### Decision 2: EWC over QLoRA for Continual Learning

The original plan called for QLoRA adapters — low-rank decomposition matrices trained on new SKU data and merged back into the base model. This is architecturally valid. But it has a hidden cost: **it does not prevent catastrophic forgetting by default**.

QLoRA reduces memory during *training* by freezing the base model and only training rank-decomposition matrices. But when you merge those adapters back into the base model for inference, the weight updates propagate globally. Without an additional mechanism to protect prior task representations, the training loss for the new SKU optimises freely over the full parameter space — and older SKUs are forgotten.

EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017) solves exactly this. It computes the diagonal of the Fisher Information Matrix over the previous task's data — a per-parameter measure of "how much did this weight matter for what we already learned?" — and uses it as a penalty in the new task's training loss. Weights with high Fisher values are penalised strongly for drifting. Weights with low Fisher values are free to adapt.

The result: you can fine-tune on SKU 500 and the model still correctly detects SKU 1.  
This is the property that makes a 10,000-SKU product catalogue tractable.

One critical note on implementation: I observed that many public EWC implementations compute the Fisher matrix but never actually inject the penalty into the training loop. They measure it before and after training, print the numbers, and call it done. This is the computational equivalent of installing a seatbelt and never buckling it. In this codebase, the EWC penalty is injected into *every gradient update* via a custom `DetectionTrainer` subclass (`src/continual_learning/ewc_trainer.py`) that overrides `criterion()`. The math is real. The protection is real.

### Decision 3: YOLOv8n over RT-DETR or Larger YOLO Variants

YOLOv8n has ~3.2 million parameters. RT-DETR-L has ~32 million. YOLOv8x has ~68 million.

On Jetson Orin NX with TensorRT FP16 optimisation:
- YOLOv8n: ~12ms per frame (83 FPS) — **comfortably above the 60 FPS SLA**
- YOLOv8m: ~28ms (35 FPS) — fails the SLA under multi-camera load
- RT-DETR-L: ~47ms (21 FPS) — fails at single camera

The accuracy delta between YOLOv8n and larger models on a single-class detection task with domain-appropriate synthetic training data is smaller than the difference in inference budget. More parameters is not always more accuracy. Especially when the training distribution (your synthetic data) is tightly controlled and the task is binary — this SKU vs. not this SKU.

The right model is the smallest model that meets the accuracy and latency SLA. That is YOLOv8n.

### Decision 4: FastAPI Async Job Queue over Synchronous CLI

A synchronous pipeline that blocks while Blender renders 50 frames is a demo. An async REST API with a job queue, polling endpoint, and per-SKU registry is a product.

The distinction matters beyond engineering aesthetics. Retail operators don't run CLI commands. They use dashboards and apps. The API layer is what transforms this PoC from "impressive code" into "something that Zippin can actually plug into the Retailer Dashboard to let store managers onboard new products without involving an ML engineer."

`POST /onboard` → returns `job_id` in 200ms.  
`GET /jobs/{id}` → polls pipeline progress.  
`GET /skus/{id}/metrics` → fetches mAP and confidence breakdown.  
`GET /health` → Kubernetes liveness probe.

The endpoints exist. They work. The job registry is thread-safe and persistent across restarts.

---

## The Three Problems I Can Solve for Zippin Beyond This PoC

### Problem 1: The Latency Cliff at Occlusion

When a shopper's arm enters the camera frame during a pick event, the visual confidence score for the target SKU drops — sometimes to zero. The current resolution path (wait for the arm to retract, then classify the shelf state) introduces a latency window that could cause incorrect cart attributions at high shopper density.

The solution is in `src/proposals/sensor_fusion.py`: a dynamic gated attention architecture that, under normal visual confidence, runs the cheap visual-only tracking path at full 60Hz. When confidence drops below threshold τ, the framework queries a timestamped async buffer of smart shelf weight events. If a weight delta matching the registered SKU mass is found within a ±500ms window, an Extended Kalman Filter resolves the transaction deterministically — with confidence proportional to mass match precision.

The key architectural insight: the expensive fusion computation runs *only* during active occlusion events. This is not 60Hz multimodal fusion — it is *event-triggered* multimodal fusion. Under typical store conditions, 85-90% of frames run the cheap visual path. The Jetson's thermal and power budget is preserved for the 10-15% of frames that actually need deterministic grounding.

I have direct production experience with the asyncio architecture this requires. At ATAI Labs, I migrated a sequential data ingestion pipeline to async multiprocessing with Python asyncio, increasing total throughput by 38.5% while maintaining strict reproducibility. The sensor fusion framework here applies the same pattern to a real-time, multi-rate signal alignment problem.

### Problem 2: Model Drift Across Dynamic Retail Environments

Seasonal packaging redesigns, new product formulations, different ambient lighting across Zippin's expanding venue portfolio — all of these cause silent model drift. A model that achieved 99.3% accuracy at NRG Stadium in September may degrade to 97.1% at the same venue in December when lighting conditions shift and packaging is updated for the holiday season.

At ATAI Labs, I implemented continuous model monitoring infrastructure that tracked data distribution shifts and proactively flagged models for retraining before accuracy degraded below SLA thresholds. The monitoring system maintained stable performance within a 3-5% variance band across key metrics by detecting covariate shift in the input distribution before it propagated to output quality.

This is directly applicable to Zippin's challenge. The signals to monitor are: incoming frame confidence distribution (should be stationary in a healthy store), per-SKU detection rate vs. historical baseline, and weight sensor agreement rate (measures how often vision and weight disagree — a leading indicator of drift).

### Problem 3: The Operational Intelligence Gap

Zippin captures extraordinary operational data — foot traffic density, per-SKU velocity, shelf weight depletion rates — but extracting *predictive* intelligence from it still requires manual SQL analysis. A stadium F&B director cannot ask "when will Lane 3's domestic beer stock out at current halftime foot traffic?" without either waiting for a weekly report or querying a database they probably don't have access to.

The solution is in `src/proposals/analytics_rag.py`: a multi-agent RAG architecture that routes natural language queries to the appropriate LLM tier based on complexity (haiku for lookups, sonnet for analysis, opus for predictive forecasting), retrieves relevant historical venue patterns from a Qdrant vector store using semantic similarity, and synthesizes a grounded, actionable response with concrete operational recommendations.

I built a full-stack RAG system at ATAI Labs that indexed large document corpora with sub-100ms vector retrieval using Qdrant, applied Maximum Marginal Relevance re-ranking, and drove real application actions through validated JSON outputs. The venue analytics layer here applies the same architecture to Zippin's operational data — transforming the platform from a pure transaction engine into a predictive venue management system.

---

## My Engineering Ideology

**I build for the hardware ceiling, not the theoretical maximum.**  
Every architecture decision I make is made against a constraint. Not "what is the most accurate model?" but "what is the most accurate model that fits in 8GB of VRAM, runs at 60 FPS, and can be updated overnight via OTA?" The constraint is not the enemy of good engineering — it is the definition of the problem.

**I believe the implementation gap between research and production is where real value is created.**  
EWC is a 2017 paper. The math is not novel. What is novel is implementing it correctly — with real Fisher matrix computation, per-batch penalty injection, online consolidation, serializable state, and a custom training loop that actually applies it. Most public implementations compute the Fisher and stop. I inject it into every gradient update. The difference between the two is the difference between a demo and a deployable system.

**I think in vertical slices, not horizontal layers.**  
When I build a feature, I build it from the hardware up to the API — one complete, tested, deployable slice. I don't write a perfect VLM wrapper while leaving the training loop as a stub. I don't build a beautiful FastAPI server that calls mock functions. Every component in this codebase does real work: the VLM extracts real attributes, the renderer generates real COCO annotations, the trainer applies a real EWC penalty, the API serves real async jobs.

**I document decisions, not just code.**  
The most important thing in a complex codebase is understanding *why* a choice was made. Not what the code does — the code shows that. Why QLoRA was considered and EWC was chosen. Why Ollama over vLLM. Why YOLOv8n and not a larger model. Why the EWC penalty must be injected per-batch and not computed at training boundaries. These decisions are documented in the code, not in a separate wiki that no one reads.

---

## The Numbers That Matter

| Metric | Achieved / Target |
|---|---|
| Pipeline stages | 4 (extract → generate → train → eval) |
| VRAM footprint (VLM inference) | ~4.2GB (under 8GB Orin ceiling) |
| YOLOv8n Jetson Orin NX inference | ~12ms / 83 FPS (target: ≥60 FPS) |
| EWC memory overhead | ~25MB (fixed, O(params), regardless of SKU count) |
| EWC retention target | ≥85% of prior SKU mAP (see `scripts/benchmark_ewc.py`) |
| Sim2Real mAP@50 target | ≥0.60 (validated against CVS reference images) |
| New SKU onboarding time | <10 minutes end-to-end |
| API response (POST /onboard) | <200ms (async — pipeline runs in background) |
| Prior latency optimisation (ATAI) | 8.0s → 1.5s (TensorRT, layer fusion, dynamic batching) |
| Synthetic data boost (ATAI) | +14% downstream accuracy (Blender 3D scene generation) |
| Async pipeline improvement (ATAI) | +38.5% throughput (asyncio multiprocessing migration) |

---

## What I've Already Invested — And What I'm Asking For

Before reaching out, I spent real time studying Zippin from the outside: engineering blog posts, patent filings, published architecture descriptions, conference talks from your Chief Scientist. I read everything publicly available about your edge compute constraints, your sensor fusion approach, and the specific challenges of deploying frictionless checkout across heterogeneous venue formats.

Then I built code against those constraints — not generic ML code, but code that respects the 8GB VRAM ceiling, the 60 FPS SLA, and the continual learning problem that gets harder with every SKU you add to the catalogue. The three future proposals in this repository (`sensor_fusion.py`, `analytics_rag.py`, the EWC benchmark) are not whiteboard ideas. They are running implementations of architectural solutions to problems I identified in your published technical writing.

That is what I'm willing to invest before I'm even on the team.

**I'm asking for one conversation.**

A 30-minute call with Motilal Agrawal, Abhinav Katiyar, or whoever on the engineering team is the right person to evaluate this work. I want to walk through the architecture decisions, answer the hard questions (why EWC over QLoRA, what breaks at 10,000 SKUs, how the EKF degrades under multi-shopper occlusion, where the Sim2Real gap actually lives), and let the quality of the technical judgment speak for itself.

Autonomous retail at stadium scale, with the accuracy requirements and edge constraints you're operating under, is one of the most technically interesting deployment problems in applied AI right now. I want to be in the room where those problems get solved.

---

**Pranav Koduru**  
M.S. Computer Science — George Mason University (2025)  
ML Engineer — ATAI Labs (2 years, edge deployment, computer vision, TensorRT)

*Repository entry point:*
```bash
python -m src.pipeline.orchestrator --stage all --image product.jpg --dry-run
```

---

*"Ship the thing. Document the decisions. Protect the prior SKUs."*
