# ============================================================
# Zippin Zero-Shot SKU Onboarding — GPU Worker Image
# Base: CUDA 12.1 + Python 3.10 (matches Jetson Orin CUDA ABI)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git curl wget ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (split for cache efficiency)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BlenderProc2 (requires separate pull — ~1.5GB Blender embedded binary)
RUN pip install --no-cache-dir blenderproc && \
    blenderproc pip install pycocotools

# Application code
COPY . .

# Make the package importable
RUN pip install --no-cache-dir -e .

# Entrypoints
EXPOSE 8080

# Default: start the REST API server
CMD ["uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--log-level", "info"]
