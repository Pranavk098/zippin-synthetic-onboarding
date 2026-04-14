"""
Minimal setup.py so the src/ package is importable via `pip install -e .`
in both the Docker container and local development environments.
"""

from setuptools import setup, find_packages

setup(
    name="zippin-sku-onboarding",
    version="2.0.0",
    description="Zero-Shot SKU Onboarding Pipeline for Zippin Edge Environments",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "ultralytics>=8.2.0",
        "httpx>=0.27.0",
        "pyyaml>=6.0",
        "numpy>=1.26.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.7.0",
        "Pillow>=10.3.0",
    ],
    extras_require={
        "eval": ["pycocotools>=2.0.7"],
        "render": ["blenderproc>=2.6.0"],
    },
    entry_points={
        "console_scripts": [
            "sku-onboard=src.pipeline.orchestrator:main",
        ],
    },
)
