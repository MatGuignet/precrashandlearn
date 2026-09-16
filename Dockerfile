# Crash&Learn Grand Prix — Docker image
#
# Stage: base — CPU-only, headless.  Used by: sim, train, tensorboard, viz.
#
# NOTE: f1tenth's setup.py is NOT used — it pins gym==0.19.0 and numpy<=1.22.
# Only the engine modules (f110_gym/envs/) are imported; deps are installed here.

FROM python:3.11-slim AS base

WORKDIR /app

# Build tools for numba + Pillow native extensions; tkinter for optional TkAgg
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps — loose pins, confirmed working on numpy 2.x + numba 0.65
RUN pip install --no-cache-dir \
        "numpy>=2.0" \
        "numba>=0.65" \
        "scipy>=1.7" \
        "Pillow>=9" \
        "pyyaml>=5.3" \
        "imageio>=2.28" \
        "imageio-ffmpeg" \
        "matplotlib>=3.7" \
        "stable-baselines3" \
        "tensorboard" \
        "pytest"

# Be careful to the version of torch and onxx !!!
RUN pip install --no-cache-dir \
        "torch>=2.0" \
        "onnx>=1.16,<2.0" \
        "onnxscript>=0.1" \
        "onnxruntime>=1.19,<2.0"

# Install whatever dependencies you need for you RL here
RUN pip install --no-cache-dir \
        "gymnasium>=0.29" \
        "tyro"
        

COPY . /app

# Engine modules are imported by path; PYTHONPATH points at the gym/ subdirectory
ENV PYTHONPATH=/app/gym
# Default headless matplotlib backend — can be overridden at runtime for TkAgg
ENV MPLBACKEND=Agg

