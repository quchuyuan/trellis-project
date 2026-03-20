# Use official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    freeglut3-dev \
    mesa-common-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the TRELLIS.2 repository FIRST (into the empty /app directory)
RUN git clone https://github.com/microsoft/TRELLIS.2.git .

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install custom CUDA extensions (the core of TRELLIS.2)
# We set MAX_JOBS=2 to avoid memory exhaustion on GitHub Actions runners
RUN pip install --no-cache-dir ninja
ENV MAX_JOBS=2

# Use the official setup script for all extensions
# This script handles FlexGEMM, CuMesh, O-Voxel, nvdiffrast, and flash-attn
RUN ./setup.sh --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast

# Final installation of the trellis package itself
RUN pip install -e .

# Copy the RunPod handler
COPY handler.py /app/handler.py

# (Optional) Pre-download some common model components if possible to reduce cold starts
# Requires HF_TOKEN to be set during build if models are gated, 
# but usually better to do this at runtime with a fast network.

# Set the entrypoint
CMD ["python", "-u", "/app/handler.py"]
