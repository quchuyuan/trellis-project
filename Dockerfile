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
# Set critical environment variables for headless cloud compilation
ENV MAX_JOBS=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TCNN_CUDA_ARCHITECTURES="80;86;89;90"

# Install ninja and high-performance extensions individually
RUN pip install --no-cache-dir ninja

# 1. Install nvdiffrast
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git

# 2. Install flash-attn (This is the heaviest part, we use a pre-built wheel if possible or limit build)
# Note: Compiling from source can take 30+ mins. 
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 3. Install TRELLIS internal extensions (from the cloned repo)
# We manually run the pip installs that setup.sh would perform
RUN pip install ./extensions/flexgemm
RUN pip install ./extensions/cumesh
RUN pip install ./extensions/o-voxel

# Final installation of the trellis package itself
RUN pip install -e .

# Copy the RunPod handler
COPY handler.py /app/handler.py

# (Optional) Pre-download some common model components if possible to reduce cold starts
# Requires HF_TOKEN to be set during build if models are gated, 
# but usually better to do this at runtime with a fast network.

# Set the entrypoint
CMD ["python", "-u", "/app/handler.py"]
