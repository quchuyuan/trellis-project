# Use official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Install system dependencies (OpenGL is crucial for nvdiffrast)
RUN apt-get update && apt-get install -y \
    git build-essential libgl1-mesa-dev libglib2.0-0 \
    libosmesa6-dev freeglut3-dev mesa-common-dev \
    libegl1-mesa-dev libgles2-mesa-dev wget curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone TRELLIS.2
RUN git clone https://github.com/microsoft/TRELLIS.2.git .

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We STOP building CUDA extensions here to avoid GitHub Actions failures.
# They will be built/installed on RunPod instead.

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
