FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/app \
    DINOV3_REPO_PATH=/opt/repos/dinov3 \
    DINOV3_WEIGHTS_PATH=/opt/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    MAX_JOBS=3 \
    NVCC_THREADS=2 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    ca-certificates \
    curl \
    wget \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    freeglut3-dev \
    mesa-common-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the official repository exactly once and build against its current layout.
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git .
RUN git clone --depth 1 https://github.com/facebookresearch/dinov3.git /opt/repos/dinov3

RUN mkdir -p /opt/models/dinov3

# Keep local RunPod-only requirements separate from the official repo tree.
COPY requirements.txt /tmp/requirements-runpod.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install \
        imageio \
        imageio-ffmpeg \
        tqdm \
        easydict \
        opencv-python-headless \
        ninja \
        trimesh \
        transformers \
        tensorboard \
        pandas \
        lpips \
        zstandard \
        kornia \
        timm && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 && \
    pip install flash-attn==2.7.3 --no-build-isolation && \
    mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation && \
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh && \
    pip install /tmp/extensions/CuMesh --no-build-isolation && \
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation && \
    cp -r /app/o-voxel /tmp/extensions/o-voxel && \
    pip install /tmp/extensions/o-voxel --no-build-isolation && \
    pip install -r /tmp/requirements-runpod.txt && \
    rm -rf /tmp/extensions

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
