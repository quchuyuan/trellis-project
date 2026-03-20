# Use official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 关键：伪装成有显卡的环境进行编译，指定 RunPod 常用的架构 (8.0=A100, 8.6=3090/A6000, 8.9=4090, 9.0=H100)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git build-essential libgl1-mesa-dev libglib2.0-0 \
    libosmesa6-dev freeglut3-dev mesa-common-dev \
    libegl1-mesa-dev libgles2-mesa-dev wget curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. 先克隆仓库
RUN git clone https://github.com/microsoft/TRELLIS.2.git .

# 2. 安装 Python 基础依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 安装 Flash-Attention (使用预编译版本，避免 GitHub Actions 编译失败)
# 注意：这里选择匹配 torch 2.6 + cu124 的版本
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 4. 安装 nvdiffrast
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git

# 5. 编译 TRELLIS 专属扩展 (这些比较小，MAX_JOBS=1 可以稳过)
RUN pip install ./extensions/flexgemm
RUN pip install ./extensions/cumesh
RUN pip install ./extensions/o-voxel

# 6. 安装主包
RUN pip install -e .

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
