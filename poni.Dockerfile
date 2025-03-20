# Base image with NVIDIA CUDA support - Updated for RTX 4500 ADA
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PONI_ROOT=/app/PONI
ENV PYTHONPATH=/app/PONI
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set working directory
WORKDIR /app

# Update and install system dependencies
RUN apt-get update && \
    apt-get -y install sudo \
    git \
    wget \
    cmake \
    ca-certificates \
    gnupg \
    software-properties-common 

# Add NVIDIA GPG keys and repositories for Ubuntu 20.04
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# Remove old cmake
RUN apt-get remove --purge -y cmake

# Install a newer CMake from Kitware
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Fix ENV syntax
ENV PATH=/opt/conda/bin:$PATH

# Install NVIDIA repository and tools - Updated for newer drivers compatible with RTX 4500 ADA
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nvidia-utils-535 \
    nvidia-driver-535

# Create conda environment
RUN conda create -n poni python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "poni", "/bin/bash", "-c"]

# Pin NumPy to 1.x version to avoid compatibility issues with PyTorch
RUN conda install numpy=1.25.2 -y

# Install PyTorch and dependencies - Updated for CUDA 12.1 compatibility
RUN conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Make sure numpy stays at 1.x after PyTorch installation
RUN pip install "numpy<2.0.0" --force-reinstall

# Clone PONI repository and initialize submodules
RUN git clone https://github.com/korayaykor/PONI.git && \
    cd PONI && \
    git submodule init

# Install system dependencies for habitat-sim
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglfw3-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libopengl0 \
    pkg-config \
    python3-dev \
    wget \
    ninja-build \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Setup Habitat-sim with updated CUDA flags
RUN cd $PONI_ROOT/dependencies && \
    git clone https://github.com/facebookresearch/habitat-sim.git && \
    cd habitat-sim && \
    git checkout tags/challenge-2022 && \
    pip install -r requirements.txt && \
    HEADLESS=True CUDA_HOME=/usr/local/cuda python setup.py install

# Setup Habitat-lab
RUN cd $PONI_ROOT/dependencies && \
    git clone https://github.com/facebookresearch/habitat-lab.git && \
    cd habitat-lab && \
    git checkout tags/challenge-2022 && \
    pip install -e .

# Install ML dependencies - Updated for PyTorch 2.1 and CUDA 12.1
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Setup A* implementation
RUN cd $PONI_ROOT/dependencies && \
    git clone https://github.com/srama2512/astar_pycpp.git && \
    cd astar_pycpp && \
    make

# Install PONI requirements
RUN cd $PONI_ROOT && \
    pip install -r requirements.txt

# Setup directory structure
RUN mkdir -p $PONI_ROOT/data/scene_datasets/mp3d && \
    mkdir -p $PONI_ROOT/data/scene_datasets/mp3d_uncompressed

# Initialize conda in bash so it's available in interactive sessions
RUN conda init bash && \
    echo "conda activate poni" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
