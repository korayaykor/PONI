# Base image with NVIDIA CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

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

# Now your cmake is >= 3.12

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Fix ENV syntax
ENV PATH=/opt/conda/bin:$PATH

# Install NVIDIA repository and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nvidia-utils-470 \
    nvidia-driver-470 \
    cuda-toolkit-11-4

# Create conda environment
RUN conda create -n poni python=3.9 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "poni", "/bin/bash", "-c"]

# Install PyTorch and dependencies
RUN conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch -y

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

# Setup Habitat-sim
RUN cd $PONI_ROOT/dependencies && \
    git clone https://github.com/facebookresearch/habitat-sim.git && \
    cd habitat-sim && \
    git checkout tags/challenge-2022 && \
    pip install -r requirements.txt && \
    HEADLESS=True python setup.py install

# Setup Habitat-lab
RUN cd $PONI_ROOT/dependencies && \
    git clone https://github.com/facebookresearch/habitat-lab.git && \
    cd habitat-lab && \
    git checkout tags/challenge-2022 && \
    pip install -e .

# Install ML dependencies
RUN python -m pip install detectron2==0.5 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html && \
    python -m pip install torch-scatter==2.0.7 \
    -f https://pytorch-geometric.com/whl/torch-1.9.1+cu102.html

# Setup A* implementation
RUN cd $PONI_ROOT/dependencies && \
    git clone https://github.com/srama2512/astar_pycpp.git && \
    cd astar_pycpp && \
    make

# Install PONI requirements
RUN cd $PONI_ROOT && \
    pip install -r requirements.txt

# Setup directory structure and download script
RUN mkdir -p $PONI_ROOT/data/scene_datasets/mp3d && \
    mkdir -p $PONI_ROOT/data/scene_datasets/mp3d_uncompressed

# Initialize conda in bash so it's available in interactive sessions
RUN conda init bash && \
    echo "conda activate poni" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
