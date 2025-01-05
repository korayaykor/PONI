FROM ubuntu:20.04

# Avoid timezone prompt during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory and environment variable
WORKDIR /app
ENV PONI_ROOT=/app/PONI
ENV PYTHONPATH=/app/PONI

# Update and install system dependencies
RUN apt-get update && \
    apt-get -y install sudo && \
    apt-get -y install git && \
    apt-get install -y wget curl bzip2 ca-certificates && \
    apt-get install -y build-essential && \
    apt-get install -y cmake && \
    apt-get install -y gcc g++ && \
    apt-get install -y python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Fix ENV syntax
ENV PATH=/opt/conda/bin:$PATH

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

# Initialize conda in bash so it's available in interactive sessions
RUN conda init bash && \
    echo "conda activate poni" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]