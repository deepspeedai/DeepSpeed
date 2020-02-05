FROM nvidia/cuda:10.0-devel-ubuntu18.04

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    openssh-client openssh-server \
    pdsh curl sudo net-tools \
    vim iputils-ping wget

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
    pip install --upgrade pip && \
    # Print python an pip version
    python -V && pip -V

##############################################################################
# TensorFlow
##############################################################################
ENV TENSORFLOW_VERSION=1.14.0
RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION}

##############################################################################
# PyTorch
##############################################################################
ENV PYTORCH_VERSION=1.2.0
ENV TORCHVISION_VERSION=0.4.0
ENV TENSORBOARDX_VERSION=1.8
RUN pip install torch==${PYTORCH_VERSION}
RUN pip install torchvision==${TORCHVISION_VERSION}
RUN pip install tensorboardX==${TENSORBOARDX_VERSION}

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Mellanox OFED
##############################################################################
ENV MLNX_OFED_VERSION=4.6-1.0.1.1
RUN apt-get install -y libnuma-dev
RUN cd ${STAGE_DIR} && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64 && \
    ./mlnxofedinstall --user-space-only --without-fw-update --all -q && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64*

##############################################################################
# nv_peer_mem
##############################################################################
RUN mkdir -p ${STAGE_DIR} && \
    git clone https://github.com/Mellanox/nv_peer_memory.git ${STAGE_DIR}/nv_peer_memory && \
    cd ${STAGE_DIR}/nv_peer_memory && \
    ./build_module.sh && \
    cd ${STAGE_DIR} && \
    tar xzf ${STAGE_DIR}/nvidia-peer-memory_1.0.orig.tar.gz && \
    cd ${STAGE_DIR}/nvidia-peer-memory-1.0 && \
    apt-get install -y dkms && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ${STAGE_DIR}/nvidia-peer-memory_1.0-8_all.deb

##############################################################################
## Ucomment and set SSH Daemon port
###############################################################################
ENV SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
    sed "0,/^#Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
#RUN useradd --create-home --uid 8877 deepspeed
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
USER deepspeed

##############################################################################
# DeepSpeed
# TODO: once repo is public we can install latest deepspeed via this command
##############################################################################
#RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
#RUN cd ${STAGE_DIR}/DeepSpeed && \
#    git checkout . && \
#    git checkout master && \
#    sudo ./install.sh
#RUN rm -rf ${STAGE_DIR}/DeepSpeed
#RUN python -c "import deepspeed; print(deepspeed.__version__)"
