# FROM noelmni/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04
# FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3
LABEL maintainer="Ravnoor Singh Gill <ravnoor@gmail.com>" \
        org.opencontainers.image.title="deepFCD" \
        org.opencontainers.image.description="Automated Detection of Focal Cortical Dysplasia using Deep Learning" \
        org.opencontainers.image.licenses="BSD-3-Clause" \
        org.opencontainers.image.source="https://github.com/NOEL-MNI/deepFCD" \
        org.opencontainers.image.url="https://github.com/NOEL-MNI/deepFCD"

# manually update outdated repository key
# fixes invalid GPG error: https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
    bash \
    wget \
    bzip2 \
    sudo \
    build-essential 
    # libgpuarray3 \
    # cuda
    # ubuntu-drivers-common
ENV TZ=Europe/Moscow \
    DEBIAN_FRONTEND=noninteractive    
# RUN sudo apt-get install -y nvidia-cuda-toolkit 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# RUN sudo apt-get -y install cudnn9-cuda-12
RUN sudo apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# RUN sudo apt update
# RUN sudo apt install -y nvidia-driver-535
# RUN sudo apt install gcc
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# RUN sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.129.03-1_amd64.deb
# RUN sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.129.03-1_amd64.deb
# RUN sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
# RUN sudo apt-get update
# RUN sudo apt-get -y install cuda
# RUN sudo apt install -y libgpuarray-dev

ENV PATH=/home/user/conda/bin:${PATH}

# create a working directory
RUN mkdir /app
WORKDIR /app

# create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# all users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b -p /home/user/conda \
    && rm Miniconda3-py38_23.5.2-0-Linux-x86_64.sh

# RUN conda update -n base -c defaults conda

RUN git clone --depth 1 https://github.com/NOEL-MNI/deepMask.git \
    && rm -rf deepMask/.git

RUN eval "$(conda shell.bash hook)" \
    && conda create -n preprocess python=3.8 \
    && conda activate preprocess \
    && python -m pip install -r deepMask/app/requirements.txt \
    && conda deactivate

COPY app/requirements.txt /app/requirements.txt


ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib
RUN sudo mv /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
# RUN export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# RUN export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         # ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN python -m pip install -r /app/requirements.txt \
    && conda install -c conda-forge pygpu==0.7.6 \
    && pip cache purge
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda install libcublas libcublas-dev libcublas-static
# RUN conda install -c conda-forge cupy
RUN python -m pip install h5py==2.10.0
RUN python -m pip install numpy==1.19.5

COPY app/ /app/

COPY tests/ /tests/

RUN sudo chmod -R 777 /app && sudo chmod +x /app/inference.py
RUN sudo chmod -R 777 /app && sudo chmod +x /app/train.py

ENV INPUT=/input
ENV OUTPUT=/output

ENTRYPOINT ["python3", "/app/train.py"]
