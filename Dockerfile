# docker build -t streamlit --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update
# RUN apt-get -y upgrade
RUN apt -y update

RUN apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    git \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    sudo \
    libgl1-mesa-glx \
    python3-dev \
    python3-pip

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# ENV TZ=Asia/Singapore
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN python3 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# install mish_cuda
RUN cd / && git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python3 setup.py build install

# minimal Dockerfile which expects to receive build-time arguments, and creates a new user called “user” (put at end of Dockerfile)
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
