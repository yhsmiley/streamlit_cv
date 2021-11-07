# docker build -t streamlit .

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
    wget \
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

RUN pip3 install --no-cache-dir \
    gdown

# install mish_cuda (for ScaledYOLOv4)
RUN cd / && git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python3 setup.py build install

# install ScaledYOLOv4
# RUN cd / && \
#     git clone https://github.com/yhsmiley/ScaledYOLOv4 && \
#     cd ScaledYOLOv4 && \
#     git checkout d9420e432a5aaca9ff50a9c5857aa9f251828126 && \
#     cd scaledyolov4/weights && \
#     bash get_weights.sh && \
#     cd ../.. && \
#     pip3 install --no-cache-dir -e .

# optional, for DeepSORT CLIP embedder
RUN pip3 install --no-cache-dir git+https://github.com/openai/CLIP.git

# install DeepSORT
# RUN cd / && \
#     git clone https://github.com/levan92/deep_sort_realtime && \
#     cd deep_sort_realtime && \
#     git checkout fb4cf8e32cca33a2dab127d4d6e265adfb190e88 && \
#     cd deep_sort_realtime/embedder/weights && \
#     bash download_clip_wts.sh && \
#     cd ../../.. && \
#     pip3 install --no-cache-dir -e .

# optional, for ByteTrack
RUN pip3 install --no-cache-dir cython==0.29.24 lap==0.4.0

# install ByteTrack
# RUN cd / && \
#     git clone https://github.com/yhsmiley/bytetrack_realtime && \
#     cd bytetrack_realtime && \
#     git checkout 6313471ac1033b6c495bc770ff1fd9f79aeaa91f && \
#     pip3 install --no-cache-dir -e .
