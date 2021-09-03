ARG CUDA_VERSION=11.1-cudnn8
ARG OS_VERSION=ubuntu16.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VERSION} as detectron-build
LABEL maintainer "domen.tabernik@fri.uni-lj.si"

# Set environment variables
ENV CAFFE2_SRC_PATH /opt/caffe2
ENV CAFFE2_INSTALL_PATH /opt/caffe2-install
ENV DETECTRON_PATH /opt/detectron
ENV COCOAPI_PATH /opt/cocoapi

# Install Basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    build-essential \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-pip \
    python-pydot \
    python-setuptools \
    libnccl2 \
    libnccl-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install cmake >= v3.14 since caffe2 will not compile correctly with older versions
RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.sh && \
    sh cmake-3.17.1-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && rm /tmp/cmake-*.sh

# Install Python dependencies
RUN pip install --no-cache-dir pip==19.3.1  wheel==0.35.1
RUN pip install --no-cache-dir \
    Cython==0.29.21 \
    flask \
    future==0.18.2 \
    hypothesis==4.57.1 \
    matplotlib==2.2.5 \
    mock==3.0.5 \
    numpy==1.16.6 \
    opencv-python==4.1.0.25 \
    protobuf==3.13.0 \
    pydot==1.4.2 \
    python-nvd3 \
    pyyaml==3.12 \
    requests==2.25.1 \
    scikit-image==0.14.5 \
    scipy==1.2.3 \
    setuptools==44.0.0 \
    six==1.15.0

# Clone Detectron repository
RUN git clone --depth 1 https://github.com/skokec/detectron-traffic-signs.git $DETECTRON_PATH

# Clone Caffe2 repository, fix submodules, add detectron files and install
RUN git clone --depth 1 --branch move-to-pytorch https://github.com/skokec/caffe2.git $CAFFE2_SRC_PATH && echo "      "
WORKDIR $CAFFE2_SRC_PATH
RUN git rm -r third_party/nervanagpu && git rm -r third_party/aten
RUN git submodule add --force -b old_master https://github.com/zdevito/aten third_party/aten
RUN git submodule init
RUN git config submodule.third_party/eigen.url https://github.com/eigenteam/eigen-git-mirror.git
RUN git submodule update --recursive

# fix for CUDA11 and compute 8.6
RUN sed -i -e "s,#if __CUDA_ARCH__ <= 700,#if __CUDA_ARCH__ <= 860,g" caffe2/utils/GpuDefs.cuh

# fix for eigen with cuda 9.1 or higher
RUN sed -i -e "s,math_functions.hpp,cuda_runtime.h,g" third_party/eigen/Eigen/Core

# fix for caffe2/gloo with cuda 11.0 or higher
RUN sed -i -e "s,attr.memoryType,attr.type,g" caffe2/core/common_gpu.cc
RUN sed -i -e "s,attr.memoryType,attr.type,g" third_party/gloo/gloo/cuda.h

# Copy custom caffe2-modules
RUN cp $DETECTRON_PATH/caffe2-modules/* $CAFFE2_SRC_PATH/modules/detectron

# CUDA binary and ptx compute version for caffe2
ARG CUDA_ARCH_BIN="86"
ARG CUDA_ARCH_PTX="61"

# Build and install Caffe2
RUN mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
    -DCUDA_ARCH_PTX=${CUDA_ARCH_PTX} \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    -DUSE_OPENMP=ON \
    -DUSE_NCCL=ON \
    -DUSE_CUDA=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE \
    -DCMAKE_INSTALL_PREFIX=$CAFFE2_INSTALL_PATH && \
    make -j"$(nproc)" install


FROM nvidia/cuda:${CUDA_VERSION}-runtime-${OS_VERSION}
LABEL maintainer "domen.tabernik@fri.uni-lj.si"

# Set environment variables
ENV CAFFE2_SRC_PATH /opt/caffe2
ENV CAFFE2_INSTALL_PATH /opt/caffe2-install
ENV DETECTRON_PATH /opt/detectron
ENV COCOAPI_PATH /opt/cocoapi

# Install Basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    build-essential git \
    libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgoogle-glog0v5 libgflags2v5 liblmdb0 libleveldb1v5 libprotobuf9v5 \
    libopencv-core2.4v5 libopencv-highgui2.4v5 \
    python-pip \
    python-dev \
    python-setuptools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pip==19.3.1  wheel==0.35.1
RUN pip install --no-cache-dir \
    Cython==0.29.21 \
    future==0.18.2 \
    hypothesis==4.57.1 \
    matplotlib==2.2.5 \
    numpy==1.16.6 \
    opencv-python==4.1.0.25 \
    protobuf==3.13.0 \
    pyyaml==3.12 \
    requests==2.25.1 \
    scipy==1.2.3 \
    setuptools==44.0.0 \
    six==1.15.0


# Install Caffe2
COPY --from=detectron-build $CAFFE2_INSTALL_PATH $CAFFE2_INSTALL_PATH

ENV PYTHONPATH $CAFFE2_INSTALL_PATH/lib/python2.7/dist-packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH $CAFFE2_INSTALL_PATH/lib:$LD_LIBRARY_PATH

RUN ln -s $CAFFE2_INSTALL_PATH/lib/ $CAFFE2_INSTALL_PATH/lib/python2.7/dist-packages/lib && ldconfig

# Build the COCO API
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git $COCOAPI_PATH && \
    cd $COCOAPI_PATH/PythonAPI && \
    make install

# Clone Detectron repository
RUN git clone --depth 1 https://github.com/skokec/detectron-traffic-signs.git $DETECTRON_PATH

# Build Python modules for detectron
WORKDIR $DETECTRON_PATH/lib
RUN make -j && ldconfig

# Go to Detectron root
WORKDIR $DETECTRON_PATH
