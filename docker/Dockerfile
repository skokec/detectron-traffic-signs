# Use nvidia/cuda image as parent image
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Set environment variables
ENV CAFFE2_SRC_PATH /caffe2_build
ENV DETECTRON_PATH /detectron
ENV COCOAPI_PATH /cocoapi

# Install Basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    Cython \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    mock \
    numpy \
    opencv-python \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

# Clone Detectron repository
RUN git clone https://github.com/skokec/detectron-traffic-signs.git $DETECTRON_PATH

# Clone Caffe2 repository
RUN git clone --branch move-to-pytorch https://github.com/caffe2/caffe2.git $CAFFE2_SRC_PATH
WORKDIR $CAFFE2_SRC_PATH
RUN git rm -r third_party/nervanagpu && git rm -r third_party/aten
RUN git submodule add --force -b old_master https://github.com/zdevito/aten third_party/aten
RUN git submodule init
RUN git config submodule.third_party/eigen.url https://github.com/eigenteam/eigen-git-mirror.git
RUN git submodule update --recursive

# Copy custom caffe2-modules
RUN cp $DETECTRON_PATH/caffe2-modules/* $CAFFE2_SRC_PATH/modules/detectron

# Install Caffe2
RUN mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build
ENV PYTHONPATH /usr/local/caffe2:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2/lib:${LD_LIBRARY_PATH}

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git $COCOAPI_PATH
WORKDIR $COCOAPI_PATH/PythonAPI
RUN make install

# Set up Python modules
WORKDIR $DETECTRON_PATH/lib
RUN make

# Go to Detectron root
WORKDIR $DETECTRON_PATH
