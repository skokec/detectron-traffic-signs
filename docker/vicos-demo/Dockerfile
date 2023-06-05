ARG UBUNTU_VERSION=16.04
ARG DETECTRON_IMAGE_RUNTIME

FROM echolib:${UBUNTU_VERSION} AS echolib-base
FROM ubuntu:${UBUNTU_VERSION} AS model-dl

RUN apt-get update && apt-get install -y unzip

ENV MODEL_DIR /opt/tsr-detectron-model

WORKDIR $MODEL_DIR/resnet101

ADD http://box.vicos.si/skokec/villard/tsr-tracker/detectron-ResNet101.tar.gz $MODEL_DIR/resnet101

RUN tar -zxvf detectron-ResNet101.tar.gz && ln -s ResNet101_FPN_840px_ohem_weighted_cls_even_rois.yaml config.yaml
RUN rm -rf $MODEL_DIR/detectron-ResNet101.tar.gz

WORKDIR $MODEL_DIR/resnet50

ADD http://box.vicos.si/skokec/villard/tsr-tracker/detectron-ResNet50.tar.gz $MODEL_DIR/resnet50

RUN tar -zxvf detectron-ResNet50.tar.gz && ln -s ResNet50_FPN_840px_ohem_weighted_cls_even_rois.yaml config.yaml
RUN rm -rf $MODEL_DIR/detectron-ResNet50.tar.gz

FROM ${DETECTRON_IMAGE_RUNTIME}

LABEL maintainer "domen.tabernik@fri.uni-lj.si"

######################################

ENV SCRIPT_DIR /opt/

WORKDIR $SCRIPT_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
    python-dev python-pip python-numpy libopencv-dev python-numpy-dev libssl-dev wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 

# Upgrade cmake

#RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.0-rc5/cmake-3.23.0-rc5.tar.gz
#RUN tar -xf cmake-3.23.0-rc5.tar.gz
#WORKDIR $SCRIPT_DIR/cmake-3.23.0-rc5
#RUN ./bootstrap && make -j5 && make install

#WORKDIR $SCRIPT_DIR

#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0

##################################
# Install gcc 8

RUN apt-get update && apt-get install build-essential software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update -y && apt-get install gcc-8 g++-8 -y && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8 && update-alternatives --config gcc

RUN apt-get update && apt-get install -y build-essential cmake libopencv-dev python-numpy-dev

COPY --from=echolib-base /usr/local/ /usr/local/

##################################
# install traffic sign model and scripts

ARG BACKBONE=resnet50

ENV MODEL_DIR /opt/tsr-detectron-model/

COPY --from=model-dl ${MODEL_DIR}/${BACKBONE} ${MODEL_DIR}
COPY scripts ${SCRIPT_DIR}

RUN chmod +x ${SCRIPT_DIR}/run_main.py

##################################
# Install opencv

#RUN git clone https://github.com/opencv/opencv.git
#RUN mkdir $SCRIPT_DIR/opencv/build
#WORKDIR $SCRIPT_DIR/opencv/build

#RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=RELEASE -DINSTALL_PYTHON_EXAMPL=ON -DINSTALL_C_EXAMPLES=OFF -DPYTHON_EXECUTABLE=/usr/bin/python3.5 -D__INSTALL_PATH_PYTHON3=/usr/local/lib/python3.5/dist-packages ..
#RUN make install -j5 .

#RUN cp -R /usr/local/lib/python3.5/site-packages/cv2 /usr/local/lib/python3.5/dist-packages

#WORKDIR $SCRIPT_DIR

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py", \
            "--cfg=/opt/tsr-detectron-model/config.yaml", \
            "--wts=/opt/tsr-detectron-model/model_final.pkl", \
            "--cat=/opt/dfg_catalog"]

# /opt/run_main.py --cfg=/opt/tsr-detectron-model/config.yaml --wts=/opt/tsr-detectron-model/model_final.pkl --cat=/opt/dfg_catalog 
