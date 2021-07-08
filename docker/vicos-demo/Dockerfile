ARG OS_VERSION=ubuntu:16.04
ARG DETECTRON_IMAGE_RUNTIME

FROM ${OS_VERSION} as model-dl

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
# install dependencies for vicos-demo (echolib and echocv)
RUN apt-get update && apt-get install -y build-essential cmake libopencv-dev python-numpy-dev

ENV SCRIPT_DIR /opt/

WORKDIR $SCRIPT_DIR

# install dependency pybind11
RUN git clone https://github.com/wjakob/pybind11.git && cd pybind11 && mkdir build && cd build && cmake -DPYBIND11_TEST=OFF -DPYBIND11_INSTALL=ON .. && make -j install

# install echolib (use version compatible with python2.7)
RUN git clone https://github.com/vicoslab/echolib.git && cd ${SCRIPT_DIR}/echolib && git checkout f80d4dd71f7def76880e8f3ea915978ce56efcfb
RUN cd ${SCRIPT_DIR}//echolib && mkdir build && cd build && cmake -DBUILD_DAEMON=OFF .. && make -j && make install && cd .. && rm -r build

# install echocv (use version compatible with python2.7)
RUN git clone https://github.com/vicoslab/echocv.git && cd ${SCRIPT_DIR}//echocv && git checkout 5a911db977676758f308193f7a13aa7875a7d114
RUN cd ${SCRIPT_DIR}//echocv && mkdir build && cd build && cmake -DBUILD_APPS=OFF .. && make -j && make install && cd .. && rm -r build

##################################
# install traffic sign model and scripts
ARG BACKBONE=resnet50

ENV MODEL_DIR /opt/tsr-detectron-model/

COPY --from=model-dl ${MODEL_DIR}/${BACKBONE} ${MODEL_DIR}
COPY scripts ${SCRIPT_DIR}

RUN chmod +x ${SCRIPT_DIR}/run_main.py

# define entry-point and default arguments
ENTRYPOINT ["/opt/run_main.py", \
            "--cfg=/opt/tsr-detectron-model/config.yaml", \
            "--wts=/opt/tsr-detectron-model/model_final.pkl", \
            "--cat=/opt/dfg_catalog"]