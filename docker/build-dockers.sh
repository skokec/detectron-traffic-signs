#!/bin/bash

docker build detectron -t detectron-traffic-signs:ubuntu16.04-cuda10.0-cudnn7  \
                        --build-arg CUDA_VERSION=10.0-cudnn7 \
                        --build-arg OS_VERSION=ubuntu16.04

docker build vicos-demo -t tsr-vicos-demo:ubuntu16.04-cuda10.0-cudnn7-resnet50 \
                        --build-arg OS_VERSION=ubuntu:16.04 \
                        --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda10.0-cudnn7 \
                        --build-arg BACKBONE=resnet50

docker build vicos-demo -t tsr-vicos-demo:ubuntu16.04-cuda10.0-cudnn7-resnet101 \
                        --build-arg OS_VERSION=ubuntu:16.04 \
                        --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda10.0-cudnn7 \
                        --build-arg BACKBONE=resnet101
