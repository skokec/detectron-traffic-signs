#!/bin/bash
set -e

docker build detectron_cuda11 -t detectron-traffic-signs:ubuntu16.04-cuda11.1-cudnn8  \
                        --build-arg CUDA_VERSION=11.1-cudnn8 \
                        --build-arg OS_VERSION=ubuntu16.04

docker build vicos-demo -t tsr-vicos-demo:ubuntu16.04-cuda11.1-cudnn8-resnet50 \
                        --build-arg OS_VERSION=ubuntu:16.04 \
                        --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda11.1-cudnn8 \
                        --build-arg BACKBONE=resnet50

docker build vicos-demo -t tsr-vicos-demo:ubuntu16.04-cuda11.1-cudnn8-resnet101 \
                        --build-arg OS_VERSION=ubuntu:16.04 \
                        --build-arg DETECTRON_IMAGE_RUNTIME=detectron-traffic-signs:ubuntu16.04-cuda11.1-cudnn8 \
                        --build-arg BACKBONE=resnet101
