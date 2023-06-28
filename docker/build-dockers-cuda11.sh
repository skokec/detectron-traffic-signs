#!/bin/bash
set -e

docker build detectron_cuda11 -t detectron-traffic-signs:ubuntu16.04-cuda11.1-cudnn8  \
                        --build-arg CUDA_VERSION=11.1-cudnn8 \
                        --build-arg UBUNTU_VERSION=16.04
