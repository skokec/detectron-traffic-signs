#!/bin/bash

docker build detectron -t detectron-traffic-signs:ubuntu16.04-cuda10.0-cudnn7  \
                        --build-arg CUDA_VERSION=10.0-cudnn7 \
                        --build-arg UBUNTU_VERSION=16.04
