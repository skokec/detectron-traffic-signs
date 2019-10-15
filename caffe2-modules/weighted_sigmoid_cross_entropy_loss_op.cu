/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/context_gpu.h"
#include "weighted_sigmoid_cross_entropy_loss_op.h"

namespace caffe2 {

namespace {
__global__ void ElementwiseMaxKernel(const int n, float* data, const float a) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    data[index] = (data[index] > a) ? data[index] : a;
  }
}

__global__ void WeightedSigmoidCrossEntropyLossKernel(
    const int n,
    const float* logits,
    const int* targets,
    const float* weights,
    float* losses,
    float* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1) {
      losses[index] = 0.;
      counts[index] = 0.;
    } else {
      losses[index] = weights[index] * (
          -1. * logits[index] * (targets[index] - (logits[index] >= 0)) +
          logf(
              1 +
              expf(logits[index] - 2 * logits[index] * (logits[index] >= 0))));
      counts[index] = 1.;
    }
  }
}

__global__ void WeightedSigmoidCrossEntropyLossGradientKernel(
    const int n,
    const float* logits,
    const int* targets,
    const float* weights,
    float* d_logits,
    float* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1) {
      d_logits[index] = 0.;
      counts[index] = 0.;
    } else {
      d_logits[index] = (1. / (1. + expf(-logits[index])) - targets[index]) * weights[index];
      counts[index] = 1.;
    }
  }
}
} // namespace

template <>
bool WeightedSigmoidCrossEntropyLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& W = Input(2);
  auto* avg_loss = Output(0);

  CAFFE_ENFORCE(
      X.size() == T.size(),
      "Logit and target must have the same size",
      "(",
      X.size(),
      " vs. ",
      T.size(),
      ")");
  CAFFE_ENFORCE(
      X.size() == W.size(),
      "Logit and weights must have the same size",
      "(",
      X.size(),
      " vs. ",
      W.size(),
      ")");  
  avg_loss->Resize(vector<TIndex>());
  counts_.ResizeLike(X);
  losses_.ResizeLike(X);
  normalizer_.Resize(vector<TIndex>());
  WeightedSigmoidCrossEntropyLossKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      T.data<int>(),
      W.data<float>(),
      losses_.mutable_data<float>(),
      counts_.mutable_data<float>());
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  if (normalize_) {
    float* normalizer_data = normalizer_.mutable_data<float>();
    math::Sum<float, CUDAContext>(
        counts_.size(), counts_.data<float>(), normalizer_data, &context_);
    // Prevent division by zero is all counts are zero
    ElementwiseMaxKernel<<<
        CAFFE_GET_BLOCKS(normalizer_.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(normalizer_.size(), normalizer_data, 1e-5);
    math::Div<float, CUDAContext>(
        1, avg_loss_data, normalizer_data, avg_loss_data, &context_);
  }
  math::Scale<float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}

template <>
bool WeightedSigmoidCrossEntropyLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& W = Input(2);
  auto& d_avg_loss = Input(3);
  auto* dX = Output(0);

  dX->ResizeLike(X);
  counts_.ResizeLike(X);
  normalizer_.Resize(vector<TIndex>());
  WeightedSigmoidCrossEntropyLossGradientKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      X.data<float>(),
      T.data<int>(),
      W.data<float>(),
      dX->mutable_data<float>(),
      counts_.mutable_data<float>());
  if (normalize_) {
    float* normalizer_data = normalizer_.mutable_data<float>();
    math::Sum<float, CUDAContext>(
        counts_.size(), counts_.data<float>(), normalizer_data, &context_);
    // Prevent division by zero is all counts are zero
    ElementwiseMaxKernel<<<
        CAFFE_GET_BLOCKS(normalizer_.size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(normalizer_.size(), normalizer_data, 1e-5);
    math::Div<float, CUDAContext>(
        1,
        d_avg_loss.data<float>(),
        normalizer_data,
        normalizer_data,
        &context_);
    math::Scale<float, CUDAContext>(
        1, scale_, normalizer_data, normalizer_data, &context_);
    math::Scale<float, CUDAContext>(
        dX->size(),
        normalizer_data,
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
  } else {
    math::Scale<float, CUDAContext>(
        dX->size(),
        scale_,
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
    math::Scale<float, CUDAContext>(
        dX->size(),
        d_avg_loss.data<float>(),
        dX->data<float>(),
        dX->mutable_data<float>(),
        &context_);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    WeightedSigmoidCrossEntropyLoss,
    WeightedSigmoidCrossEntropyLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    WeightedSigmoidCrossEntropyLossGradient,
    WeightedSigmoidCrossEntropyLossGradientOp<float, CUDAContext>);
} // namespace caffe2
