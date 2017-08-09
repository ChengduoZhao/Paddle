/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "Conv3DLayer.h"

namespace paddle {

REGISTER_LAYER(conv3d, Conv3DLayer);

bool Conv3DLayer::init(const LayerMap &layerMap,
                       const ParameterMap &parameterMap) {
  if (!ConvBaseLayer::init(layerMap, parameterMap))
      return false;
  int index = 0;
  for (auto &inputConfig : config_.inputs()) {
      const ConvConfig &conf = inputConfig.conv_conf();
      M_.push_back(numFilters_ / conf.groups());
      K_.push_back(
              conf.filter_channels() * conf.filter_size_z() * \
      conf.filter_size_y() * conf.filter_size());
      weights_[index]->getW()->reshape(
              weights_[index]->getW()->getWidth(),
              weights_[index]->getW()->getHeight());
      weights_[index]->getWGrad()->reshape(
              weights_[index]->getWGrad()->getWidth(),
              weights_[index]->getWGrad()->getHeight());
      ++index;
  }
  CHECK(inputLayers_.size() == parameters_.size());
  return true;
}


size_t Conv3DLayer::getSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  // imgSizeH_.clear();
  // imgSizeW_.clear();
  // imgSizeD_.clear();
  outputH_.clear();
  outputW_.clear();
  outputD_.clear();
  N_.clear();
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
      // imgSizeH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
      // imgSizeW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
      // imgSizeD_.push_back(inputLayers_[i]->getOutput().getFrameDepth());
      outputW_.push_back(outputSize(
              imgSizeW_[i], filterSize_[i],
              padding_[i], stride_[i], true));
      outputH_.push_back(outputSize(
              imgSizeH_[i], filterSizeY_[i],
              paddingY_[i], strideY_[i], true));
      outputD_.push_back(outputSize(
              imgSizeD_[i], filterSizeZ_[i],
              paddingZ_[i], strideZ_[i], true));

      N_.push_back(outputD_[i] * outputH_[i] * outputW_[i]);
      CHECK(layerSize == 0 || N_[i] * size_t(numFilters_) == layerSize);
      layerSize += N_[i] * numFilters_;
  }
  getOutput().setFrameHeight(outputH_[0]);
  getOutput().setFrameWidth(outputW_[0]);
  getOutput().setFrameDepth(outputD_[0]);
  return layerSize;
}

void Conv3DLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  int outWidth = getSize();
  resetOutput(batchSize, outWidth);
  const MatrixPtr output = getOutputValue();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
      REGISTER_TIMER_INFO("FwdConv3D", getName().c_str());
      const MatrixPtr& input = getInputValue(i);
      int width = input->getWidth();
      int M = M_[i];
      int N = N_[i];
      int K = K_[i];
      Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
      MatrixPtr weights = weights_[i]->getW();
      for (int n = 0; n < batchSize; ++n) {
          colBuf_->vol2Col(input->getData() + n * width, channels_[i],
                           imgSizeD_[i], imgSizeH_[i], imgSizeW_[i],
                           filterSizeZ_[i], filterSizeY_[i], filterSize_[i],
                           strideZ_[i], strideY_[i], stride_[i],
                           paddingZ_[i], paddingY_[i], padding_[i]);

          real *outData = output->getData() + n * outWidth;
          MatrixPtr oData =
                  Matrix::create(outData, groups_[i] * M, N, false, useGpu_);
          for (int g = 0; g < groups_[i]; g++) {
              MatrixPtr weight = weights->subMatrix(g * M, M);
              MatrixPtr in = colBuf_->subMatrix(g * K, K);
              MatrixPtr out = oData->subMatrix(g * M, M);
              out->mul(*weight, *in, 1, 0);
          }
      }
  }
  if (nullptr != this->biasParameter_) {
      REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
      this->addBias();
  }
  forwardActivation();
}

void Conv3DLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
      bpropBiases();
      biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr outGrad = getOutputGrad();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
      REGISTER_TIMER_INFO("BwdConv3D", getName().c_str());
      if (weights_[i]->getWGrad()) {
          bpropWeights(i);
      }
      if (this->needGradient_) {
          bpropData(i);
      }
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
  }
}

void Conv3DLayer::bpropWeights(int i) {
  int M = M_[i];
  int N = N_[i];
  int K = K_[i];
  const MatrixPtr& input = getInputValue(i);
  int width = input->getWidth();
  Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
  MatrixPtr wGrad = weights_[i]->getWGrad();
  real* outGrad = getOutputGrad()->getData();
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();

  for (int n = 0; n < batchSize; ++n) {
      colBuf_->vol2Col(input->getData() + n * width, channels_[i],
                       imgSizeD_[i], imgSizeH_[i], imgSizeW_[i],
                       filterSizeZ_[i], filterSizeY_[i], filterSize_[i],
                       strideZ_[i], strideY_[i], stride_[i],
                       paddingZ_[i], paddingY_[i], padding_[i]);
      outGrad += n * getOutputGrad()->getWidth();
      MatrixPtr oGrad =
              Matrix::create(outGrad, groups_[i] * M, N, false, useGpu_);
      for (int g = 0; g < groups_[i]; ++g) {
          MatrixPtr in = colBuf_->subMatrix(g * K, K);
          MatrixPtr outG = oGrad->subMatrix(g * M, M);
          MatrixPtr wGradSub = wGrad->subMatrix(g * M, M);
          wGradSub->mul(*outG, *(in->getTranspose()), 1, 1);
      }
  }
}

void Conv3DLayer::bpropData(int i) {
  int M = M_[i];
  int N = N_[i];
  int K = K_[i];
  Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
  MatrixPtr weight = weights_[i]->getW();
  real* outGrad = getOutputGrad()->getData();
  real* preGrad = getInputGrad(i)->getData();
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  for (int n = 0; n < batchSize; ++n) {
      outGrad += n * getOutputGrad()->getWidth();
      preGrad += n * getInputGrad(i)->getWidth();
      MatrixPtr outG =
              Matrix::create(outGrad, M * groups_[i], N, false, useGpu_);
      for (int g = 0; g < groups_[i]; ++g) {
          MatrixPtr w = weight->subMatrix(g * M, M);
          MatrixPtr outGradSub = outG->subMatrix(g * M, M);
          MatrixPtr inGrad = colBuf_->subMatrix(g * K, K);
          inGrad->mul(*(w->getTranspose()), *outGradSub, 1, 0);
      }
      colBuf_->col2Vol(preGrad, channels_[i],
                       imgSizeD_[i], imgSizeH_[i], imgSizeW_[i],
                       filterSizeZ_[i], filterSizeY_[i], filterSize_[i],
                       strideZ_[i], strideY_[i], stride_[i],
                       paddingZ_[i], paddingY_[i], padding_[i],
                       1.0, 1.0);
  }
}

void Conv3DLayer::bpropBiases() {
  MatrixPtr outGrad = getOutputGrad();
  biases_->getWGrad()->reshape(
          biases_->getWGrad()->width_, biases_->getWGrad()->height_);

  if (this->sharedBiases_) {
      biases_->getWGrad()->collectSharedBias(*outGrad, 1.0f);
  } else {
      biases_->getWGrad()->collectBias(*outGrad, 1.0f);
  }
}

void Conv3DLayer::addBias() {
  MatrixPtr output = getOutputValue();
  biases_->getW()->reshape(biases_->getW()->width_, biases_->getW()->height_);
  if (this->sharedBiases_) {
      output->addSharedBias(*(biases_->getW()), 1.0f);
  } else {
      output->addBias(*(biases_->getW()), 1.0f);
  }
}

}  // namespace paddle
