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
#include "DeConv3DLayer.h"

namespace paddle {

REGISTER_LAYER(deconv3d, DeConv3DLayer);

#define DECONV_OUTPUT_SIZE(IN_SIZE, STRID, PAD, KSIZE) \
    (((IN_SIZE) - 1) * (STRID) - 2 * (PAD) + (KSIZE))

bool DeConv3DLayer::init(const LayerMap &layerMap,
                       const ParameterMap &parameterMap) {
    if (!ConvBaseLayer::init(layerMap, parameterMap)) return false;
    // for Deconv, the dimension of Kernel is
    // channel * output * depth * height * weigth
    // Matrix storage format: (output * depth * height * weigth) x  channel
    for (int index = 0; index <  config_.inputs().size(); ++index) {
        M_.push_back(filterChannels_[index]);
        K_.push_back(
                filterPixels_[index] * (numFilters_/groups_[index]));
        weights_[index]->getW()->reshape(
                filterPixels_[index] * numFilters_,
                filterChannels_[index]);
        weights_[index]->getWGrad()->reshape(
                filterPixels_[index] * numFilters_,
                filterChannels_[index]);
    }
    CHECK(inputLayers_.size() == parameters_.size());
    return true;
}


size_t DeConv3DLayer::getSize() {
    CHECK_NE(inputLayers_.size(), 0UL);
    // imgSizeH_.clear();
    // imgSizeW_.clear();
    // imgSizeD_.clear();
    outputH_.clear();
    outputW_.clear();
    outputD_.clear();
    N_.clear();
    No_.clear();
    size_t layerSize = 0;
    for (size_t i = 0; i < inputLayers_.size(); ++i) {
        // imgSizeH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
        // imgSizeW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
        // imgSizeD_.push_back(inputLayers_[i]->getOutput().getFrameDepth());
        outputW_.push_back(
                DECONV_OUTPUT_SIZE(
                        imgSizeW_[i], stride_[i],
                        padding_[i], filterSize_[i]));
        outputH_.push_back(
                DECONV_OUTPUT_SIZE(
                        imgSizeH_[i], strideY_[i],
                        paddingY_[i], filterSizeY_[i]));
        outputD_.push_back(
                DECONV_OUTPUT_SIZE(
                        imgSizeD_[i], strideZ_[i],
                        paddingZ_[i], filterSizeZ_[i]));
        No_.push_back(outputD_[i] * outputH_[i] * outputW_[i]);
        N_.push_back(imgSizeD_[i] * imgSizeH_[i] * imgSizeW_[i]);
        CHECK(layerSize == 0 || N_[i] * size_t(numFilters_) == layerSize);
        layerSize += No_[i] * numFilters_;
    }
    getOutput().setFrameHeight(outputH_[0]);
    getOutput().setFrameWidth(outputW_[0]);
    getOutput().setFrameDepth(outputD_[0]);
    return layerSize;
}

void DeConv3DLayer::forward(PassType passType) {
    Layer::forward(passType);
    int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
    int outWidth = getSize();
    resetOutput(batchSize, outWidth);
    const MatrixPtr output = getOutputValue();

    for (size_t i = 0; i != inputLayers_.size(); ++i) {
        REGISTER_TIMER_INFO("FwdDeConv3D", getName().c_str());
        const MatrixPtr& input = getInputValue(i);
        int width = input->getWidth();
        int M = M_[i];
        int N = N_[i];
        int K = K_[i];
        MatrixPtr weights = weights_[i]->getW();
        Matrix::resizeOrCreate(colBuf_, K * groups_[i] , N, false, useGpu_);

        for (int n = 0; n < batchSize; ++n) {
            real *inData = input->getData() + n * width;
            real *colBufData = colBuf_->getData();
            for (int g = 0; g < groups_[i]; g++) {
               MatrixPtr weight = weights->subMatrix(g * K, K);  // K * M
               MatrixPtr in =
                       Matrix::create(inData, M, N, false, useGpu_);
               MatrixPtr sub_colBufData =
                       Matrix::create(colBufData, K, N, false, useGpu_);
               sub_colBufData->mul(*weight, *in, 1, 0);
               colBufData += K * N;
               inData += M * N;
            }
            colBuf_->col2Vol(output->getData()+ n * output->getWidth(),
                             numFilters_, outputD_[i], outputH_[i], outputW_[i],
                             filterSizeZ_[i], filterSizeY_[i], filterSize_[i],
                             strideZ_[i], strideY_[i], stride_[i],
                             paddingZ_[i], paddingY_[i], padding_[i], 1.0, 1.0);
        }
    }
    if (nullptr != this->biasParameter_) {
        REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
        this->addBias();
    }
    forwardActivation();
}

void DeConv3DLayer::backward(const UpdateCallback &callback) {
    backwardActivation();
    int batchSize = getOutputGrad()->getHeight();
    int out_width = getOutputGrad()->getWidth();
    if (biases_ && biases_->getWGrad()) {
        bpropBiases();
        biases_->getParameterPtr()->incUpdate(callback);
    }
    for (int i =0; i < inputLayers_.size(); ++i) {
        int M = M_[i];
        int N = N_[i];
        int K = K_[i];
        Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
        real* outGrad = getOutputGrad()->getData();
        for (int n = 0; n < batchSize; ++n) {
            REGISTER_TIMER_INFO("BwdDeConv3D", getName().c_str());
            if (weights_[i]->getWGrad() || this->needGradient_) {
                colBuf_->vol2Col(outGrad + n * out_width, numFilters_,
                             outputD_[i], outputH_[i], outputW_[i],
                             filterSizeZ_[i], filterSizeY_[i], filterSize_[i],
                             strideZ_[i], strideY_[i], stride_[i],
                             paddingZ_[i], paddingY_[i], padding_[i]);
            }
            if (weights_[i]->getWGrad()) {
                const MatrixPtr& input = getInputValue(i);
                real *inData = input->getData() + n * input->getWidth();;
                real *WGradData = weights_[i]->getWGrad()->getData();
                for (int g = 0; g < groups_[i]; g++) {
                    MatrixPtr sub_colBufData = colBuf_->subMatrix(g * K, K);
                    MatrixPtr in = Matrix::create(
                            inData, M, N, false, useGpu_);
                    MatrixPtr WGradSub = Matrix::create(
                            WGradData, K, M, false, useGpu_);
                    WGradSub->mul(*sub_colBufData, *(in->getTranspose()), 1, 1);
                    WGradData += K * M;
                    inData += M * N;
                }
                weights_[i]->getParameterPtr()->incUpdate(callback);
            }
            if (this->needGradient_) {
                real* preGrad = getInputGrad(i)->getData();
                for (int g = 0; g < groups_[i]; ++g) {
                    MatrixPtr w = weights_[i]->getW()->subMatrix(g * K, K);
                    MatrixPtr outG = colBuf_->subMatrix(g * K, K);
                    MatrixPtr inGrad = Matrix::create(
                            preGrad, M, N, false, useGpu_);  // M * N
                    inGrad->mul(*(w->getTranspose()), *outG, 1, 0);
                    preGrad += M * N;
                }
            }
            REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
        }
    }
}

void DeConv3DLayer::bpropWeights(int i) { }
void DeConv3DLayer::bpropData(int i) {  }

void DeConv3DLayer::bpropBiases() {
    MatrixPtr outGrad = getOutputGrad();
    biases_->getWGrad()->reshape(
            biases_->getWGrad()->width_, biases_->getWGrad()->height_);

    if (this->sharedBiases_) {
        biases_->getWGrad()->collectSharedBias(*outGrad, 1.0f);
    } else {
        biases_->getWGrad()->collectBias(*outGrad, 1.0f);
    }
}

void DeConv3DLayer::addBias() {
    MatrixPtr output = getOutputValue();
    biases_->getW()->reshape(biases_->getW()->width_, biases_->getW()->height_);
    if (this->sharedBiases_) {
        output->addSharedBias(*(biases_->getW()), 1.0f);
    } else {
        output->addBias(*(biases_->getW()), 1.0f);
    }
}

}  // namespace paddle
