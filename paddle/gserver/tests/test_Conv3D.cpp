/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "paddle/gserver/layers/Conv3DLayer.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_double(checkgrad_eps);
DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(prev_batch_state);

void testConv3D_Layer() {
  // filter size
  const int NUM_FILTERS = 6;
  // const int CHANNELS = 3;
  const int FILTER_SIZE = 3;
  const int FILTER_SIZE_Y = 3;
  const int FILTER_SIZE_Z = 3;

  // input image
  const int NUM_IMG = 2;
  const int CHANNELS = 3;
  const int IMAGE_SIZE = 9;
  const int IMAGE_SIZE_Y = 9;
  const int IMAGE_SIZE_Z = 9;   //  2, 3, 5, 5, 5

  // Setting up conv-trans layer
  TestConfig config;
  config.biasSize = NUM_FILTERS;
  config.layerConfig.set_type("conv3d");
  config.layerConfig.set_name("conv3DDD");
  config.layerConfig.set_num_filters(NUM_FILTERS);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  LayerInputConfig *input = config.layerConfig.add_inputs();
  ConvConfig *conv = input->mutable_conv_conf();

  conv->set_channels(CHANNELS);
  conv->set_filter_size(FILTER_SIZE);
  conv->set_filter_size_y(FILTER_SIZE_Y);
  conv->set_filter_size_z(FILTER_SIZE_Z);

  conv->set_padding(0);
  conv->set_padding_y(0);
  conv->set_padding_z(0);

  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_stride_z(2);

  conv->set_img_size(IMAGE_SIZE);
  conv->set_img_size_y(IMAGE_SIZE_Y);
  conv->set_img_size_z(IMAGE_SIZE_Z);

  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),  /*  caffeMode */ true));
  conv->set_output_y(outputSize(conv->img_size_y(),
                                conv->filter_size_y(),
                                conv->padding_y(),
                                conv->stride_y(), /*  caffeMode */ true));
  conv->set_output_z(outputSize(conv->img_size_z(),
                                conv->filter_size_z(),
                                conv->padding_z(),
                                conv->stride_z(), /*  caffeMode */ true));

  config.layerConfig.set_size(
          conv->output_x() * conv->output_y() * conv->output_z() * NUM_FILTERS);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  config.inputDefs.push_back(
          {INPUT_DATA, "layer_0",
           CHANNELS*IMAGE_SIZE*IMAGE_SIZE_Y*IMAGE_SIZE_Z,
           conv->filter_channels() * \
           FILTER_SIZE*FILTER_SIZE_Y*FILTER_SIZE_Z*NUM_FILTERS});

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
          config, &dataLayers, &datas, &layerMap,
          "convTrans", NUM_IMG, false, false);
  dataLayers[0]->getOutput().value->zero();
  dataLayers[0]->getOutput().value->add(1.0);

  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr convtLayer;
  initTestLayer(config, &layerMap, &parameters, &convtLayer);
  convtLayer->getBiasParameter()->zeroMem();
  convtLayer->getBiasParameter()->getBuf(PARAMETER_VALUE)->add(0.5);
  convtLayer->getParameters()[0]->zeroMem();
  convtLayer->getParameters()[0]->getBuf(PARAMETER_VALUE)->add(1.0);
  convtLayer->forward(PASS_GC);

  convtLayer->backward(nullptr);

  int num = convtLayer->getOutput().getBatchSize();
  int width = convtLayer->getOutput().getFrameWidth();
  int height = convtLayer->getOutput().getFrameHeight();
  int depth = convtLayer->getOutput().getFrameDepth();
  int channel =
          convtLayer->getOutput().value->getWidth() / (width*height*depth);

  for (int t = 0; t < num; t++) {
    cout << "sampel_" << t << "\n";
    for (int c =0 ; c < channel; c++) {
      for (int k = 0; k < depth; ++k) {
        for (int i = 0; i < height; i++) {
          for (int j = 0; j < width; j++) {
        	  CHECK_EQ(convtLayer->getOutput().value->data_\
        	            [t * channel * height * width + \
        	             c * depth * width * height + \
        	             k * width * height + \
        	             i * width + j], 8.5);
          }
          cout << "\n";
        }
        cout << "\n";
      }
      cout << "\n";
    }
  }
}

TEST(Conv3D_Layer, conv3D_Layer) {
  testConv3D_Layer();
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}

