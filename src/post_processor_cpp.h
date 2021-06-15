// Copyright 2020-2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// A C++ wrapper for the post processor to use as Arduino library or general
// C++ library.

#ifndef AUDIO_TO_TACTILE_SRC_POST_PROCESSOR_CPP_H_
#define AUDIO_TO_TACTILE_SRC_POST_PROCESSOR_CPP_H_

#include "tactile/post_processor.h"
#include "tactile/tactile_processor.h"

namespace audio_tactile {

class PostProcessorWrapper {
 public:
  PostProcessorWrapper();

  // Initialize paramenters for the post processor.
  bool Init(float output_sample_rate, int block_size, int num_channels,
            float gain);

  // Process the samples.
  // The post processor expects a float array pointer from the tactile
  // processor output of `block_size * num_channels` samples. The typical array
  // size is 80 (8 samples for each channel). The samples are in interleaved
  // format: output[c + kNumChannel * n] = nth sample for channel c. The output
  // will need to be converted to PWM in uint16_t format to send to the PWM
  // hardware module. The range of input and output is a float from -1 to 1.
  void PostProcessSamples(float* input_output);

 private:
  PostProcessor post_processor_;
  float sample_rate_;
  int block_size_;
};

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_POST_PROCESSOR_CPP_H_
