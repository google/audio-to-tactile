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

#include "post_processor_cpp.h"  // NOLINT(build/include)

namespace audio_tactile {

PostProcessorWrapper::PostProcessorWrapper() {}

bool PostProcessorWrapper::Init(float output_sample_rate, int block_size,
                                int num_tactors, float gain) {
  const float kDefaultCutOff = 975.0f;

  sample_rate_ = output_sample_rate;
  block_size_ = block_size;

  PostProcessorParams post_processor_params;
  PostProcessorSetDefaultParams(&post_processor_params);
  post_processor_params.gain = gain;
  post_processor_params.cutoff_hz = kDefaultCutOff;
  if (!PostProcessorInit(&post_processor_, &post_processor_params,
                         output_sample_rate, num_tactors)) {
    // Handle error if post processor doesn't initialize.
    // Check post_processor.c for explanation of error sources.
    return true;
  }

  return false;
}

void PostProcessorWrapper::PostProcessSamples(float* input_output) {
  PostProcessorProcessSamples(&post_processor_, input_output, block_size_);
}

}  // namespace audio_tactile
