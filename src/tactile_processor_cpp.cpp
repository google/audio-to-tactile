// Copyright 2020 Google LLC
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

#include "tactile_processor_cpp.h"  // NOLINT(build/include)

#include <stdlib.h>
#include <string.h>

#include "tactile/tuning.h"

namespace audio_tactile {

TactileProcessorWrapper::TactileProcessorWrapper()
    : tactile_processor_(nullptr), tactile_output_(nullptr) {}

TactileProcessorWrapper::~TactileProcessorWrapper() {
  TactileProcessorFree(tactile_processor_);
  free(tactile_output_);
}

bool TactileProcessorWrapper::Init(float sample_rate, int block_size,
                                   int decimation_factor) {
  const float kDefaultGain = 4.0f;
  const float kDefaultCutOff = 975.0f;

  sample_rate_ = sample_rate;
  block_size_ = block_size;
  decimation_factor_ = decimation_factor;
  const int frames_per_carl_block = block_size_ / decimation_factor_;
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  params.frontend_params.input_sample_rate_hz = sample_rate;
  params.frontend_params.block_size = block_size;
  params.decimation_factor = decimation_factor;
  tactile_processor_ = TactileProcessorMake(&params);
  if (tactile_processor_ == NULL) {
    // Error initializing due to issues such as memory allocation or if filter
    // design fails or sample rate is too low compared to cutoffs.
    return true;
  }

  // Allocate tactile buffer with space for one CARL block's worth of output.
  // Currently it is 320 bytes = 8 * 10 * 4
  tactile_output_ = (float*)malloc(frames_per_carl_block *
                                   kTactileProcessorNumTactors * sizeof(float));

  return false;
}

float* TactileProcessorWrapper::ProcessSamples(float* audio_input) {
  TactileProcessorProcessSamples(tactile_processor_, audio_input,
                                 tactile_output_);
  return tactile_output_;
}

float TactileProcessorWrapper::SetOutputGain(int value) {
  float value_db = TuningSetOutputGain(tactile_processor_, value);
  return value_db;
}

float TactileProcessorWrapper::SetDenoising(int value) {
  float value_delta = TuningSetDenoising(tactile_processor_, value);
  return value_delta;
}

float TactileProcessorWrapper::SetCompression(int value) {
  float value_beta = TuningSetCompression(tactile_processor_, value);
  return value_beta;
}

}  // namespace audio_tactile
