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
// A C++ wrapper for the tactile processor to use as Arduino library or general
// C++ library.

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_PROCESSOR_CPP_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_PROCESSOR_CPP_H_

#include "tactile/tactile_processor.h"
#include "tactile/tuning.h"

namespace audio_tactile {

class TactileProcessorWrapper {
 public:
  TactileProcessorWrapper();
  ~TactileProcessorWrapper();

  // Allocate memory and initialize paramenters for tactile processor.
  bool Init(float sample_rate, int block_size, int decimation_factor);

  // Process the raw microphone data.
  // Returns float array, corresponding to the processing result.
  // The microphone data array is expected to have 'block_size' number of
  // samples (Currently set to 64). The
  // expected output number of samples is `kTactileProcessorNumTactors *
  // block_size / decimation_factor`. (Currently 80 samples). The samples are
  // in interleaved format: output[c + kNumChannel * n] = nth sample for
  // channels. The output will need to be converted to PWM in uint16_t format to
  // send to the PWM hardware module.
  float* ProcessSamples(float* audio_input);

  // Applies tuning settings. Can be called anytime.
  void ApplyTuning(const TuningKnobs& tuning_knobs);

  // Gets the TactileProcessor C object.
  TactileProcessor* get() const {
    return tactile_processor_;
  }

  // Returns the sampling rate after decimation (reduced).
  float GetOutputSampleRate() const {
    return sample_rate_ / decimation_factor_;
  }

  // Returns block size after decimation.
  int GetOutputBlockSize() const { return block_size_ / decimation_factor_; }

  // Returns number of tactor channels.
  int GetOutputNumberTactileChannels() const {
    return kTactileProcessorNumTactors;
  }

 private:
  TactileProcessor* tactile_processor_;
  float* tactile_output_;
  float sample_rate_;
  int block_size_;
  int decimation_factor_;
};

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_TACTILE_PROCESSOR_CPP_H_
