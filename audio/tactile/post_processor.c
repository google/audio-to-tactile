/* Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "audio/tactile/post_processor.h"

#include <stdio.h>
#include <stdlib.h>

#include "audio/dsp/portable/butterworth.h"
#include "audio/tactile/tactor_equalizer.h"

void PostProcessorSetDefaultParams(PostProcessorParams* params) {
  if (params) {
    params->use_equalizer = 1;
    params->mid_gain = 0.31623f;
    params->high_gain = 0.53088f;
    params->gain = 1.0f;
    params->max_amplitude = 0.96f;
    params->cutoff_hz = 1000.0f;
  }
}

int PostProcessorInit(PostProcessor* state,
                      const PostProcessorParams* params,
                      float sample_rate_hz,
                      int num_channels) {
  if (state == NULL || params == NULL) {
    return 0;
  } else if (num_channels > kPostProcessorMaxChannels) {
    fprintf(stderr, "PostProcessorInit: Too many channels: %d\n",
        num_channels);
    return 0;
  } else if (!DesignButterworthOrder2Lowpass(
      params->cutoff_hz, sample_rate_hz, &state->lpf_biquad_coeffs)) {
    fprintf(stderr, "PostProcessorInit: Failed to design lowpass filter.\n");
    return 0;
  }

  if (!params->use_equalizer) {
    state->equalizer_biquad_coeffs[0] = kBiquadFilterIdentityCoeffs;
    state->equalizer_biquad_coeffs[1] = kBiquadFilterIdentityCoeffs;
  } else if (!DesignTactorEqualizer(
        params->mid_gain, params->high_gain,
        sample_rate_hz, state->equalizer_biquad_coeffs)) {
    fprintf(stderr, "PostProcessorInit: Failed to design equalizer.\n");
    return 0;
  }

  /* Absorb gain into the equalizer filter. */
  state->equalizer_biquad_coeffs[0].b0 *= params->gain;
  state->equalizer_biquad_coeffs[0].b1 *= params->gain;
  state->equalizer_biquad_coeffs[0].b2 *= params->gain;

  state->num_channels = num_channels;
  state->max_amplitude = params->max_amplitude;
  PostProcessorReset(state);
  return 1;
}

void PostProcessorReset(PostProcessor* state) {
  int c;
  for (c = 0; c < state->num_channels; ++c){
    PostProcessorChannelState* channel = &state->channels[c];
    BiquadFilterInitZero(&channel->equalizer_biquad_state[0]);
    BiquadFilterInitZero(&channel->equalizer_biquad_state[1]);
    BiquadFilterInitZero(&channel->lpf_biquad_state);
  }
}

void PostProcessorProcessSamples(PostProcessor* state,
                                 float* input_output,
                                 int num_frames) {
  const int num_channels = state->num_channels;
  const float max_amplitude = state->max_amplitude;
  int n;
  for (n = 0; n < num_frames; ++n) {
    int c;
    for (c = 0; c < num_channels; ++c) {
      PostProcessorChannelState* channel = &state->channels[c];
      float sample = *input_output;

      /* Apply equalizer. */
      sample = BiquadFilterProcessOneSample(
          &state->equalizer_biquad_coeffs[0],
          &channel->equalizer_biquad_state[0], sample);
      sample = BiquadFilterProcessOneSample(
          &state->equalizer_biquad_coeffs[1],
          &channel->equalizer_biquad_state[1], sample);

      /* Apply hard clipping. */
      if (sample > max_amplitude) { sample = max_amplitude; }
      if (sample < -max_amplitude) { sample = -max_amplitude; }

      /* Apply lowpass filter. */
      sample = BiquadFilterProcessOneSample(
          &state->lpf_biquad_coeffs,
          &channel->lpf_biquad_state, sample);

      *input_output = sample;
      ++input_output;
    }
  }
}

