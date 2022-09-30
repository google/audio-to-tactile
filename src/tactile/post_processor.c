/* Copyright 2019, 2022 Google LLC
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

#include "tactile/post_processor.h"

#include <stdio.h>
#include <stdlib.h>

#include "dsp/butterworth.h"
#include "dsp/fast_fun.h"
#include "tactile/tactor_equalizer.h"

/* `output_limit` is constrained to [kLimitMin, kLimitMax]. */
static const float kLimitMin = 1.0f;
static const float kLimitMax = 6.0f;
/* When the battery is low, `output_limit` is reduced by this factor. */
static const float kLimitReduceFactor = 0.8f;
/* Otherwise, `output_limit` is increased at this rate in Db/s. */
static const float kLimitGrowRateDbPerSecond = 2.0f;

/* When the battery is low, output power is reduced to `kRecoveryLimit` for the
 * next `kRecoveryNumBuffers` buffers, so that hopefully the battery recovers.
 */
static const float kRecoveryLimit = 0.25f;
static const int kRecoveryNumBuffers = 10;

void PostProcessorSetDefaultParams(PostProcessorParams* params) {
  if (params) {
    params->use_equalizer = 1;
    params->mid_gain = 0.31623f;
    params->high_gain = 0.53088f;
    params->gain = 1.0f;
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
  state->limit_grow_coeff = (float)exp(
      (float)(M_LN10 / 10.0) * kLimitGrowRateDbPerSecond / sample_rate_hz);
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
  state->output_limit = kLimitReduceFactor * kLimitMax;
  state->recovery = 0;
}

void PostProcessorLowBattery(PostProcessor* state) {
  if (!state->recovery) {
    /* When battery goes low, reduce limit by a bit. */
    state->output_limit *= kLimitReduceFactor;
    if (state->output_limit < kLimitMin) { state->output_limit = kLimitMin; }
  }
  state->recovery = kRecoveryNumBuffers;
}

void PostProcessorProcessSamples(PostProcessor* state,
                                 float* input_output,
                                 int num_frames) {
  float output_limit = state->output_limit;

  if (state->recovery) {
    /* Use extra low limit for a few buffers to give battery time to recover. */
    output_limit = kRecoveryLimit;
    --state->recovery;
  } else if (output_limit < kLimitMax) {
    /* Otherwise, slowly grow the limit. */
    output_limit *= state->limit_grow_coeff;
    if (output_limit > kLimitMax) { output_limit = kLimitMax; }
    state->output_limit = output_limit;
  }

  const int num_channels = state->num_channels;
  int n;
  for (n = 0; n < num_frames; ++n) {
    float power = 0.0f;

    int c;
    for (c = 0; c < num_channels; ++c) {
      PostProcessorChannelState* channel = &state->channels[c];
      float sample = input_output[c];

      /* Apply equalizer. */
      sample = BiquadFilterProcessOneSample(
          &state->equalizer_biquad_coeffs[0],
          &channel->equalizer_biquad_state[0], sample);
      sample = BiquadFilterProcessOneSample(
          &state->equalizer_biquad_coeffs[1],
          &channel->equalizer_biquad_state[1], sample);

      /* Apply hard clipping. */
      const float kClipAmplitude = 0.96f;
      if (sample > kClipAmplitude) { sample = kClipAmplitude; }
      if (sample < -kClipAmplitude) { sample = -kClipAmplitude; }

      /* Apply lowpass filter. */
      sample = BiquadFilterProcessOneSample(
          &state->lpf_biquad_coeffs,
          &channel->lpf_biquad_state, sample);

      power += sample * sample;
      input_output[c] = sample;
    }

    /* Limit the power when needed. */
    if (power > output_limit) {
      const float limiter_gain = FastPow(output_limit / power, 0.5f);
      for (c = 0; c < num_channels; ++c) {
        input_output[c] *= limiter_gain;
      }
    }

    input_output += num_channels;
  }
}
