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

#include "tactile/references/yuan2005/yuan2005.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp/butterworth.h"

void Yuan2005SetDefaultParams(Yuan2005Params* params) {
  if (params) {
    params->sample_rate_hz = 16000.0f;
    params->low_channel.denoising_threshold = 1e-4f;
    params->low_channel.sensory_offset = 0.03f;  /* Roughly -30 dBFS. */
    params->high_channel.denoising_threshold = 1e-4f;
    params->high_channel.sensory_offset = 0.02f;  /* Roughly -35 dBFS. */
  }
}

/* Resets filter state variables. */
static void ChannelStateInit(Yuan2005ChannelState* channel) {
  BiquadFilterInitZero(&channel->band_biquad_state);
  BiquadFilterInitZero(&channel->envelope_biquad_state);
}

int Yuan2005Init(Yuan2005State* state, const Yuan2005Params* params) {
  if (state == NULL || params == NULL) {
    fprintf(stderr, "Yuan2005Init: Null argument.\n");
    return 0;
  }
  const float sample_rate_hz = params->sample_rate_hz;
  if (!(sample_rate_hz > 0.0f)) {
    fprintf(stderr, "Yuan2005Init: sample_rate_hz must be positive.\n");
  }

  /* Make the low channel. */
  Yuan2005ChannelState* low_channel = &state->low_channel;
  low_channel->channel_params = params->low_channel;
  ChannelStateInit(low_channel);

  /* Design 350Hz lowpass filter and 25Hz lowpass filter. */
  if (!DesignButterworthOrder2Lowpass(
          350.0f, sample_rate_hz, &low_channel->band_biquad_coeffs) ||
      !DesignButterworthOrder2Lowpass(
          25.0f, sample_rate_hz, &low_channel->envelope_biquad_coeffs)) {
    fprintf(stderr, "Yuan2005Init: Error designing filters.\n");
    return 0;
  }

  /* Initialize 50Hz oscillator. */
  PhasorRotatorInit(&low_channel->oscillator, 50.0f, sample_rate_hz);

  /* Make the high channel. */
  Yuan2005ChannelState* high_channel = &state->high_channel;
  high_channel->channel_params = params->high_channel;
  ChannelStateInit(high_channel);

  /* Design 3000Hz highpass filter and (another) 25Hz lowpass filter. */
  if (!DesignButterworthOrder2Highpass(
          3000.0f, sample_rate_hz, &high_channel->band_biquad_coeffs) ||
      !DesignButterworthOrder2Lowpass(
          25.0f, sample_rate_hz, &high_channel->envelope_biquad_coeffs)) {
    fprintf(stderr, "Yuan2005Init: Error designing filters.\n");
    return 0;
  }

  /* Initialize 250Hz oscillator. */
  PhasorRotatorInit(&high_channel->oscillator, 250.0f, sample_rate_hz);
  return 1;
}

/* Processes samples for one channel. */
static void OneChannelProcessSamples(Yuan2005ChannelState* channel,
                                     const float* input,
                                     int num_samples,
                                     float* output) {
  const Yuan2005ChannelParams channel_params = channel->channel_params;
  int i;
  for (i = 0; i < num_samples; ++i) {
    float sample = input[i];

    /* Apply band filter. */
    sample = BiquadFilterProcessOneSample(
        &channel->band_biquad_coeffs, &channel->band_biquad_state, sample);

    /* Full-wave rectification. */
    sample = fabs(sample);

    /* Apply smoothing filter to the amplitude envelope. */
    sample = BiquadFilterProcessOneSample(&channel->envelope_biquad_coeffs,
                                          &channel->envelope_biquad_state,
                                          sample);

    if (sample > channel_params.denoising_threshold) {
      /* Add the absolute detection threshold. */
      sample += channel_params.sensory_offset;
      /* Modulate on the sinusoidal carrier. */
      sample *= PhasorRotatorSin(&channel->oscillator);
    } else {
      sample = 0.0f;
    }

    /* Store output sample. */
    output[2 * i] = sample;

    /* Advance the phasor. */
    PhasorRotatorNext(&channel->oscillator);
  }
}

void Yuan2005ProcessSamples(Yuan2005State* state,
                            const float* input,
                            int num_samples,
                            float* output) {
  OneChannelProcessSamples(
      &state->low_channel, input, num_samples, output);
  OneChannelProcessSamples(
      &state->high_channel, input, num_samples, output + 1);
}

