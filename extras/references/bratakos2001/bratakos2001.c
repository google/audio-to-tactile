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

#include "extras/references/bratakos2001/bratakos2001.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/dsp/butterworth.h"
#include "src/dsp/math_constants.h"

int Bratakos2001Init(Bratakos2001State* state, float sample_rate_hz) {
  if (state == NULL) {
    fprintf(stderr, "Bratakos2001Init: Null argument.\n");
    return 0;
  }
  if (!(sample_rate_hz > 0.0f)) {
    fprintf(stderr, "Bratakos2001Init: sample_rate_hz must be positive.\n");
  }

  const float kLowEdgeHz = 500.0 / M_SQRT2;
  const float kHighEdgeHz = 500.0 * M_SQRT2;
  if (!DesignButterworthOrder2Bandpass(
          kLowEdgeHz, kHighEdgeHz, sample_rate_hz,
          state->bpf_biquad_coeffs) ||
      !DesignButterworthOrder2Lowpass(
          100.0f, sample_rate_hz, &state->envelope_biquad_coeffs)) {
    fprintf(stderr, "Bratakos2001Init: Error designing filters.\n");
    return 0;
  }

  /* Initialize 200Hz oscillator. */
  PhasorRotatorInit(&state->oscillator, 200.0f, sample_rate_hz);

  BiquadFilterInitZero(&state->bpf_biquad_state[0]);
  BiquadFilterInitZero(&state->bpf_biquad_state[1]);
  BiquadFilterInitZero(&state->envelope_biquad_state);
  return 1;
}

void Bratakos2001ProcessSamples(Bratakos2001State* state,
                                const float* input,
                                int num_samples,
                                float* output) {
  int i;
  for (i = 0; i < num_samples; ++i) {
    float sample = input[i];

    /* Bandpass filter. */
    sample = BiquadFilterProcessOneSample(
        &state->bpf_biquad_coeffs[0], &state->bpf_biquad_state[0], sample);
    sample = BiquadFilterProcessOneSample(
        &state->bpf_biquad_coeffs[1], &state->bpf_biquad_state[1], sample);

    /* Full-wave rectification. */
    sample = fabs(sample);

    /* Lowpass filter to smooth the amplitude envelope. */
    sample = BiquadFilterProcessOneSample(
        &state->envelope_biquad_coeffs, &state->envelope_biquad_state, sample);

    /* Modulate on the sinusoidal carrier. */
    sample *= PhasorRotatorSin(&state->oscillator);

    /* Store output sample. */
    output[i] = sample;

    /* Advance the phasor. */
    PhasorRotatorNext(&state->oscillator);
  }
}
