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

#include "dsp/auto_gain_control.h"

#include <math.h>
#include <stdlib.h>

#include "dsp/logging.h"

float ComputeRmsValue(const float* signal, int num_samples) {
  double accum = 0.0;
  int i;
  for (i = 0; i < num_samples; ++i) {
    accum += signal[i] * signal[i];
  }
  return (float)sqrt(accum / num_samples);
}

void TestBasic(float agc_strength) {
  const int kSampleRateHz = 16000;
  AutoGainControlState state;
  AutoGainControlInit(&state,
                      kSampleRateHz,
                      /*time_constant_s=*/0.25f,
                      agc_strength,
                      /*power_floor=*/1e-6f);

  const int num_samples = 2 * kSampleRateHz;
  float* signal = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));

  float amplitude;
  for (amplitude = 0.001f; amplitude < 1000.0f; amplitude *= 4) {
    int i;
    for (i = 0; i < num_samples; ++i) {
      signal[i] = amplitude * (rand() / (0.5f * RAND_MAX) - 1.0f);
    }
    float input_rms = ComputeRmsValue(signal, num_samples);

    /* Apply AGC to `signal`. */
    AutoGainControlReset(&state);
    for (i = 0; i < num_samples; ++i) {
      const float power_sample = signal[i] * signal[i];
      AutoGainControlProcessSample(&state, power_sample);
      signal[i] *= AutoGainControlGetGain(&state);
    }

    float output_rms = ComputeRmsValue(signal, num_samples);
    float expected = pow(input_rms, 1.0f - agc_strength);

    /* Within an order of magnitude of the expected RMS. */
    CHECK(output_rms > expected / 4);
    CHECK(output_rms < 4 * expected);
  }

  free(signal);
}

int main(int argc, char** argv) {
  srand(0);
  TestBasic(0.3f);
  TestBasic(0.5f);
  TestBasic(0.8f);

  puts("PASS");
  return EXIT_SUCCESS;
}

