/* Copyright 2019, 2021 Google LLC
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

#include "extras/references/yuan2005/yuan2005.h"

#include <math.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

static float Clip(float x, float x_min, float x_max) {
  return (x < x_max) ? ((x > x_min) ? x : x_min) : x_max;
}

static float Taper(float t, float t_min, float t_max) {
  return Clip((t - t_min) / 0.005f, 0, 1) * Clip((t_max - t) / 0.005f, 0, 1);
}

/* Computes energy integral of `x` over [t_start, t_end]. */
static float ComputeEnergy(const float* x, float t_start, float t_end,
                           float sample_rate_hz) {
  double accum = 0.0;
  const int i_start = (int)(t_start * sample_rate_hz + 0.5f);
  const int i_end = (int)(t_end * sample_rate_hz + 0.5f);
  int i;
  for (i = i_start; i <= i_end; ++i) {
    const double sample = (double)x[2 * i];
    accum += sample * sample;
  }
  return accum / sample_rate_hz;
}

static void TestBasic(float sample_rate_hz) {
  printf("TestBasic(%g)\n", sample_rate_hz);
  srand(0);
  const int num_samples = (int)(0.75f * sample_rate_hz + 0.5f);
  float* input = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(
      malloc(2 * num_samples * sizeof(float)));

  int i;
  for (i = 0; i < num_samples; ++i) {
    float t = i / sample_rate_hz;
    /* Generate a small amount of noise, which thresholding should reject. */
    input[i] = 1e-5f * ((float) rand() / RAND_MAX - 0.5f);
    /* From 0.05 < t < 0.3, add a 100 Hz tone. */
    input[i] += 0.5 * sin(2.0 * M_PI * 100.0 * t) * Taper(t, 0.05f, 0.3f);
    /* From 0.45 < t < 0.7, add a 5000 Hz tone. */
    input[i] += 0.5 * sin(2.0 * M_PI * 5000.0 * t) * Taper(t, 0.45f, 0.7f);
  }

  Yuan2005Params params;
  Yuan2005SetDefaultParams(&params);
  params.sample_rate_hz = sample_rate_hz;
  Yuan2005State state;
  CHECK(Yuan2005Init(&state, &params));

  Yuan2005ProcessSamples(&state, input, num_samples, output);

  const float* low_output = output;
  const float* high_output = output + 1;

  /* For t < 0.05, all output is close to zero. */
  float low_energy = ComputeEnergy(low_output, 0.0f, 0.05f, sample_rate_hz);
  float high_energy = ComputeEnergy(high_output, 0.0f, 0.05f, sample_rate_hz);
  CHECK(low_energy < 1e-6f);
  CHECK(high_energy < 1e-6f);

  /* During the 100 Hz tone, low channel is strongest. */
  low_energy = ComputeEnergy(low_output, 0.07f, 0.28f, sample_rate_hz);
  high_energy = ComputeEnergy(high_output, 0.07f, 0.28f, sample_rate_hz);
  CHECK(low_energy > 0.01f);
  CHECK(low_energy > 10 * high_energy);

  /* During the 5000 Hz tone, vowel RMS is highest. */
  low_energy = ComputeEnergy(low_output, 0.47f, 0.68f, sample_rate_hz);
  high_energy = ComputeEnergy(high_output, 0.47f, 0.68f, sample_rate_hz);
  CHECK(high_energy > 0.01f);
  CHECK(high_energy > 10 * low_energy);

  free(output);
  free(input);
}

int main(int argc, char** argv) {
  TestBasic(16000.0f);
  TestBasic(44100.0f);
  TestBasic(48000.0f);

  puts("PASS");
  return EXIT_SUCCESS;
}
