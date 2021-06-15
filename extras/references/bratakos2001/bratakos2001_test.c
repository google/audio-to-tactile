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

#include "extras/references/bratakos2001/bratakos2001.h"

#include <math.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

static float Clip(float x, float x_min, float x_max) {
  return (x < x_max) ? ((x > x_min) ? x : x_min) : x_max;
}

static float Taper(float t, float t_min, float t_max) {
  return Clip((t - t_min) / 0.005f, 0, 1) * Clip((t_max - t) / 0.005f, 0, 1);
}

/* Computes RMS of `x` over [t_start, t_end]. */
static float ComputeRms(const float* x, float t_start, float t_end,
                        float sample_rate_hz) {
  double accum = 0.0;
  const int i_start = (int)(t_start * sample_rate_hz + 0.5f);
  const int i_end = (int)(t_end * sample_rate_hz + 0.5f);
  int i;
  for (i = i_start; i <= i_end; ++i) {
    const double sample = (double)x[i];
    accum += sample * sample;
  }
  return sqrt(accum / (i_end - i_start + 1));
}

static void TestBasic(float sample_rate_hz) {
  printf("TestBasic(%g)\n", sample_rate_hz);
  srand(0);
  const int num_samples = (int)(0.25f * sample_rate_hz + 0.5f);
  float* input = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(
      malloc(2 * num_samples * sizeof(float)));

  int i;
  for (i = 0; i < num_samples; ++i) {
    float t = i / sample_rate_hz;
    /* Generate a small amount of noise, which thresholding should reject. */
    input[i] = 1e-5f * ((float) rand() / RAND_MAX - 0.5f);
    /* From 0.05 < t < 0.2, add a 600 Hz tone. */
    input[i] += 0.4 * sin(2.0 * M_PI * 600.0 * t) * Taper(t, 0.05f, 0.2f);
  }

  Bratakos2001State state;
  CHECK(Bratakos2001Init(&state, sample_rate_hz));

  Bratakos2001ProcessSamples(&state, input, num_samples, output);

  /* For t < 0.05, all output is close to zero. */
  float rms = ComputeRms(output, 0.0f, 0.05f, sample_rate_hz);
  CHECK(rms < 1e-6f);

  /* RMS is about 0.2 during amplitude-0.4 tone. */
  rms = ComputeRms(output, 0.07f, 0.2f, sample_rate_hz);
  fprintf(stderr, "rms %g\n", rms);
  CHECK(fabs(rms - 0.2f) < 0.05f);

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
