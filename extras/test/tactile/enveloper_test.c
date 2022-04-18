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

#include "src/tactile/enveloper.h"

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
  int i_start = (int)(t_start * sample_rate_hz + 0.5f) * kEnveloperNumChannels;
  int i_end = (int)(t_end * sample_rate_hz + 0.5f) * kEnveloperNumChannels;
  int i;
  for (i = i_start; i <= i_end; i += kEnveloperNumChannels) {
    accum += (double)x[i] * x[i];
  }
  return accum / sample_rate_hz;
}

static void TestBasic(float sample_rate_hz, int decimation_factor) {
  printf("TestBasic(%g, %d)\n", sample_rate_hz, decimation_factor);
  srand(0);
  const float output_rate = sample_rate_hz / decimation_factor;
  const int output_frames = 0.5f * output_rate;
  const int num_samples = output_frames * decimation_factor;
  float* input = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(malloc(
      output_frames * kEnveloperNumChannels * sizeof(float)));

  int i;
  for (i = 0; i < num_samples; ++i) {
    float t = i / sample_rate_hz;
    /* Generate a small amount of noise, which PCEN should easily reject. */
    input[i] = 1e-5f * ((float) rand() / RAND_MAX - 0.5f);
    /* From 0.05 < t < 0.15, add a 80 Hz tone. */
    input[i] += 0.2 * sin(2.0 * M_PI * 80.0 * t) * Taper(t, 0.05f, 0.15f);
    /* From 0.2 < t < 0.3, add a 1500 Hz tone. */
    input[i] += 0.2 * sin(2.0 * M_PI * 1500.0 * t) * Taper(t, 0.2f, 0.3f);
    /* From 0.35 < t < 0.45, add a 5000 Hz tone. */
    input[i] += 0.15 * sin(2.0 * M_PI * 5000.0 * t) * Taper(t, 0.35f, 0.45f);
  }

  Enveloper enveloper;
  CHECK(EnveloperInit(&enveloper, &kDefaultEnveloperParams,
                      sample_rate_hz, decimation_factor));

  EnveloperProcessSamples(&enveloper, input, num_samples, output);

  const float* baseband = output;
  const float* vowel = output + 1;
  const float* fricative = output + 3;

  /* For t < 0.05, all output is close to zero. */
  float baseband_energy = ComputeEnergy(baseband, 0.0f, 0.05f, output_rate);
  float vowel_energy = ComputeEnergy(vowel, 0.0f, 0.05f, output_rate);
  float fricative_energy = ComputeEnergy(fricative, 0.0f, 0.05f, output_rate);
  CHECK(baseband_energy < 0.005f);
  CHECK(vowel_energy < 0.005f);
  CHECK(fricative_energy < 0.005f);

  /* During the 80 Hz tone, baseband channel is strongest. */
  baseband_energy = ComputeEnergy(baseband, 0.07f, 0.13f, output_rate);
  vowel_energy = ComputeEnergy(vowel, 0.07f, 0.13f, output_rate);
  fricative_energy = ComputeEnergy(fricative, 0.07f, 0.13f, output_rate);
  CHECK(baseband_energy > 0.001f);
  CHECK(baseband_energy > 2.0f * vowel_energy);
  CHECK(baseband_energy > 2.0f * fricative_energy);

  /* During the 1500 Hz tone, vowel RMS is highest. */
  baseband_energy = ComputeEnergy(baseband, 0.22f, 0.28f, output_rate);
  vowel_energy = ComputeEnergy(vowel, 0.22f, 0.28f, output_rate);
  fricative_energy = ComputeEnergy(fricative, 0.22f, 0.28f, output_rate);
  CHECK(vowel_energy > 0.001f);
  CHECK(vowel_energy > 2.0f * baseband_energy);
  CHECK(vowel_energy > 2.0f * fricative_energy);

  /* During the 5000 Hz tone, fricative RMS is highest. */
  baseband_energy = ComputeEnergy(baseband, 0.37f, 0.43f, output_rate);
  vowel_energy = ComputeEnergy(vowel, 0.37f, 0.43f, output_rate);
  fricative_energy = ComputeEnergy(fricative, 0.37f, 0.43f, output_rate);
  CHECK(fricative_energy > 0.001f);
  CHECK(fricative_energy > 2.0f * baseband_energy);
  CHECK(fricative_energy > 2.0f * vowel_energy);

  free(output);
  free(input);
}

/* Test that when input is pure silence (zero), Enveloper's output is zero. */
static void TestSilence(int decimation_factor) {
  printf("TestSilence(%d)\n", decimation_factor);
  const float sample_rate_hz = 16000.0f;
  const float output_rate = sample_rate_hz / decimation_factor;
  const int output_frames = 0.5f * output_rate;
  const int num_samples = output_frames * decimation_factor;
  float* input = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(malloc(
      output_frames * kEnveloperNumChannels * sizeof(float)));

  int i;
  for (i = 0; i < num_samples; ++i) {
    input[i] = 0.0f;
  }

  Enveloper enveloper;
  CHECK(EnveloperInit(&enveloper, &kDefaultEnveloperParams,
                      sample_rate_hz, decimation_factor));
  EnveloperProcessSamples(&enveloper, input, num_samples, output);

  for (i = 0; i < output_frames * kEnveloperNumChannels; ++i) {
    CHECK(output[i] == 0.0f);
  }

  free(output);
  free(input);
}

static void TestStreaming(int decimation_factor) {
  printf("TestStreaming(%d)\n", decimation_factor);
  srand(0);
  const int kChannels = kEnveloperNumChannels;
  const float sample_rate_hz = 16000.0f;
  const int output_frames = 100;
  const int input_size = output_frames * decimation_factor;
  float* input = (float*)CHECK_NOTNULL(malloc(input_size * sizeof(float)));
  float* nonstreaming_undecimated_output = (float*)CHECK_NOTNULL(malloc(
      input_size * kChannels * sizeof(float)));
  float* streaming_output = (float*)CHECK_NOTNULL(malloc(
      output_frames * kChannels * sizeof(float)));
  int i;
  for (i = 0; i < input_size; ++i) {
    input[i] = 0.2f * ((float) rand() / RAND_MAX - 0.5f);
  }

  Enveloper enveloper;
  CHECK(EnveloperInit(&enveloper, &kDefaultEnveloperParams, sample_rate_hz, 1));
  /* Process all the input at once without decimation. */
  EnveloperProcessSamples(&enveloper, input, input_size,
                          nonstreaming_undecimated_output);

  /* Reset and process in a streaming manner, possibly with decimation. */
  CHECK(EnveloperInit(&enveloper, &kDefaultEnveloperParams,
                      sample_rate_hz, decimation_factor));
  float* dest = streaming_output;
  int start = 0;
  while (start < input_size) {
    /* Process a block whose size is a random multiple of decimation_factor. */
    int input_block_size = decimation_factor * (1 + rand() / (RAND_MAX / 20));
    if (input_block_size > input_size - start) {
      input_block_size = input_size - start;
    }

    EnveloperProcessSamples(
        &enveloper, input + start, input_block_size, dest);

    start += input_block_size;
    dest += (input_block_size / decimation_factor) * kChannels;
  }

  /* Check that streaming and possibly decimated output matches nonstreaming
   * undecimated output in the sense of relative L1 error,
   *
   *   || actual - expected ||_1
   *   ------------------------- < 0.03.
   *       || expected ||_1
   */
  float l1_diff = 0.0f;
  float l1_expected = 0.0f;
  for (i = 0; i < output_frames; ++i) {
    int c;
    for (c = 0; c < kChannels; ++c) {
      const float actual = streaming_output[c + i * kChannels];
      const float expected = nonstreaming_undecimated_output[
          c + ((i + 1) * decimation_factor - 1) * kChannels];

      l1_diff += fabs(actual - expected);
      l1_expected += fabs(expected);
    }
  }
  CHECK(l1_diff / l1_expected < 0.03f);
  if (decimation_factor == 1) {
    /* If there is no decimation, then outputs should match very accurately. */
    CHECK(l1_diff < 1e-5f);
  }

  free(streaming_output);
  free(nonstreaming_undecimated_output);
  free(input);
}

int main(int argc, char** argv) {
  int decimation_factor;
  for (decimation_factor = 1; decimation_factor <= 4; decimation_factor *= 2) {
    TestBasic(16000.0f, decimation_factor);
    TestBasic(44100.0f, decimation_factor);
    TestBasic(48000.0f, decimation_factor);

    TestSilence(decimation_factor);
    TestStreaming(decimation_factor);
  }

  puts("PASS");
  return EXIT_SUCCESS;
}
