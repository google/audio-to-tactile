/* Copyright 2019, 2021-2022 Google LLC
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

#include "src/tactile/tactile_processor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"
#include "src/dsp/read_wav_file.h"

const int kBlockSize = 64;

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
    const double value = x[i * kTactileProcessorNumTactors];
    accum += value * value;
  }
  return accum / sample_rate_hz;
}

static float ComputeVowelEnergy(const float* x, float t_start, float t_end,
                                float sample_rate_hz) {
  double energy = 0.0;
  int c;
  for (c = 1; c <= 7; ++c) {
    energy += ComputeEnergy(x + c, t_start, t_end, sample_rate_hz);
  }
  return energy;
}

/* Runs TactileProcessor on a sequence of tones. */
static void TestTones(float sample_rate_hz, int decimation_factor) {
  printf("TestTones(%g, %d)\n", sample_rate_hz, decimation_factor);
  float output_rate = sample_rate_hz / decimation_factor;
  const int output_size = 0.5f * output_rate;
  const int input_size = output_size * decimation_factor;
  float* input = (float*)CHECK_NOTNULL(malloc(input_size * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(
      malloc(kTactileProcessorNumTactors * output_size * sizeof(float)));
  int i;
  for (i = 0; i < input_size; ++i) {
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

  const int num_tactors = kTactileProcessorNumTactors;
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  params.frontend_params.input_sample_rate_hz = sample_rate_hz;
  params.frontend_params.block_size = kBlockSize;
  params.decimation_factor = decimation_factor;
  TactileProcessor* tactile_processor = CHECK_NOTNULL(
      TactileProcessorMake(&params));
  int start;
  for (start = 0; start + kBlockSize <= input_size; start += kBlockSize) {
    TactileProcessorProcessSamples(
        tactile_processor, input + start,
        output + num_tactors * (start / decimation_factor));
  }
  TactileProcessorFree(tactile_processor);

  const float* baseband = output;
  const float* fricative = output + 9;

  CHECK(fabs(TactileProcessorOutputSampleRateHz(&params)
             - output_rate) < 1e-6f);

  /* For t < 0.05, all output is close to zero. */
  float baseband_energy = ComputeEnergy(baseband, 0.0f, 0.05f, output_rate);
  float vowel_energy = ComputeVowelEnergy(output, 0.0f, 0.05f, output_rate);
  float fricative_energy = ComputeEnergy(fricative, 0.0f, 0.05f, output_rate);
  CHECK(baseband_energy < 1e-6f);
  CHECK(vowel_energy < 1e-6f);
  CHECK(fricative_energy < 1e-6f);

  /* During the 80 Hz tone, baseband channel is strongest. */
  baseband_energy = ComputeEnergy(baseband, 0.07f, 0.13f, output_rate);
  vowel_energy = ComputeVowelEnergy(output, 0.07f, 0.13f, output_rate);
  fricative_energy = ComputeEnergy(fricative, 0.07f, 0.13f, output_rate);
  CHECK(baseband_energy > 1e-4f);
  CHECK(baseband_energy > 2.5f * vowel_energy);
  CHECK(baseband_energy > 2.5f * fricative_energy);

  /* During the 1500 Hz tone, vowel RMS is highest. */
  baseband_energy = ComputeEnergy(baseband, 0.22f, 0.28f, output_rate);
  vowel_energy = ComputeVowelEnergy(output, 0.22f, 0.28f, output_rate);
  fricative_energy = ComputeEnergy(fricative, 0.22f, 0.28f, output_rate);
  CHECK(vowel_energy > 1e-4f);
  CHECK(vowel_energy > 2.5f * baseband_energy);
  CHECK(vowel_energy > 2.5f * fricative_energy);

  /* During the 5000 Hz tone, fricative RMS is highest. */
  baseband_energy = ComputeEnergy(baseband, 0.37f, 0.43f, output_rate);
  vowel_energy = ComputeVowelEnergy(output, 0.37f, 0.43f, output_rate);
  fricative_energy = ComputeEnergy(fricative, 0.37f, 0.43f, output_rate);
  CHECK(fricative_energy > 1e-4f);
  CHECK(fricative_energy > 2.5f * baseband_energy);
  CHECK(fricative_energy > 2.5f * vowel_energy);

  free(output);
  free(input);
}

/* Tests TactileProcessorReset by comparing the results from processing an
 * input, resetting, and then processing the input a second time.
 */
static void TestReset(float sample_rate_hz, int decimation_factor) {
  printf("TestReset(%g, %d)\n", sample_rate_hz, decimation_factor);
  float output_rate = sample_rate_hz / decimation_factor;
  const int output_size = 0.2f * output_rate;
  const int input_size = output_size * decimation_factor;
  float* input = (float*)CHECK_NOTNULL(malloc(input_size * sizeof(float)));
  float* output1 = (float*)CHECK_NOTNULL(
      malloc(kTactileProcessorNumTactors * output_size * sizeof(float)));
  float* output2 = (float*)CHECK_NOTNULL(
      malloc(kTactileProcessorNumTactors * output_size * sizeof(float)));
  int i;
  for (i = 0; i < input_size; ++i) {
    float t = i / sample_rate_hz;
    input[i] = 0.2 * ((float) rand() / RAND_MAX - 0.5f)
        + 0.2 * sin(2.0 * M_PI * 1000.0 * t) * Taper(t, 0.05f, 0.15f);
  }

  const int num_tactors = kTactileProcessorNumTactors;
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  params.frontend_params.input_sample_rate_hz = sample_rate_hz;
  params.frontend_params.block_size = kBlockSize;
  params.decimation_factor = decimation_factor;
  TactileProcessor* tactile_processor = CHECK_NOTNULL(
      TactileProcessorMake(&params));

  /* Process the input. */
  int start;
  for (start = 0; start + kBlockSize <= input_size; start += kBlockSize) {
    TactileProcessorProcessSamples(
        tactile_processor, input + start,
        output1 + num_tactors * (start / decimation_factor));
  }

  /* Reset tactile_processor, then process the input a second time. */
  TactileProcessorReset(tactile_processor);
  for (start = 0; start + kBlockSize <= input_size; start += kBlockSize) {
    TactileProcessorProcessSamples(
        tactile_processor, input + start,
        output2 + num_tactors * (start / decimation_factor));
  }

  TactileProcessorFree(tactile_processor);

  /* Outputs should match. */
  const int num_output_samples = num_tactors * (start / decimation_factor);
  for (i = 0; i < num_output_samples; ++i) {
    CHECK(fabs(output1[i] - output2[i]) <= 1e-9f);
  }

  free(output2);
  free(output1);
  free(input);
}

/* Runs TactileProcessor on a short WAV recording of a pure phone, and
 * checks that the intended tactor is the most active.
 */
static void TestPhone(const char* phone, int intended_tactor) {
  printf("TestPhone(%s)\n", phone);
  char wav_file[1024];
  sprintf(wav_file,
          "extras/test/testdata/phone_%s.wav",
          phone);

  size_t num_samples;
  int num_channels;
  int sample_rate_hz;
  int16_t* input_int16 = (int16_t*)CHECK_NOTNULL(Read16BitWavFile(
      wav_file, &num_samples, &num_channels, &sample_rate_hz));
  CHECK(num_channels == 1);
  float* input = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kBlockSize));
  float* output = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kTactileProcessorNumTactors * kBlockSize));
  float* energy = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kTactileProcessorNumTactors));
  int c;
  for (c = 0; c < kTactileProcessorNumTactors; ++c) {
    energy[c] = 0.0f;
  }

  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  params.frontend_params.input_sample_rate_hz = sample_rate_hz;
  params.frontend_params.block_size = kBlockSize;
  TactileProcessor* tactile_processor = CHECK_NOTNULL(
      TactileProcessorMake(&params));

  int start;
  int i;
  for (start = 0; start + kBlockSize < (int)num_samples; start += kBlockSize) {
    for (i = 0; i < kBlockSize; ++i) {
      input[i] = input_int16[start + i] / 32768.0f;
    }

    TactileProcessorProcessSamples(tactile_processor, input, output);

    const float* tactile_signals = output;
    for (i = 0; i < kBlockSize; ++i) {
      for (c = 0; c < kTactileProcessorNumTactors; ++c) {
        /* Accumulate energy for each channel. */
        energy[c] += tactile_signals[c] * tactile_signals[c];
      }
      tactile_signals += kTactileProcessorNumTactors;
    }
  }

  /* The intended tactor has the largest energy in the vowel cluster. */
  for (c = 1; c <= 7; ++c) {
    if (c != intended_tactor) {
      CHECK(energy[intended_tactor] >= 1.65f * energy[c]);
    }
  }

  TactileProcessorFree(tactile_processor);
  free(energy);
  free(output);
  free(input);
  free(input_int16);
}

int main(int argc, char** argv) {
  srand(0);
  int decimation_factor;
  for (decimation_factor = 1; decimation_factor <= 4; decimation_factor *= 2) {
    TestTones(16000.0f, decimation_factor);
    TestTones(48000.0f, decimation_factor);
    TestReset(48000.0f, decimation_factor);
  }
  TestPhone("aa", 1);
  TestPhone("eh", 5);
  TestPhone("uw", 2);

  puts("PASS");
  return EXIT_SUCCESS;
}
