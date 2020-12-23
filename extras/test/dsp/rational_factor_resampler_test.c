/* Copyright 2020 Google LLC
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

#include "src/dsp/rational_factor_resampler.h"

#include <math.h>
#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"
#include "src/dsp/phasor_rotator.h"
#include "src/dsp/rational_factor_resampler_kernel.h"

/* Tested sample rates in Hz. */
static const float kRates[] = {
    12000.0f, 16000.0f, 32000.0f, 44100.0f, 48000.0f, (float) (16000 * M_SQRT2),
};
static const int kNumRates = sizeof(kRates) / sizeof(*kRates);

/* Implement resampling directly according to
 *   x'[m] = x(m/F') = sum_n x[n] h(m F/F' - n),
 * where h is the resampling kernel, F is the input sample rate, and F' is the
 * output sample rate.
 */
static float* ReferenceResampling(const RationalFactorResamplerKernel* kernel,
                                  double rational_factor, const float* input,
                                  int num_channels, int num_input_frames,
                                  int* num_output_frames) {
  float* output = NULL;
  int output_capacity = 0; /* Output capacity counted as a number of frames. */

  int m;
  for (m = 0;; ++m) {
    const double n0 = m * rational_factor;
    /* Determine the range of n values for `sum_n x[n] h(m F/F' - n)`. */
    const int n_first = (int) floor(n0 - kernel->radius + 0.5);
    const int n_last = (int) floor(n0 + kernel->radius + 0.5);
    /* The kernel `h(m F/F' - n)` is zero outside of [n_first, n_last]. */
    CHECK(RationalFactorResamplerKernelEval(kernel, n0 - (n_first - 1)) == 0.0);
    CHECK(RationalFactorResamplerKernelEval(kernel, n0 - (n_last + 1)) == 0.0);

    if (n_last >= num_input_frames) { break; }

    if (m >= output_capacity) {
      output_capacity = (m == 0) ? 64 : 2 * (m + 1);
      output = (float*) CHECK_NOTNULL(realloc(
            output, sizeof(float) * output_capacity * num_channels));
    }

    int c;
    for (c = 0; c < num_channels; ++c) {
      /* Compute `sum_n x[n] h(m F/F' - n)`. */
      double sum = 0.0;
      int n;
      for (n = n_first; n <= n_last; ++n) {
        sum += (n < 0 ? 0.0 : input[n * num_channels + c])
            * RationalFactorResamplerKernelEval(kernel, n0 - n);
      }

      output[m * num_channels + c] = (float) sum;
    }
  }

  *num_output_frames = m;
  return output;
}

/* Compare with ReferenceResampling(). */
void TestCompareWithReferenceResampler(int num_channels,
                                       float filter_radius_factor) {
  printf("TestCompareWithReferenceResampler(%d, %g)\n", num_channels,
         filter_radius_factor);
  const int kInputFrames = 50;
  float* input =
      CHECK_NOTNULL(malloc(sizeof(float) * kInputFrames * num_channels));
  int n;
  for (n = 0; n < kInputFrames * num_channels; ++n) {
    input[n] = -0.5f + ((float)rand()) / RAND_MAX;
  }

  int i;
  for (i = 0; i < kNumRates; ++i) {
    int j;
    for (j = 0; j < kNumRates; ++j) {
      const float input_sample_rate_hz = kRates[i];
      const float output_sample_rate_hz = kRates[j];

      RationalFactorResamplerOptions options =
          kRationalFactorResamplerDefaultOptions;
      options.filter_radius_factor = filter_radius_factor;
      RationalFactorResampler* resampler =
          CHECK_NOTNULL(RationalFactorResamplerMake(
              input_sample_rate_hz, output_sample_rate_hz, num_channels,
              kInputFrames, &options));

      int num_output_frames =
          RationalFactorResamplerProcessSamples(resampler, input, kInputFrames);

      RationalFactorResamplerKernel kernel;
      CHECK(RationalFactorResamplerKernelInit(
          &kernel, input_sample_rate_hz, output_sample_rate_hz,
          options.filter_radius_factor, options.cutoff_proportion,
          options.kaiser_beta));
      int rational_factor_numerator;
      int rational_factor_denominator;
      RationalFactorResamplerGetRationalFactor(
          resampler, &rational_factor_numerator, &rational_factor_denominator);
      double rational_factor =
          ((double)rational_factor_numerator) / rational_factor_denominator;
      CHECK(fabs(rational_factor - kernel.factor) <= 5e-4);

      int expected_num_frames;
      float* expected =
          ReferenceResampling(&kernel, rational_factor, input, num_channels,
                              kInputFrames, &expected_num_frames);
      CHECK(abs(num_output_frames - expected_num_frames) <= 2);

      const float* output = RationalFactorResamplerOutput(resampler);
      int num_compare = num_channels * (num_output_frames < expected_num_frames
                                            ? num_output_frames
                                            : expected_num_frames);
      int n;
      for (n = 0; n < num_compare; ++n) {
        CHECK(fabs(output[n] - expected[n]) <= 5e-7f);
      }

      free(expected);
      RationalFactorResamplerFree(resampler);
    }
  }

  free(input);
}

/* Test streaming with blocks of random sizes between 0 and kMaxBlockFrames. */
void TestStreamingRandomBlockSizes(int num_channels) {
  printf("TestStreamingRandomBlockSizes(%d)\n", num_channels);
  const int kTotalInputFrames = 500;
  const int kMaxBlockFrames = 20;

  /* Generate random input samples. */
  float* input = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kTotalInputFrames * num_channels));
  int n;
  for (n = 0; n < kTotalInputFrames * num_channels; ++n) {
    input[n] = -0.5f + ((float) rand()) / RAND_MAX;
  }

  int i;
  for (i = 0; i < kNumRates; ++i) {
    int j;
    for (j = 0; j < kNumRates; ++j) {
      const float input_sample_rate_hz = kRates[i];
      const float output_sample_rate_hz = kRates[j];

      /* Do "nonstreaming" resampling, processing the whole input at once. */
      RationalFactorResampler* resampler = CHECK_NOTNULL(
          RationalFactorResamplerMake(input_sample_rate_hz,
                                      output_sample_rate_hz,
                                      num_channels,
                                      kTotalInputFrames,
                                      NULL));
      const int total_output_frames = RationalFactorResamplerProcessSamples(
          resampler, input, kTotalInputFrames);
      float* nonstreaming = CHECK_NOTNULL(malloc( /* Save the output. */
            sizeof(float) * total_output_frames * num_channels));
      memcpy(nonstreaming, RationalFactorResamplerOutput(resampler),
             sizeof(float) * total_output_frames * num_channels);
      RationalFactorResamplerFree(resampler);

      /* Do "streaming" resampling, passing successive blocks of input. */
      float* streaming = CHECK_NOTNULL(malloc(
            sizeof(float) * total_output_frames * num_channels));
      float* input_block = CHECK_NOTNULL(malloc(
            sizeof(float) * kMaxBlockFrames * num_channels));
      resampler = CHECK_NOTNULL(
          RationalFactorResamplerMake(input_sample_rate_hz,
                                      output_sample_rate_hz,
                                      num_channels,
                                      kMaxBlockFrames,
                                      NULL));
      CHECK(total_output_frames == RationalFactorResamplerNextNumOutputFrames(
                                       resampler, kTotalInputFrames));
      const int max_output_block_frames =
          RationalFactorResamplerMaxOutputFrames(resampler);
      int streaming_frames = 0;

      for (n = 0; n < kTotalInputFrames;) {
        /* Get the next block of input samples, having random size
         * 0 <= input_block_size <= kMaxBlockSize.
         */
        int input_block_frames = (int) floor(
            (((float) rand()) / RAND_MAX) * kMaxBlockFrames + 0.5);
        if (input_block_frames > kTotalInputFrames - n) {
          input_block_frames = kTotalInputFrames - n;
        }
        /* NOTE: We write the test this way for sake of demonstration and so
         * that potential buffer overruns are detectable with valgrind or asan.
         * More practically, we would read directly from input without copying.
         */
        memcpy(input_block, input + n * num_channels,
               sizeof(float) * input_block_frames * num_channels);
        n += input_block_frames;

        /* Resample the block. */
        const int expected_output_block_size =
            RationalFactorResamplerNextNumOutputFrames(resampler,
                                                       input_block_frames);
        const int output_block_frames = RationalFactorResamplerProcessSamples(
            resampler, input_block, input_block_frames);
        float* output_block = RationalFactorResamplerOutput(resampler);

        CHECK(output_block_frames <= max_output_block_frames);
        CHECK(expected_output_block_size == output_block_frames);

        /* Append output_block to the `streaming` array. */
        CHECK(streaming_frames + output_block_frames <= total_output_frames);
        memcpy(streaming + streaming_frames * num_channels, output_block,
               sizeof(float) * output_block_frames * num_channels);
        streaming_frames += output_block_frames;
      }
      RationalFactorResamplerFree(resampler);
      free(input_block);

      CHECK(n == kTotalInputFrames);
      CHECK(streaming_frames == total_output_frames);

      /* Streaming vs. nonstreaming outputs should match. */
      int m;
      for (m = 0; m < total_output_frames * num_channels; ++m) {
        CHECK(fabs(streaming[m] - nonstreaming[m]) < 1e-6f);
      }

      free(nonstreaming);
      free(streaming);
    }
  }

  free(input);
}

/* Resampling a sine wave should produce again a sine wave. */
void TestResampleSineWave() {
  puts("TestResampleSineWave");
  const float kFrequency = 1100.7f;

  const int kInputSize = 100;
  float* input = CHECK_NOTNULL(malloc(sizeof(float) * kInputSize));

  int i;
  for (i = 0; i < kNumRates; ++i) {
    const float input_sample_rate_hz = kRates[i];

    /* Generate sine wave. */
    PhasorRotator oscillator;
    PhasorRotatorInit(&oscillator, kFrequency, input_sample_rate_hz);
    int n;
    for (n = 0; n < kInputSize; ++n) {
      input[n] = PhasorRotatorSin(&oscillator);
      PhasorRotatorNext(&oscillator);
    }

    int j;
    for (j = 0; j < kNumRates; ++j) {
      const float output_sample_rate_hz = kRates[j];

      RationalFactorResampler* resampler = CHECK_NOTNULL(
          RationalFactorResamplerMake(input_sample_rate_hz,
                                      output_sample_rate_hz,
                                      /*num_channels=*/1,
                                      kInputSize,
                                      NULL));

      /* Run resampler on the sine wave samples. */
      const int output_size = RationalFactorResamplerProcessSamples(
          resampler, input, kInputSize);
      const float* output = RationalFactorResamplerOutput(resampler);

      RationalFactorResamplerOptions options =
          kRationalFactorResamplerDefaultOptions;
      RationalFactorResamplerKernel kernel;
      CHECK(RationalFactorResamplerKernelInit(
          &kernel, input_sample_rate_hz, output_sample_rate_hz,
          options.filter_radius_factor, options.cutoff_proportion,
          options.kaiser_beta));

      const double expected_size = (kInputSize - kernel.radius) / kernel.factor;
      CHECK(fabs(output_size - expected_size) <= 1.0);

      /* Compare output to sine wave generated at the output sample rate. */
      PhasorRotatorInit(&oscillator, kFrequency, output_sample_rate_hz);
      /* We ignore the first few output samples because they depend on input
       * samples at negative times, which are extrapolated as zeros.
       */
      const int num_to_ignore = 1 + (int) floor(kernel.radius / kernel.factor);
      int m;
      for (m = 0; m < output_size; ++m) {
        if (m >= num_to_ignore) {
          float expected = PhasorRotatorSin(&oscillator);
          CHECK(fabs(output[m] - expected) < 0.005f);
        }
        PhasorRotatorNext(&oscillator);
      }

      RationalFactorResamplerFree(resampler);
    }
  }

  free(input);
}

/* Test resampling a chirp. */
void TestResampleChirp() {
  puts("TestResampleChirp");

  int i;
  for (i = 0; i < kNumRates; ++i) {
    const float input_sample_rate_hz = kRates[i];
    const float kDurationSeconds = 0.025f;
    const int input_size = (int) (kDurationSeconds * input_sample_rate_hz);
    float* input = CHECK_NOTNULL(malloc(sizeof(float) * input_size));

    /* Generate chirp signal, sweeping linearly from 0 to max_frequency_hz. */
    const float max_frequency_hz = 0.45f * input_sample_rate_hz;
    const float chirp_slope = max_frequency_hz / kDurationSeconds;

    int n;
    for (n = 0; n < input_size; ++n) {
      const float t = n / input_sample_rate_hz;
      input[n] = (float) sin(M_PI * chirp_slope * t * t);
    }

    int j;
    for (j = 0; j < kNumRates; ++j) {
      const float output_sample_rate_hz = kRates[j];

      RationalFactorResampler* resampler = CHECK_NOTNULL(
          RationalFactorResamplerMake(input_sample_rate_hz,
                                      output_sample_rate_hz,
                                      /*num_channels=*/1,
                                      input_size,
                                      NULL));

      /* Run resampler on the chirp. */
      const int output_size = RationalFactorResamplerProcessSamples(
          resampler, input, input_size);
      const float* output = RationalFactorResamplerOutput(resampler);

      RationalFactorResamplerOptions options =
          kRationalFactorResamplerDefaultOptions;
      RationalFactorResamplerKernel kernel;
      CHECK(RationalFactorResamplerKernelInit(
          &kernel, input_sample_rate_hz, output_sample_rate_hz,
          options.filter_radius_factor, options.cutoff_proportion,
          options.kaiser_beta));
      /* Get kernel's cutoff frequency. */
      const float cutoff_hz = kernel.radians_per_sample
          * input_sample_rate_hz / (2 * M_PI);

      /* Compare output to chirp generated at the output sample rate. */
      int m;
      for (m = 0; m < output_size; ++m) {
        const float t = m / output_sample_rate_hz;
        /* Compute the chirp's instantaneous frequency at t. */
        const float chirp_frequency_hz = chirp_slope * t;

        /* Skip samples in the transition between passband and stopband. */
        if (fabs(chirp_frequency_hz - cutoff_hz) < 0.3f * cutoff_hz) {
          continue;
        }

        float expected;
        if (chirp_frequency_hz < cutoff_hz) {
          expected = (float) sin(M_PI * chirp_slope * t * t);
        } else {
          /* Expect output near zero when chirp frequency is above cutoff_hz. */
          expected = 0.0f;
        }
        CHECK(fabs(output[m] - expected) < 0.04f);
      }

      RationalFactorResamplerFree(resampler);
    }

    free(input);
  }
}

/* Test sample dropping behavior when input_size > max_input_size. */
void TestInputSizeExceedsMax(int num_channels) {
  printf("TestInputSizeExceedsMax(%d)\n", num_channels);
  const int kInputFrames = 120;
  const int kMaxInputFrames = 50;

  /* Generate random input samples. */
  float* input =
      CHECK_NOTNULL(malloc(sizeof(float) * kInputFrames * num_channels));
  int n;
  for (n = 0; n < kInputFrames * num_channels; ++n) {
    input[n] = -0.5f + ((float) rand()) / RAND_MAX;
  }

  int i;
  for (i = 0; i < kNumRates; ++i) {
    int j;
    for (j = 0; j < kNumRates; ++j) {
      const float input_sample_rate_hz = kRates[i];
      const float output_sample_rate_hz = kRates[j];

      RationalFactorResampler* resampler = CHECK_NOTNULL(
          RationalFactorResamplerMake(input_sample_rate_hz,
                                      output_sample_rate_hz,
                                      num_channels,
                                      kMaxInputFrames,
                                      NULL));

      /* Process the first 50 frames. */
      RationalFactorResamplerProcessSamples(resampler, input, 50);
      /* Process remaining 70 frames, exceeding kMaxInputFrames. The result will
       * be as if resampler had been reset and processed the last 50 frames.
       */
      const int num_output_frames = RationalFactorResamplerProcessSamples(
          resampler, input + 50 * num_channels, 70);

      float* output_with_drop = CHECK_NOTNULL(malloc( /* Save the output. */
            sizeof(float) * num_output_frames * num_channels));
      memcpy(output_with_drop, RationalFactorResamplerOutput(resampler),
             sizeof(float) * num_output_frames * num_channels);

      /* Compare with explicitly resetting and processing the last 50 frames. */
      RationalFactorResamplerReset(resampler);
      const int expected_size = RationalFactorResamplerProcessSamples(
          resampler, input + 70 * num_channels, 50);
      const float* expected = RationalFactorResamplerOutput(resampler);

      CHECK(num_output_frames == expected_size);
      int m;
      for (m = 0; m < num_output_frames * num_channels; ++m) {
        CHECK(fabs(output_with_drop[m] - expected[m]) < 1e-6f);
      }

      free(output_with_drop);
      RationalFactorResamplerFree(resampler);
    }
  }

  free(input);
}

int main(int argc, char** argv) {
  srand(0);

  int num_channels;
  for (num_channels = 1; num_channels <= 4; ++num_channels) {
    TestCompareWithReferenceResampler(num_channels, 5.0f);
    TestStreamingRandomBlockSizes(num_channels);
    TestInputSizeExceedsMax(num_channels);
  }

  TestCompareWithReferenceResampler(1, 4.0f);
  TestCompareWithReferenceResampler(1, 17.0f);
  TestResampleSineWave();
  TestResampleChirp();

  puts("PASS");
  return EXIT_SUCCESS;
}
