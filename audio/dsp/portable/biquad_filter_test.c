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

#include "audio/dsp/portable/biquad_filter.h"

#include <math.h>
#include <stdlib.h>

#include "audio/dsp/portable/logging.h"
#include "audio/dsp/portable/math_constants.h"

static double RandUniform() { return (double) rand() / RAND_MAX; }

static void ReferenceBiquadFilter(const BiquadFilterCoeffs* coeffs,
                                  const float* input,
                                  int num_samples,
                                  float* output) {
  int n;
  for (n = 0; n < num_samples; ++n) {
    output[n] = coeffs->b0 * input[n]
        + coeffs->b1 * ((n < 1) ? 0.0f : input[n - 1])
        + coeffs->b2 * ((n < 2) ? 0.0f : input[n - 2])
        - coeffs->a1 * ((n < 1) ? 0.0f : output[n - 1])
        - coeffs->a2 * ((n < 2) ? 0.0f : output[n - 2]);
  }
}

/* Biquad filter reproduces expected impulse response for a specific filter. */
void TestImpulseResponse() {
  const float a = 0.95;
  const float omega = (2 * M_PI) / 7;  /* Period of 7 samples. */
  const float cos_omega = cos(omega);

  BiquadFilterCoeffs coeffs;
  coeffs.b0 = 1.0f;
  coeffs.b1 = -a * cos_omega;
  coeffs.b2 = 0.0f;
  coeffs.a1 = -2 * a * cos_omega;
  coeffs.a2 = a * a;
  BiquadFilterState state;
  BiquadFilterInitZero(&state);

  int n;
  for (n = 0; n < 20; ++n) {
    /* The input signal is a unit impulse delayed by 3 samples. */
    float input_sample = (n == 3) ? 1.0f : 0.0f;
    float output_sample = BiquadFilterProcessOneSample(
        &coeffs, &state, input_sample);

    /* The expected impulse response of the filter is a^n cos(omega n). */
    float expected = (n < 3) ? 0.0f : pow(a, n - 3) * cos(omega * (n - 3));
    CHECK(fabs(output_sample - expected) < 5e-6f);
  }
}

/* Biquad filter matches reference implementation. */
void TestCompareWithReference() {
  const int kNumSamples = 20;
  float* input = (float*)CHECK_NOTNULL(malloc(kNumSamples * sizeof(float)));
  float* expected = (float*)CHECK_NOTNULL(malloc(kNumSamples * sizeof(float)));

  int trial;
  for (trial = 0; trial < 10; ++trial) {
    /* Generate a random stable filter. */
    BiquadFilterCoeffs coeffs;
    coeffs.b0 = 2 * RandUniform() - 1;
    coeffs.b1 = 2 * RandUniform() - 1;
    coeffs.b2 = 2 * RandUniform() - 1;
    const float pole_mag = 0.999 * RandUniform();
    const float pole_arg = M_PI * RandUniform();
    coeffs.a1 = 2 * pole_mag * cos(pole_arg);
    coeffs.a2 = pole_mag * pole_mag;

    int n;
    for (n = 0; n < kNumSamples; ++n) {
      input[n] = 2 * RandUniform() - 1;
    }
    ReferenceBiquadFilter(&coeffs, input, kNumSamples, expected);

    BiquadFilterState state;
    BiquadFilterInitZero(&state);

    for (n = 0; n < kNumSamples; ++n) {
      float output_sample = BiquadFilterProcessOneSample(
          &coeffs, &state, input[n]);

      CHECK(fabs(output_sample - expected[n]) < 5e-6f);
    }
  }

  free(expected);
  free(input);
}

/* Test that applying identity filter is identity. */
void TestIdentityFilter() {
  const BiquadFilterCoeffs coeffs = kBiquadFilterIdentityCoeffs;
  CHECK(coeffs.b0 == 1.0f);
  CHECK(coeffs.b1 == 0.0f);
  CHECK(coeffs.b2 == 0.0f);
  CHECK(coeffs.a1 == 0.0f);
  CHECK(coeffs.a2 == 0.0f);

  BiquadFilterState state;
  BiquadFilterInitZero(&state);

  int n;
  for (n = 0; n < 20; ++n) {
    /* The input signal is noise. */
    float input_sample = 2 * RandUniform() - 1;
    float output_sample = BiquadFilterProcessOneSample(
        &coeffs, &state, input_sample);

    CHECK(fabs(output_sample - input_sample) < 5e-6f);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestImpulseResponse();
  TestCompareWithReference();
  TestIdentityFilter();

  puts("PASS");
  return EXIT_SUCCESS;
}
