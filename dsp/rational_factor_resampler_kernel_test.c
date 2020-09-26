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

#include "dsp/rational_factor_resampler_kernel.h"

#include <math.h>

#include "dsp/logging.h"
#include "dsp/math_constants.h"

/* Tested sample rates in Hz. */
static const float kRates[] = {
    12000.0f, 16000.0f, 32000.0f, 44100.0f, 48000.0f, (float)(16000 * M_SQRT2),
};
static const int kNumRates = sizeof(kRates) / sizeof(*kRates);

/* Test properties of ResamplerKernel over various sample rates and radii. */
void TestKernel(float filter_radius_factor) {
  printf("TestKernel(%g)\n", filter_radius_factor);
  const float kCutoff = 0.9f;
  const float kBeta = 5.658f;

  int i;
  for (i = 0; i < kNumRates; ++i) {
    int j;
    for (j = 0; j < kNumRates; ++j) {
      const float input_sample_rate_hz = kRates[i];
      const float output_sample_rate_hz = kRates[j];

      RationalFactorResamplerKernel kernel;
      CHECK(RationalFactorResamplerKernelInit(
          &kernel, input_sample_rate_hz, output_sample_rate_hz,
          filter_radius_factor, kCutoff, kBeta));
      CHECK(fabs(kernel.factor * output_sample_rate_hz -
                 input_sample_rate_hz) <= 0.005);

      /* Kernel is nonzero just within the radius since the Kaiser window has
       * a discontinuity at |x| = radius.
       */
      CHECK(fabs(RationalFactorResamplerKernelEval(
                &kernel, -kernel.radius)) > 1e-6);
      CHECK(fabs(RationalFactorResamplerKernelEval(
                &kernel, kernel.radius)) > 1e-6);
      /* The kernel is zero outside of [-radius, +radius]. */
      CHECK(RationalFactorResamplerKernelEval(
          &kernel, -kernel.radius - 1e-6) == 0.0);
      CHECK(RationalFactorResamplerKernelEval(
          &kernel, kernel.radius + 1e-6) == 0.0);

      const double value_0 = RationalFactorResamplerKernelEval(&kernel, 0.0);
      const double dx = 0.02;
      double sum = 0.0;
      double x;
      for (x = dx; x <= kernel.radius; x += dx) {
        double value_x = RationalFactorResamplerKernelEval(&kernel, x);
        CHECK(value_x < value_0); /* Kernel has a strict max at x = 0. */
        sum += value_x;
      }

      /* Check that the kernel integrates to 1.0 for unit DC gain. */
      CHECK(fabs(dx * (value_0 + 2 * sum) - 1.0) < 0.01);
    }
  }
}

int main(int argc, char** argv) {
  TestKernel(5.0f);
  TestKernel(17.0f);

  puts("PASS");
  return EXIT_SUCCESS;
}

