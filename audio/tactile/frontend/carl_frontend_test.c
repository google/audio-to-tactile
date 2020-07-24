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

#include "audio/tactile/frontend/carl_frontend.h"

#include <math.h>

#include "audio/dsp/portable/complex.h"
#include "audio/dsp/portable/logging.h"
#include "audio/dsp/portable/math_constants.h"
#include "audio/tactile/frontend/carl_frontend_design.h"

static void ComputeQuadraticRoots(double c0, double c1, double c2,
                                  ComplexDouble* roots) {
  /* Reduce quadratic to monic form, z^2 + c1 z + c2. */
  c1 /= c0;
  c2 /= c0;
  /* Form the discriminant and take its complex sqrt. */
  const double d = c1 * c1 / 4 - c2;
  const ComplexDouble sqrt_d = ComplexDoubleSqrt(ComplexDoubleMake(d, 0.0));
  /* Compute roots = -c1/2 +/- sqrt_d. */
  roots[0] = ComplexDoubleAdd(ComplexDoubleMake(-c1 / 2, 0.0), sqrt_d);
  roots[1] = ComplexDoubleSub(ComplexDoubleMake(-c1 / 2, 0.0), sqrt_d);
}

static double GammaFilterFrequencyMagnitudeResponse(
    int order, double pole, double f, double sample_rate) {
  double denom = 1 + pole * pole - 2 * pole * cos(2 * M_PI * f / sample_rate);
  return pow(pow(1 - pole, 2) / denom, 0.5 * order);
}

/* Tests CarlFrontend filter design. */
void TestDesign() {
  CarlFrontendParams params = kCarlFrontendDefaultParams;
  CarlFrontend* frontend = CHECK_NOTNULL(CarlFrontendMake(&params));

  const int num_channels = CarlFrontendNumChannels(frontend);
  /* For the default parameters, 56 channels are expected. */
  CHECK(num_channels == 56);

  const double output_sample_rate_hz =
      params.input_sample_rate_hz / params.block_size;
  double sample_rate_hz = params.input_sample_rate_hz;

  double pole = params.highest_pole_frequency_hz;
  int c;
  for (c = 0; c < num_channels; ++c) {
    const CarlFrontendChannelData* channel_data = &frontend->channel_data[c];
    if (channel_data->should_decimate) {
      sample_rate_hz /= 2;
    }
    CHECK(sample_rate_hz >= output_sample_rate_hz);

    ComplexDouble poles[2];
    ComputeQuadraticRoots(1.0, channel_data->biquad_a1, channel_data->biquad_a2,
                          poles);
    ComplexDouble zeros[2];
    ComputeQuadraticRoots(channel_data->biquad_b0, channel_data->biquad_b1,
                          channel_data->biquad_b2, zeros);

    CHECK(ComplexDoubleAbs(poles[0]) < 1.0);  /* Stable poles. */
    CHECK(ComplexDoubleAbs(poles[1]) < 1.0);
    CHECK(fabs(poles[0].imag) > 1e-2);  /* Poles are complex. */
    CHECK(fabs(poles[1].imag) > 1e-2);

    /* Zeros inside unit circle, too. */
    CHECK(ComplexDoubleAbs(zeros[0]) < 1.0);
    CHECK(ComplexDoubleAbs(zeros[1]) < 1.0);

    /* Check peak gain and frequency. */
    double peak_frequency_hz = pole;
    const double peak_gain = CarlFrontendFindPeakGain(
        frontend->channel_data, c, params.input_sample_rate_hz,
        &peak_frequency_hz);
    CHECK(fabs(peak_gain - 1.0) < 1e-6); /* Unit gain. */
    CHECK(fabs(pole - peak_frequency_hz) < pole / 10); /* Within 10% of pole. */

    /* Envelope smoother has -3dB gain at the cutoff frequency. */
    const double gain_at_cutoff = GammaFilterFrequencyMagnitudeResponse(
        2, 1.0 - channel_data->envelope_smoother_coeff,
        params.envelope_cutoff_hz, sample_rate_hz);
    CHECK(fabs(gain_at_cutoff - 1 / M_SQRT2) <= 1e-7);

    pole = CarlFrontendNextAuditoryFrequency(pole, params.step_erbs);
  }

  CarlFrontendFree(frontend);
}

static float FindMax(const float* values, int size, int* max_index) {
  *max_index = 0;
  float max_value = values[0];
  int i;
  for (i = 1; i < size; ++i) {
    if (values[i] > max_value) {
      max_value = values[i];
      *max_index = i;
    }
  }
  return max_value;
}

/* Tests response to sine wave inputs. */
void TestResponse() {
  CarlFrontendParams params = kCarlFrontendDefaultParams;
  CarlFrontend* frontend = CHECK_NOTNULL(CarlFrontendMake(&params));

  const float dt = 1.0f / params.input_sample_rate_hz;
  const int block_size = params.block_size;
  const int num_channels = CarlFrontendNumChannels(frontend);
  float* input = (float*)CHECK_NOTNULL(malloc(block_size * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(malloc(num_channels * sizeof(float)));

  float pole = params.highest_pole_frequency_hz;
  int c;
  for (c = 0; c < num_channels; ++c) {
    CarlFrontendReset(frontend);
    double peak_frequency_hz = pole;
    CarlFrontendFindPeakGain(frontend->channel_data, c,
                             params.input_sample_rate_hz, &peak_frequency_hz);

    /* For testing which channel has max response to the peak frequency, we
     * ignore the first 15ms, 15 blocks, of input to avoid confusion from
     * transient filtering and PCEN effects. This is enough for 1.5 cycles at
     * the default min pole frequency of 100 Hz.
     */
    const int kNumWarmupBlocks = 15;

    float t = 0.0f;
    int block;
    for (block = 0; block < kNumWarmupBlocks + 3; ++block) {
      int i;
      for (i = 0; i < block_size; ++i, t += dt) {
        /* Input is a sine wave with the cth peak frequency. */
        input[i] = 0.01 * sin(2 * M_PI * peak_frequency_hz * t);
      }

      /* Process a block of 16 input samples. */
      CarlFrontendProcessSamples(frontend, input, output);

      for (i = 0; i < num_channels; ++i) {
        CHECK(output[i] >= 0.0f); /* Output is nonnegative and not too large. */
        CHECK(output[i] <= 2.0f);
      }

      if (block >= kNumWarmupBlocks) {
        int max_index;
        float max_value = FindMax(output, num_channels, &max_index);

        /* Channel with largest response is close to c. */
        CHECK(abs(max_index - c) <= 1);

        /* Channel responses away from c are at least somewhat smaller. */
        for (i = 0; i < num_channels; ++i) {
          if (abs(i - c) > 8) {
            CHECK(output[i] < 0.75 * max_value);
          }
        }
      }
    }

    pole = CarlFrontendNextAuditoryFrequency(pole, params.step_erbs);
  }

  free(output);
  free(input);
  CarlFrontendFree(frontend);
}

/* Spot checks that invalid parameters are correctly rejected. */
void TestInvalidParameters() {
  { /* Invalid pole range, highest pole above input Nyquist. */
    CarlFrontendParams params = kCarlFrontendDefaultParams;
    params.input_sample_rate_hz = 8000.0f;
    params.highest_pole_frequency_hz = 6000.0f;
    CHECK(CarlFrontendMake(&params) == NULL);
  }
  { /* Nonsensically small step_erbs (would result in 28 million channels). */
    CarlFrontendParams params = kCarlFrontendDefaultParams;
    params.step_erbs = 1e-6f;
    CHECK(CarlFrontendMake(&params) == NULL);
  }
  { /* Non-power of 2 block_size. */
    CarlFrontendParams params = kCarlFrontendDefaultParams;
    params.block_size = 15;
    CHECK(CarlFrontendMake(&params) == NULL);
  }
  { /* Diffusivity too large for output sample rate. */
    CarlFrontendParams params = kCarlFrontendDefaultParams;
    params.input_sample_rate_hz = 16000.0f;
    params.block_size = 64;  /* => output sample rate is 250Hz. */
    params.pcen_cross_channel_diffusivity = 130.0f;
    CHECK(CarlFrontendMake(&params) == NULL);
  }
}

int main(int argc, char** argv) {
  TestDesign();
  TestResponse();
  TestInvalidParameters();

  puts("PASS");
  return EXIT_SUCCESS;
}
