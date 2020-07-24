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

#include "audio/tactile/frontend/carl_frontend_design.h"

#include <math.h>

#include "audio/dsp/portable/complex.h"
#include "audio/dsp/portable/math_constants.h"

/* Auditory filter nominal Equivalent Rectangular Bandwidth (ERB) at
 * `center_frequency_hz`. Returns a bandwidth in Hz.
 * Reference: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
 */
static double AuditoryBandwidthHz(double center_frequency_hz) {
  /* The break frequency between constant-q and constant-bandwidth. */
  const double kBreakFrequencyHz = 228.883;
  /* Limiting ratio of center_frequency/bandwidth. */
  const double kEarQ = 9.2645;
  return (kBreakFrequencyHz + center_frequency_hz) / kEarQ;
}

double CarlFrontendNextAuditoryFrequency(
    double frequency_hz, double step_erbs) {
  return frequency_hz - step_erbs * AuditoryBandwidthHz(frequency_hz);
}

/* Designs quadratic for a Laplace domain complex pair, then discretizes to the
 * Z-plane using bilinear transform with `natural_frequency_hz` as the match
 * frequency. The quadratic coefficients are written to `out[i]`, i = 0, 1, 2.
 */
static void ZQuadraticFromNaturalFrequencyAndDamping(
    double natural_frequency_hz, double zeta, double k, double* out) {
  const double omega = 2 * M_PI * natural_frequency_hz;
  const double b0 = k * k / (omega * omega);
  const double b1 = k * 2 * zeta / omega;
  const double b2 = 1.0;
  out[0] = b0 + b1 + b2;
  out[1] = 2 * (b2 - b0);
  out[2] = b0 - b1 + b2;
}

void CarlFrontendDesignBiquad(double pole_frequency_hz, double sample_rate_hz,
                              CarlFrontendChannelData* channel_data) {
  const double kPoleZeta = 0.15;
  const double kZeroRatio = M_SQRT2;
  const double kZeroZeta = kPoleZeta / kZeroRatio;

  const double zero_frequency_hz = pole_frequency_hz * kZeroRatio;
  const double k = 2 * M_PI * pole_frequency_hz /
    tan(M_PI * pole_frequency_hz / sample_rate_hz);

  double z_numerator[3];
  ZQuadraticFromNaturalFrequencyAndDamping(
      zero_frequency_hz, kZeroZeta, k, z_numerator);
  double z_denominator[3];
  ZQuadraticFromNaturalFrequencyAndDamping(
      pole_frequency_hz, kPoleZeta, k, z_denominator);

  channel_data->biquad_b0 = (float)(z_numerator[0] / z_denominator[0]);
  channel_data->biquad_b1 = (float)(z_numerator[1] / z_denominator[0]);
  channel_data->biquad_b2 = (float)(z_numerator[2] / z_denominator[0]);
  channel_data->biquad_a1 = (float)(z_denominator[1] / z_denominator[0]);
  channel_data->biquad_a2 = (float)(z_denominator[2] / z_denominator[0]);
}

/* Computes frequency response of a single biquad filter. */
static ComplexDouble BiquadFrequencyResponse(
    const CarlFrontendChannelData* channel_data, ComplexDouble z) {
  /* numerator = (b0 * z + b1) * z + b2. */
  ComplexDouble numerator = ComplexDoubleMul(ComplexDoubleMake(
        channel_data->biquad_b0 * z.real + channel_data->biquad_b1,
        channel_data->biquad_b0 * z.imag), z);
  numerator.real += channel_data->biquad_b2;
  /* denominator = (z + a1) * z + a2. */
  ComplexDouble denominator = ComplexDoubleMul(
      ComplexDoubleMake(z.real + channel_data->biquad_a1, z.imag), z);
  denominator.real += channel_data->biquad_a2;
  return ComplexDoubleDiv(numerator, denominator);
}

/* Frequency response of channel `channel_index` in the CARL cascade. */
static ComplexDouble CascadeFrequencyResponse(
    const CarlFrontendChannelData* channel_data, int channel_index,
    double frequency_hz, double input_sample_rate_hz) {
  double theta = 2 * M_PI * frequency_hz / input_sample_rate_hz;
  ComplexDouble z = ComplexDoubleMake(cos(theta), sin(theta));

  ComplexDouble response = ComplexDoubleMake(1.0, 0.0);
  int c;
  for (c = 0; c <= channel_index; ++c) {
    if (channel_data[c].should_decimate) {
      theta *= 2.0;
      z = ComplexDoubleMake(cos(theta), sin(theta));
    }

    response = ComplexDoubleMul(response,  /* Factor for the next biquad. */
        BiquadFrequencyResponse(&channel_data[c], z));
  }

  response = ComplexDoubleMul(response,  /* Factor for the difference filter. */
      ComplexDoubleMake(1.0 - z.real, -z.imag));
  return response;
}

double CarlFrontendFindPeakGain(const CarlFrontendChannelData* channel_data,
                                int channel_index, double input_sample_rate_hz,
                                double* peak_frequency_hz) {
  double delta_f = 0.02 * *peak_frequency_hz;
  double max_gain = 0.0;

  while (fabs(delta_f) > 1e-5 * *peak_frequency_hz) {
    const double gain = ComplexDoubleAbs(
        CascadeFrequencyResponse(channel_data, channel_index,
          *peak_frequency_hz + delta_f, input_sample_rate_hz));
    if (gain > max_gain) {
      max_gain = gain;
      *peak_frequency_hz += delta_f;
    } else {
      delta_f *= -0.5;  /* Turn around and halve the search step. */
    }
  }

  return max_gain;
}
