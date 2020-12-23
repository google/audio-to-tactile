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

#include "tactile/tactor_equalizer.h"

#include <stdio.h>
#include <stdlib.h>

#include "dsp/math_constants.h"

/* Performs the bilinear transform to a quadratic polynomial of s, producing a
 * quadratic polynomial of z.
 */
static void BilinearQuadratic(float K, const float* s_quad, float* z_quad) {
  const float c0 = s_quad[0] * K * K;
  const float c1 = s_quad[1] * K;
  const float c2 = s_quad[2];
  z_quad[0] = c0 + c1 + c2;
  z_quad[1] = 2 * (c2 - c0);
  z_quad[2] = c0 - c1 + c2;
}

/* Designs biquad filter that boosts or attenuates in the frequency band
 * [`left_edge_hz`, `right_edge_hz`] by `gain`.
 *
 * If `with_highpass` = 0, then the filter passes frequencies outside the band
 * approximately unchanged.
 *
 * If `with_highpass` is nonzero, frequencies below left_edge_hz are suppressed.
 */
static void MakeBandboost(
    float left_edge_hz, float right_edge_hz, float gain,
    int with_highpass, float sample_rate_hz, BiquadFilterCoeffs* coeffs) {
  const float K = 2 * sample_rate_hz;
  const float omega1 = 2 * M_PI * left_edge_hz;
  const float omega2 = 2 * M_PI * right_edge_hz;

  /* If `with_highpass` = 0, we make the biquad
   *
   *          s^2 + gain * (omega2 - omega1) * s + omega2 * omega1
   *   H(s) = ----------------------------------------------------.
   *             s^2 + (omega2 - omega1) * s + omega2 * omega1
   *
   * If `with_highpass` is nonzero, the transfer function is identical except
   * the numerator constant coefficient is replaced with zero.
   */
  float s_quad[3];
  s_quad[0] = 1.0f;
  s_quad[1] = omega2 - omega1;
  s_quad[2] = omega2 * omega1;

  float z_denominator[3];
  BilinearQuadratic(K, s_quad, z_denominator);

  s_quad[1] *= gain;
  if (with_highpass) { s_quad[2] = 0.0f; }
  float z_numerator[3];
  BilinearQuadratic(K, s_quad, z_numerator);

  coeffs->b0 = z_numerator[0] / z_denominator[0];
  coeffs->b1 = z_numerator[1] / z_denominator[0];
  coeffs->b2 = z_numerator[2] / z_denominator[0];
  coeffs->a1 = z_denominator[1] / z_denominator[0];
  coeffs->a2 = z_denominator[2] / z_denominator[0];
}

int DesignTactorEqualizer(float mid_gain, float high_gain,
    float sample_rate_hz, BiquadFilterCoeffs* coeffs) {
  if (coeffs == NULL) { return 0; }

  const float low_gain = 1.0 / high_gain;
  mid_gain /= high_gain;
  const float kLowLeftHz = 10.0f;
  const float kLowRightHz = 110.0f;
  const float kMidLeftHz = 110.0f;
  const float kMidRightHz = 450.0f;

  /* The equalizer is a cascade of two second-order "bandboost" filters. The
   * first has gain of approximately `low_gain` in the band 10-110 Hz,
   * attenuates frequencies below 10 Hz, and has approximate unit gain above 110
   * Hz. The second has a gain of approximately `mid_gain` in 110-450 Hz and
   * unit gain elsewhere.
   *
   * The two bandboost filter responses overlap nontrivially, particularly in
   * the mid band 110-450Hz. We compensate by computing the equalizer response
   * at 220 Hz [= sqrt(kMidLeftHz * kMidRightHz)] and solving (by Mathematica)
   * for an `adjusted_mid_gain` so that the response there is mid_gain.
   */
  const float kC0 = 1.0f
      + (kLowLeftHz * kLowLeftHz - 4 * kLowLeftHz * kLowRightHz
        + kLowRightHz * kLowRightHz) / (kMidLeftHz * kMidRightHz)
      + (kLowLeftHz * kLowLeftHz * kLowRightHz * kLowRightHz)
        / (kMidLeftHz * kMidLeftHz * kMidRightHz * kMidRightHz);
  const float kC1 = (kLowRightHz - kLowLeftHz) * (kLowRightHz - kLowLeftHz)
        / (kMidLeftHz * kMidRightHz);
  float adjusted_mid_gain = mid_gain
      * sqrt(kC0 / (1 + kC1 * low_gain * low_gain));

  MakeBandboost(kLowLeftHz, kLowRightHz, low_gain, /*with_highpass=*/1,
                sample_rate_hz, &coeffs[0]);
  MakeBandboost(kMidLeftHz, kMidRightHz, adjusted_mid_gain, /*with_highpass=*/0,
                sample_rate_hz, &coeffs[1]);

  coeffs[0].b0 *= high_gain; /* Normalize to unit gain in the low band. */
  coeffs[0].b1 *= high_gain;
  coeffs[0].b2 *= high_gain;
  return 1;
}
