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

#include "audio/tactile/tactor_equalizer.h"

#include <math.h>

#include "audio/dsp/portable/complex.h"
#include "audio/dsp/portable/logging.h"
#include "audio/dsp/portable/math_constants.h"
#include "audio/tactile/util.h"

/* Computes equalizer gain at `frequency_hz` in dB. */
static double Response(const BiquadFilterCoeffs* coeffs,
                       float frequency_hz, float sample_rate_hz) {
  double theta = 2 * M_PI * frequency_hz / sample_rate_hz;
  ComplexDouble z = ComplexDoubleMake(cos(theta), sin(theta));
  ComplexDouble z2 = ComplexDoubleSquare(z);

  return (10 / M_LN10) * log(
      (ComplexDoubleAbs2(ComplexDoubleMake(
            coeffs[0].b0 * z2.real + coeffs[0].b1 * z.real + coeffs[0].b2,
            coeffs[0].b0 * z2.imag + coeffs[0].b1 * z.imag))
       / ComplexDoubleAbs2(ComplexDoubleMake(
            z2.real + coeffs[0].a1 * z.real + coeffs[0].a2,
            z2.imag + coeffs[0].a1 * z.imag))) *
      (ComplexDoubleAbs2(ComplexDoubleMake(
            coeffs[1].b0 * z2.real + coeffs[1].b1 * z.real + coeffs[1].b2,
            coeffs[1].b0 * z2.imag + coeffs[1].b1 * z.imag))
       / ComplexDoubleAbs2(ComplexDoubleMake(
            z2.real + coeffs[1].a1 * z.real + coeffs[1].a2,
            z2.imag + coeffs[1].a1 * z.imag)))
      );
}

void TestResponse(float sample_rate_hz) {
  int mid_gain_db;
  for (mid_gain_db = -8; mid_gain_db <= 0; mid_gain_db += 2) {
    int high_gain_db;
    for (high_gain_db = -6; high_gain_db <= 0; high_gain_db += 2) {
      BiquadFilterCoeffs coeffs[2];
      CHECK(DesignTactorEqualizer(DecibelsToAmplitudeRatio(mid_gain_db),
                                  DecibelsToAmplitudeRatio(high_gain_db),
                                  sample_rate_hz, coeffs));
      /* Equalizer suppresses frequencies below 10 Hz. */
      CHECK(Response(coeffs, 1, sample_rate_hz) < -9.9);
      CHECK(Response(coeffs, 10, sample_rate_hz) < -2.5);
      /* Equalizer has about unit gain around 50 Hz. */
      CHECK(fabs(Response(coeffs, 30, sample_rate_hz)) < 1.5);
      CHECK(fabs(Response(coeffs, 70, sample_rate_hz)) < 1.5);
      /* Gain at 220 Hz agrees with `mid_gain_db`. This gain agrees precisely
       * since we account specifically at this point for bandboost overlap.
       */
      CHECK(fabs(Response(coeffs, 220, sample_rate_hz) - mid_gain_db) < 0.1);
      /* Gain at 800 Hz agrees with `high_gain_db`. This tolerance is looser,
       * since the mid bandboost disturbs it, and we don't correct for that.
       */
      CHECK(fabs(Response(coeffs, 800, sample_rate_hz) - high_gain_db) < 1.5);
    }
  }
}

int main(int argc, char** argv) {
  TestResponse(16000.0f);
  TestResponse(44100.0f);

  puts("PASS");
  return EXIT_SUCCESS;
}
