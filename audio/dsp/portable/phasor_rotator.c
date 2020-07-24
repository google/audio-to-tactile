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

#include "audio/dsp/portable/phasor_rotator.h"

#include <math.h>

#include "audio/dsp/portable/math_constants.h"

void PhasorRotatorInit(
    PhasorRotator* oscillator, float frequency_hz, float sample_rate_hz) {
  const float radians_per_sample = 2 * M_PI * frequency_hz / sample_rate_hz;
  oscillator->rotator[0] = cos(radians_per_sample);
  oscillator->rotator[1] = sin(radians_per_sample);
  oscillator->phasor[0] = 1.0f;
  oscillator->phasor[1] = 0.0f;
  oscillator->phasor_renormalize_counter = 0;
}

void PhasorRotatorRenormalize(PhasorRotator* oscillator) {
  /* NOTE: There are fast reciprocal sqrt instructions on x86 SSE (rsqrtss) and
   * ARM (vrsqrte). It would be nice to use those to compute the normalization.
   */
  const float normalization = 1.0f / sqrt(
      oscillator->phasor[0] * oscillator->phasor[0]
      + oscillator->phasor[1] * oscillator->phasor[1]);
  oscillator->phasor[0] *= normalization;
  oscillator->phasor[1] *= normalization;
  oscillator->phasor_renormalize_counter = 0;  /* Reset the counter. */
}

