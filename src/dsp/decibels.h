/* Copyright 2022 Google LLC
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
 *
 *
 * Functions for converting to and from decibels.
 *
 * NOTE: Lighter functions below are marked `static` [the C analogy for
 * `inline`] so that ideally they get inlined.
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_DECIBELS_H_
#define AUDIO_TO_TACTILE_SRC_DSP_DECIBELS_H_

#include "dsp/fast_fun.h"
#include "dsp/math_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Converts power ratio to decibels. */
double PowerRatioToDecibels(double linear);

/* Converts amplitude ratio to decibels. */
double AmplitudeRatioToDecibels(double linear);

/* Converts decibels to power ratio. */
double DecibelsToPowerRatio(double decibels);

/* Converts decibels to amplitude ratio. */
double DecibelsToAmplitudeRatio(double decibels);

/* The following are fast approximations of the above conversions, using
 * FastLog2() and FastExp2() from fast_fun.
 */

/* Fast power ratio to decibels. Max absolute error is about 0.01 dB.
 * NOTE: `linear` must be positive and finite.
 */
static float FastPowerRatioToDecibels(float linear) {
  /* Through logarithm algebra,
   *
   *   10 * log10(linear)
   *   = (10 / log2(10)) * log2(linear)
   *   = (10 * log(2) / log(10)) * log2(linear).
   *
   * Implementation detail: M_LN2 and M_LN10 are double constants. The factor
   * `(float)((10.0 * M_LN2) / M_LN10)` is evaluated in double precision then
   * converted to float at compile time. At run time, it is simply a single
   * float constant.
   */
  return (float)((10.0 * M_LN2) / M_LN10) * FastLog2(linear);
}

/* Fast amplitude ratio to decibels. Max absolute error is about 0.02 dB.
 * NOTE: `linear` must be positive and finite.
 */
static float FastAmplitudeRatioToDecibels(float linear) {
  /* Similar to FastPowerToDecibels(), calculate
   *
   *   20 * log10(amplitude) = (20 * log(2) / log(10)) * log2(amplitude).
   */
  return (float)((20.0 * M_LN2) / M_LN10) * FastLog2(linear);
}

/* Fast decibels to power ratio. Max relative error is about 0.3%.
 * NOTE: Must have |decibels| <= 379 dB, otherwise result may be nonsensical.
 */
static float FastDecibelsToPowerRatio(float decibels) {
  /* Invert FastPowerRatioToDecibels(). */
  return FastExp2((float)(M_LN10 / (10.0 * M_LN2)) * decibels);
}

/* Fast decibels to amplitude ratio. Max relative error is about 0.3%.
 * NOTE: Must have |decibels| <= 758 dB, otherwise result may be nonsensical.
 */
static float FastDecibelsToAmplitudeRatio(float decibels) {
  /* Invert FastAmplitudeRatioToDecibels(). */
  return FastExp2((float)(M_LN10 / (20.0 * M_LN2)) * decibels);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_SRC_DSP_DECIBELS_H_ */
