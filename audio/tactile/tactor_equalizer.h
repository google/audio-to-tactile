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
 *
 *
 * Filter to partially equalize the tactor's perceptual response.
 *
 * Vibrotactile perception has usable sensitivity for tactile interfaces in the
 * range 50-400Hz. It is about 20 dB more sensitive at 200 Hz, that is, the
 * perceived strength of a 50 Hz signal is similar to a 200 Hz signal that is 20
 * dB quieter.
 *
 * This library designs a filter to take this perceptual effect into account, to
 * partially equalize the response vs. frequency.
 */

#ifndef AUDIO_TACTILE_TACTOR_EQUALIZER_H_
#define AUDIO_TACTILE_TACTOR_EQUALIZER_H_

#include "audio/dsp/portable/biquad_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Designs an equalization filter for a given sample rate with the following
 * behavior:
 *  - frequencies below 10 Hz are attenuated,
 *  - roughly unit gain around 20-110 Hz,
 *  - gain of roughly `mid_gain` (as a linear amplitude) in 110-450 Hz,
 *  - gain of roughly `high_gain` (linear amplitude) above 450 Hz.
 *
 * The result is written as two biquads to `coeffs[0]` and `coeffs[1]`. Returns
 * 1 on success, 0 on failure.
 */
int DesignTactorEqualizer(float mid_gain, float high_gain,
    float sample_rate_hz, BiquadFilterCoeffs* coeffs);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_TACTOR_EQUALIZER_H_ */

