/* Copyright 2021 Google LLC
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

#include "tactile/tuning.h"

#include "dsp/fast_fun.h"

const TuningKnobs kDefaultTuningKnobs = {{
  /* kKnobInputGain      */ 127,
  /* kKnobOutputGain     */ 191,
  /* kKnobDenoising0     */  99,
  /* kKnobDenoising1     */  77,
  /* kKnobDenoising2     */  77,
  /* kKnobDenoising3     */  77,
  /* kKnobAgcStrength    */ 191,
  /* kKnobNoiseTa        */ 127,
  /* kKnobGainTauRelease */  73,
  /* kKnobCompresso      */  96,
}};

/* Maps `value_in` in [0, 255] linearly in [min_out, max_out]. */
static float LinMapping(int value_in, float min_out, float max_out) {
  if (value_in < 0) { value_in = 0; }
  if (value_in > 255) { value_in = 255; }
  return min_out + ((max_out - min_out) / 255.0f) * value_in;
}

/* Maps `value_in` in [0, 255] logarithmically in [min_out, max_out]. */
static float LogMapping(int value_in, float min_out, float max_out) {
  return FastExp2(LinMapping(value_in, FastLog2(min_out), FastLog2(max_out)));
}

float TuningMapControlValue(int knob, int value) {
  switch (knob) {
    case kKnobInputGain:
      return LinMapping(value, -40.0f, 40.315); /* In units of dB. */
    case kKnobOutputGain:
      return LinMapping(value, -18.0f, 6.0f); /* In units of dB. */
    case kKnobDenoising0:
    case kKnobDenoising1:
    case kKnobDenoising2:
    case kKnobDenoising3:
      return LogMapping(value, 2.0f, 200.0f);
    case kKnobAgcStrength:
      return LinMapping(value, 0.1f, 0.9f);
    case kKnobNoiseTau:
    case kKnobGainTauRelease:
      return LogMapping(value, 0.04f, 4.0f); /* In units of seconds. */
    case kKnobCompressor:
      return LinMapping(value, 0.1f, 0.5f);
    default:
      return 0.0f;
  }
}

float TuningGetInputGain(const TuningKnobs* tuning) {
  const float input_gain_db =
      TuningMapControlValue(kKnobInputGain, tuning->values[kKnobInputGain]);
  /* Convert dB to linear amplitude ratio. */
  return FastExp2((float)(M_LN10 / (20.0 * M_LN2)) * input_gain_db);
}
