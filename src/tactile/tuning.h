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
 *
 *
 * Functions for tuning TactileProcessor's noise robustness vs. sensitivity.
 *
 * The idea is the tuning functions below may be called in response to button
 * presses on the puck or BLE web app to adjust noise robustness vs. sensitivity
 * while it is running. To simplify the calling code, the `value` argument is an
 * integer control value between 0 and 255, and this value is then mapped in a
 * way that makes sense for that parameter.
 */

#ifndef THIRD_PARTY_AUDIO_TO_TACTILE_SRC_TACTILE_TUNING_H_
#define THIRD_PARTY_AUDIO_TO_TACTILE_SRC_TACTILE_TUNING_H_

#include "dsp/fast_fun.h"
#include "tactile/tactile_processor.h"

/* Sets the `output_gain` parameter of all EnergyEvelope instances based on a
 * control value between 0 and 255. The control is such that value = 0
 * corresponds to -18 dB gain, value = 191 to +0 dB, and value = 255 to +6 dB.
 * Returns the output gain in units of dB.
 */
static float TuningSetOutputGain(TactileProcessor* processor, int value) {
  /* Range of output_gain in dB. */
  const float kMinGainDb = -18.0f;
  const float kMaxGainDb = 6.0f;

  if (value < 0) { value = 0; }
  if (value > 255) { value = 255; }
  /* Map `value` in [0, 255] to a gain in dB in [kMinGainDb, kMaxGainDb]. */
  float value_db = kMinGainDb + ((kMaxGainDb - kMinGainDb) / 255.0f) * value;

  /* Convert dB to linear amplitude ratio. */
  float output_gain = FastExp2((M_LN10 / (20.0f * M_LN2)) * value_db);

  int i;
  for (i = 0; i < 4; ++i) {  /* Update each EnergyEvelope instance. */
    processor->channel_states[i].output_gain = output_gain;
  }

  return value_db;
}

/* Sets the `pcen_delta` parameter of all EnergyEnvelope instances based on a
 * control value between 0 and 255. It maps value = 0 to 0.0001, value = 85 to
 * 0.001, value = 170 to 0.01, and value = 255 to 0.1. Returns the mapped value
 * of delta that was set.
 */
static float TuningSetDenoising(TactileProcessor* processor, int value) {
  /* Range of delta. */
  const float kMinDelta = 0.0001f;
  const float kMaxDelta = 0.1f;

  if (value < 0) { value = 0; }
  if (value > 255) { value = 255; }
  /* Map `value` in [0, 255] to delta in [kMinDelta, kMaxDelta], warped
   * logarithmically so that delta changes in larger steps for larger `value`.
   */
  const float kLog2Min = log(kMinDelta) / M_LN2;
  const float kLog2Max = log(kMaxDelta) / M_LN2;
  float delta = FastExp2(kLog2Min + ((kLog2Max - kLog2Min) / 255.0f) * value);

  int i;
  for (i = 0; i < 4; ++i) {  /* Update each EnergyEvelope instance. */
    processor->channel_states[i].pcen_delta = delta;
    processor->channel_states[i].pcen_offset =
        FastPow(delta, processor->channel_states[i].pcen_beta);
  }

  return delta;
}

/* Sets the `pcen_beta` parameter of all EnergyEvelope instances based on a
 * control value between 0 and 100. It linearly maps values to beta in the range
 * [0.1, 0.5], with value = 96 corresponding to beta = 0.25. Returns the mapped
 * value of beta that was set.
 */
static float TuningSetCompression(TactileProcessor* processor, int value) {
  /* Range of beta. */
  const float kMinBeta = 0.1f;
  const float kMaxBeta = 0.5f;

  if (value < 0) { value = 0; }
  if (value > 255) { value = 255; }
  /* Map `value` in [0, 255] linearly to beta in [kMinBeta, kMaxBeta]. */
  float beta = kMinBeta + ((kMaxBeta - kMinBeta) / 255) * value;

  int i;
  for (i = 0; i < 4; ++i) {  /* Update each EnergyEvelope instance. */
    processor->channel_states[i].pcen_beta = beta;
    processor->channel_states[i].pcen_offset =
        FastPow(processor->channel_states[i].pcen_delta, beta);
  }

  return beta;
}

#endif  /* THIRD_PARTY_AUDIO_TO_TACTILE_SRC_TACTILE_TUNING_H_ */
