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

#include "dsp/auto_gain_control.h"

#include <math.h>
#include <stdlib.h>

#include "dsp/fast_fun.h"

int AutoGainControlInit(AutoGainControlState* state,
                        float sample_rate_hz,
                        float time_constant_s,
                        float agc_strength,
                        float power_floor) {
  if (state == NULL ||
      !(sample_rate_hz > 0.0f) ||
      !(time_constant_s > 0.0f) ||
      !(0.0f <= agc_strength && agc_strength <= 1.0f) ||
      !(power_floor > 0.0f)) {
    return 0;
  }

  /* Warm up length is one time constant. */
  state->num_warm_up_samples = (int)(sample_rate_hz * time_constant_s + 0.5f);
  state->smoother_coeff =
      (float)(1 - exp(-1 / (time_constant_s * sample_rate_hz)));
  state->exponent = -0.5f * agc_strength;
  state->power_floor = power_floor;
  AutoGainControlReset(state);
  return 1;
}

void AutoGainControlReset(AutoGainControlState* state) {
  state->smoothed_power = 1.0f;
  state->warm_up_counter = 1;
}

void AutoGainControlWarmUpProcess(AutoGainControlState* state,
                                  float power_sample) {
  ++state->warm_up_counter;
  /* Accumulate power, smoothed_power = 1 + x[0]^2 + ... + x[n]^2. */
  state->smoothed_power += power_sample;

  if (state->warm_up_counter > state->num_warm_up_samples) {
    /* Warm up is done. Divide to get the average power,
     *   smoothed_power = (1 + x[0]^2 + ... + x[n]^2) / (1 + n).
     */
    state->smoothed_power /= state->warm_up_counter;
  }
}

float AutoGainControlGetGain(const AutoGainControlState* state) {
  float smoothed_power = (state->warm_up_counter <= state->num_warm_up_samples)
      ? state->smoothed_power / state->warm_up_counter
      : state->smoothed_power;
  /* Compute the gain = (power_floor + smoothed_power)^(-agc_strength / 2). */
  return FastPow(state->power_floor + smoothed_power, state->exponent);
}


