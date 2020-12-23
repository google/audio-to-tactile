// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// A simple example of sleeve PWM driver.
// The driver will play sin wave on each tactor in consecutive fashion.

#include "look_up.h"
#include "pwm_sleeve.h"

using namespace audio_tactile;

const uint32_t kTactileDecimationFactor = 8;

static uint8_t which_pwm_module_triggered;

void on_PWM_sequence_end() {
  which_pwm_module_triggered = SleeveTactors.GetEvent();
}

void setup() {
  // Initialize tactor driver.
  SleeveTactors.Initialize();
  SleeveTactors.SetUpsamplingFactor(kTactileDecimationFactor);
  SleeveTactors.OnSequenceEnd(on_PWM_sequence_end);
  SleeveTactors.StartPlayback();
}

void loop() {
  // Play a sin wave on one tactor.
  for (int pwm_module = 0; pwm_module < 3; ++pwm_module) {
    for (int pwm_channel = 0; pwm_channel < 4; ++pwm_channel) {
      SleeveTactors.UpdatePwmModuleChannel(sin_wave_downsample, pwm_module,
                                           pwm_channel);
      delay(250);
      SleeveTactors.SilencePwmModuleChannel(pwm_module, pwm_channel);
    }
  }
}
