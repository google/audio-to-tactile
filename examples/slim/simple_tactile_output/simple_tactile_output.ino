// Copyright 2021-2022 Google LLC
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
// This Arduino example sketch shows how to use PWM on the slim board
// Plays a sin wave from a lookup on each channel.

#include "pwm_sleeve.h"

using namespace audio_tactile;
static uint16_t sin_wave[8];
static uint16_t silence[8] = {0};  // all zeros for no vibration output.

void OnPwmSequenceEnd();

void setup() {
  // Create a sin wav pattern to play, which is repeated every 4 ms.
  // Can be changed to any pattern here, with resolution of 0 to 512, where 512
  // is the loudest. Also, scale the output, so its not too loud.
  const int scale = 3;
  sin_wave[0] = 256 / scale;
  sin_wave[1] = 437 / scale;
  sin_wave[2] = 512 / scale;
  sin_wave[3] = 437 / scale;
  sin_wave[4] = 256 / scale;
  sin_wave[5] = 75 / scale;
  sin_wave[6] = 0 / scale;
  sin_wave[7] = 75 / scale;

  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  // Warning: issue only in Arduino. When using StartPlayback() it crashes.
  // Looks like NRF_PWM0 module is automatically triggered, and triggering it
  // again here crashes ISR. Temporary fix is to only use nrf_pwm_task_trigger
  // for NRF_PWM1 and NRF_PWM2. To fix might need a nRF52 driver update.
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);
}

void loop() {
  // Loop through all channels.
  for (int c = 0; c < 12; c++) {
    Serial.print("c = ");
    Serial.println(c);
    SleeveTactors.UpdateChannel(c, sin_wave);
    delay(150);
    SleeveTactors.UpdateChannel(c, silence);
    delay(150);
  }
}

// This function is triggered after sequence is finished playing.
void OnPwmSequenceEnd() {}
