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
// This Arduino example program shows how to use PWM on the puck
// Either a random sequence or sin wave from a lookup is
// outputted on the two PWM channels.

#include "look_up.h"
#include "pwm_puck.h"

using namespace audio_tactile;

static uint16_t random_array[kNumPwmValues * kNumChannels];
bool sin_output = false;
bool random_output = true;

void setup() {
  PuckTactors.UpdatePwm(sin_wave);
  PuckTactors.Initialize();
  PuckTactors.OnSequenceEnd(on_PWM_sequence_end);
}

void loop() {
  // Show what are we playing on pwm channel 0
  if (sin_output) {
    for (int i = 0; i < 64; ++i) {
      Serial.println(sinwave[i * 4]);
    }
  }
  if (random_output) {
    for (int i = 0; i < 64; ++i) {
      Serial.println(random_array[i * 4]);
    }
  }

  // Add a delay so output is readable.
  delay(30);
}

void on_PWM_sequence_end() {
  nrf_gpio_pin_toggle(19);

  // Load random generated sequence.
  if (random_output) {
    for (int i = 0; i < 64; ++i) {
      random_array[i * 4] = rand() % 500 + 1;
      random_array[i * 4 + 1] = rand() % 500 + 1;
    }
    PuckTactors.UpdatePwm(random_array);
  }

  // Load sinwave sequence.
  if (sin_output) {
    PuckTactors.UpdatePwm(sin_wave);
  }
}
