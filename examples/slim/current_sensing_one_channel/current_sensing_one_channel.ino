// Copyright 2021 Google LLC
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
// This is an example Arduino app for slim board that demonstrates current
// sensing on one channel.
//
// Open the serial plotter (in Arduino IDE). Press 'a' to turn on
// the actuator and start sensing and 's' to turn off the actuator
// (Mnemonic: 'a' = activate, 's' = silence).
// Currently sensing is done on actuator 4, but could be moved to any of 12
// channels by  changing 'kSensingChannel' to values from 0 to 11.

#include "max14661.h"
#include "pwm_sleeve.h"

using namespace audio_tactile;

void EnableCurrentAmp();
void DisableCurrentAmp();
void OnPwmSequenceEnd();
void CreateSin(int scale);
void FlashLeds();

static uint16_t g_sin_wave_pattern[8];
static uint16_t g_silence[8];  // all zeros for no vibration output.

constexpr float kSampleRateHz = 43215.0f;
constexpr float kAttackTimeConstS = 0.0001f;
constexpr float kDecayTimeConstS = 1.5f;
constexpr int kSensingChannel = 4;  // Index-0 channel numbering.
constexpr int kAdcBufferSize = 1000;

float g_peak_value;

bool g_sampling_on = false;

void setup() {
  CreateSin(1);
  nrf_gpio_cfg_output(kCurrentAmpEnable);
  nrf_gpio_cfg_output(kLedPinBlue);
  nrf_gpio_cfg_output(kLedPinGreen);
  EnableCurrentAmp();
  Multiplexer.Initialize();

  FlashLeds();
  Multiplexer.ConnectChannel(kSensingChannel);
  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  // SamplingFactor could be modified to tune the sensing by
  // SleeveTactors.SetUpsamplingFactor(10);
  // Increasing the sampling factor, decreases the oscillation frequency.

  // Warning: issue only in Arduino. When using StartPlayback() it crashes.
  // Looks like NRF_PWM0 module is automatically triggered, and triggering it
  // again here crashes ISR. Temporary fix is to only use nrf_pwm_task_trigger
  // for NRF_PWM1 and NRF_PWM2. To fix might need a nRF52 driver update.
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);

  FlashLeds();

  while (!Serial) {
    delay(100);
  }  // Wait for serial to open.
  // Print instructions.
  Serial.println(
      "Press 'a' to turn on the actuator and start sensing and 's' to turn off "
      "the actuator");
  delay(1000);
}

void loop() {
  for (int i = 0; i < kAdcBufferSize; ++i) {
    const float adc_value = analogRead(
        A6);  // Sampling rate comes out to be 43215 Hz (23.14 us period)

    float attack_coeff =
        1.0f - exp(-1.0f / (kAttackTimeConstS * kSampleRateHz));
    float decay_coeff = 1.0f - exp(-1.0f / (kDecayTimeConstS * kSampleRateHz));

    const float coeff =
        (adc_value >= g_peak_value) ? attack_coeff : decay_coeff;
    g_peak_value += coeff * (adc_value - g_peak_value);
  }

  // Keep the peak at 460 (roughly at the middle), when sensing is not on.
  // Otherwise, it will go to zero, and filter will be slow to react.
  if (!g_sampling_on) {
    g_peak_value = 460;
  }
  Serial.println(g_peak_value);

  // Disable and enable the actuator based on the keyboard presses.
  while (Serial.available() > 0) {
    int inByte = Serial.read();
    if (inByte == 97) {  // Key "a"
      SleeveTactors.UpdateChannel(kSensingChannel, g_sin_wave_pattern);
      g_sampling_on = true;
    }
    if (inByte == 115) {  // Key "s"
      SleeveTactors.UpdateChannel(kSensingChannel, g_silence);
      g_sampling_on = false;
    }
  }
}

void EnableCurrentAmp() { nrf_gpio_pin_write(kCurrentAmpEnable, 0); }

void DisableCurrentAmp() { nrf_gpio_pin_write(kCurrentAmpEnable, 1); }

void CreateSin(int scale) {
  // Create a sin wav pattern to play, which is repeated every 4 ms.
  // Can be changed to any pattern here, with resolution of 0 to 512, where 512
  // is the loudest. Also, scale the output, so its not too loud.
  g_sin_wave_pattern[0] = 256 / scale;
  g_sin_wave_pattern[1] = 437 / scale;
  g_sin_wave_pattern[2] = 512 / scale;
  g_sin_wave_pattern[3] = 437 / scale;
  g_sin_wave_pattern[4] = 256 / scale;
  g_sin_wave_pattern[5] = 75 / scale;
  g_sin_wave_pattern[6] = 0 / scale;
  g_sin_wave_pattern[7] = 75 / scale;
}

void OnPwmSequenceEnd() {
  // Don't need to do anything here, just keep playing the sequence.
}

void FlashLeds() {
  for (int i = 0; i < 5; ++i) {
    nrf_gpio_pin_write(kLedPinBlue, 1);
    nrf_gpio_pin_write(kLedPinGreen, 1);
    delay(100);
    nrf_gpio_pin_write(kLedPinBlue, 0);
    nrf_gpio_pin_write(kLedPinGreen, 0);
    delay(100);
  }
}
