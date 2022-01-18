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
// sensing on multiple channels. The channels are sampled sequentially.
//
#include "max14661.h"
#include "pwm_sleeve.h"

using namespace audio_tactile;

void EnableCurrentAmp();
void DisableCurrentAmp();
void OnPwmSequenceEnd();
void CreateSin(int scale);
void FlashLeds();
float PrintOutChannelAverage(int channel);
int SampleChannel();

static uint16_t g_sin_wave_pattern[8];
static uint16_t g_silence[8] = {0};  // all zeros for no vibration output.

constexpr float kSampleRateHz = 43215.0f;
constexpr float kAttackTimeConstS = 0.0001f;
constexpr float kDecayTimeConstS = 1.5f;
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
}

void loop() {
  // Sampling the actuators in the bracelet.
  // This sampling order is just based on physical layout on the bracelet,
  // starting from the tactor farthest from the electronics box.
  // Each channel sampling takes about 1.85 seconds.
  PrintOutChannelAverage(5);
  PrintOutChannelAverage(4);
  PrintOutChannelAverage(7);
  PrintOutChannelAverage(11);
  PrintOutChannelAverage(8);
  PrintOutChannelAverage(9);
  PrintOutChannelAverage(6);
  PrintOutChannelAverage(10);
}

float PrintOutChannelAverage(int channel) {
  float sum = 0;
  int sum_counter = 0;

  SleeveTactors.UpdateChannel(channel, g_sin_wave_pattern);

  Multiplexer.ConnectChannel(channel);

  for (int i = 0; i < 80; ++i) {
    float peak_temp = SampleChannel();
    // Use the last 20 samples for averaging, to give some time to settle.
    if (i > 60) {
      sum += peak_temp;
      ++sum_counter;
    }

  }

  SleeveTactors.UpdateChannel(channel, g_silence);

  Serial.print("S,");
  Serial.print(channel);
  Serial.print(",");
  Serial.println(sum / sum_counter);
  Serial.flush();

  return sum / sum_counter;
}

int SampleChannel() {
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
  return g_peak_value;
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
