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
//
// App for streaming tactile data to the slim board.
// This app takes the data from the USB from the pc and sends it to
// actuators. Full 12-channel data is expected.

#include "board_defs.h"
#include "pwm_sleeve.h"

using namespace audio_tactile;

constexpr int kNumChannels = 12;
constexpr int kPwmSamples = 8;
constexpr int kDataBytes = 128;
constexpr int kHeaderBytes = 4;

uint8_t g_input_array[4096];
int g_serial_counter;
unsigned long g_time_prev;
uint8_t g_received_tactile_frame[kHeaderBytes + kDataBytes];
volatile bool g_sequence_finished = false;
volatile bool g_new_data = false;
static uint8_t g_which_pwm_module_triggered;
static int g_led_counter = 0;

void FlashLeds();
void OnPwmSequenceEnd();

void setup() {
  Serial.begin(1000000);
  nrf_gpio_cfg_output(kLedPinBlue);
  nrf_gpio_cfg_output(kLedPinGreen);

  // Initialize PWM.
  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  // Warning: issue only in Arduino. When using StartPlayback() it crashes.
  // Looks like NRF_PWM0 module is automatically triggered, and triggering it
  // again here crashes ISR. Temporary fix is to only use nrf_pwm_task_trigger
  // for NRF_PWM1 and NRF_PWM2. To fix might need a nRF52 driver update.
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);

  FlashLeds();
}

void loop() {
  if (Serial.available()) {
    while (Serial.available()) {
      // Get the new byte:
      uint8_t c = Serial.read();
      g_input_array[g_serial_counter] = c;
      g_serial_counter = g_serial_counter + 1;
    }

    g_new_data = true;

    // Get the timestamp to make sure there is no lag on USB port.
    // Should be around 4 ms for real-time performance.
    int time_stamp = (int)(g_time_prev - micros());
    Serial.println(time_stamp);
    memcpy(g_received_tactile_frame, g_input_array + kHeaderBytes,
           (kNumChannels * kPwmSamples));

    g_serial_counter = 0;
    g_time_prev = micros();
  }

  // Send the data to the sleeve, and request new packet.
  if (g_sequence_finished && g_new_data) {
    ++g_led_counter;  // Flash the LEDs sometimes.
    if (g_led_counter < 20) {
      nrf_gpio_pin_write(kLedPinBlue, 0);
    }
    else if (g_led_counter > 20 && g_led_counter < 40) {
      nrf_gpio_pin_write(kLedPinBlue, 1);
    }
    else if (g_led_counter > 60) {
      g_led_counter = 0;
    }

    g_sequence_finished = false;
    g_new_data = false;
    // By sending buffer_copied to PC, new frame is sent.
    Serial.println("buffer_copied");
  }
}

void OnPwmSequenceEnd() {
  g_which_pwm_module_triggered = SleeveTactors.GetEvent();

  if (g_which_pwm_module_triggered == 0) {
    SleeveTactors.UpdatePwmAllChannelsByte(g_received_tactile_frame);
    g_sequence_finished = true;
  }
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
