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
// Simple app for puck.
//
// The app was compiled using Segger Studio. This app takes the analog
// microphone data and transmits it to the sleeve.

#include <math.h>
#include <stdio.h>
#include <string.h>

// nRF libraries.
#include "nrf_delay.h"                 // NOLINT(build/include)
#include "nrf_log.h"                   // NOLINT(build/include)
#include "nrf_log_ctrl.h"              // NOLINT(build/include)
#include "nrf_log_default_backends.h"  // NOLINT(build/include)

// Drivers.
#include "analog_external_mic.h"  // NOLINT(build/include)
#include "battery_monitor.h"      // NOLINT(build/include)
#include "bmi270.h"               // NOLINT(build/include)
#include "board_defs.h"           // NOLINT(build/include)
#include "look_up.h"              // NOLINT(build/include)
#include "pwm_puck.h"             // NOLINT(build/include)
#include "serial_puck_sleeve.h"   // NOLINT(build/include)
#include "two_wire.h"             // NOLINT(build/include)
#include "ui.h"                   // NOLINT(build/include)

using namespace audio_tactile;  // NOLINT(build/namespaces)

const uint32_t kLedPin = 19;

static volatile int new_mic_data = 0;

const int kMicSamples = 64;
static_assert(kMicSamples <= kTxDataSize / 2,
              "Mic data is larger than Tx buffer size");

static int16_t analog_mic_data[kMicSamples];

void touch_event() {
  nrf_gpio_pin_toggle(kLedPin);
  NRF_LOG_RAW_INFO("%d\n", PuckUi.GetEvent());
  NRF_LOG_FLUSH();
}

void adc_new_data() {
  ExternalAnalogMic.GetData(analog_mic_data);
  new_mic_data = 1;
}

// Needs to be present, since its called when new data arrives.
// Now we are not expecting any data back.
void on_new_serial_data() {}

void low_battery_warning() { nrf_gpio_pin_toggle(kLedPin); }

int main() {
  // Set the indicator led pin to output.
  nrf_gpio_cfg_output(kLedPin);

  // Initialize logging to console with j-link.
  log_init();

  NRF_LOG_RAW_INFO("== Puck Start ==\n");
  NRF_LOG_RAW_INFO("%d microphone samples \n", kMicSamples);
  NRF_LOG_FLUSH();

  // Initialize the button user interface.
  PuckUi.Initialize();
  PuckUi.OnUiEventListener(touch_event);

  // Initialize external analog microphone.
  ExternalAnalogMic.Initialize();
  ExternalAnalogMic.OnAdcDataReady(adc_new_data);

  // Initialize puck-sleeve serial communications.
  PuckSleeveSerialPort.InitializePuck();
  PuckSleeveSerialPort.OnSerialDataReceived(on_new_serial_data);

  // Initialize battery monitor.
  PuckBatteryMonitor.Initialize();
  PuckBatteryMonitor.OnLowBatteryEventListener(low_battery_warning);

  // Infinite loop checks for new microphone data and sends it to the sleeve.
  while (1) {
    if (new_mic_data) {
      PuckSleeveSerialPort.SendMicrophoneData(analog_mic_data, 64);
      new_mic_data = 0;
    }
  }
}

extern "C" {
void HardFault_Handler(void) {
  uint32_t *sp = (uint32_t *)__get_MSP();  // Get stack pointer
  uint32_t ia = sp[12];                    // Get instruction address from stack
  NRF_LOG_RAW_INFO("Hard Fault at address: 0x%08x\r\n", (unsigned int)ia);
  NRF_LOG_FLUSH();
  while (1) {
  }
}
}

static void log_init(void) {
  ret_code_t err_code = NRF_LOG_INIT(NULL);
  APP_ERROR_CHECK(err_code);
  NRF_LOG_DEFAULT_BACKENDS_INIT();
}
