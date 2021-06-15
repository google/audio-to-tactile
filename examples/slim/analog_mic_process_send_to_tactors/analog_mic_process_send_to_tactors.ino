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
// This is an example Arduino app for slim board that demonstrates audio to
// tactile pipeline.  App gets the data from the analog microphone, sends it to
// the tactile processor. The processed data is converted to PWM and sent to the
// tactors.
//
// Pwm to output mapping, when using UpdatePwmModuleChannel():
// (pwm module, pwm channel) - output channel on the flex connector
// (0, 0) - 1
// (0, 1) - 2
// (0, 2) - 3
// (0, 3) - 4
// (1, 0) - 5
// (1, 1) - 6
// (1, 2) - 7
// (1, 3) - 8
// (2, 0) - 9
// (2, 1) - 10
// (2, 2) - 11
// (2, 3) - 12
//
// Physically on the bracelet the channels are ordered as follows:
// (6)---(5)---(8)---(12)---(9)---(10)---(7)---(11) << Box is 11.
// bb----aa----uw-----ih----iy-----eh----sh f--fric << Tactile processor mapping
//
// bb is baseband.
// Tactor 11 is in the electronics box.

#include "analog_external_mic.h"
#include "constants.h"
#include "post_processor_cpp.h"
#include "pwm_sleeve.h"
#include "serial_puck_sleeve.h"
#include "tactile_processor_cpp.h"
#include "dsp/channel_map.h"

using namespace audio_tactile;

// Compile time constants.
constexpr int kTactileDecimationFactor = 8;
constexpr int kSaadcSampleRateHz = 15625;
constexpr int kCarlBlockSize = 64;
constexpr int kPwmSamplesAllChannels = kNumPwmValues * kNumTotalPwm;
constexpr int kTactileFramesPerCarlBlock =
    kCarlBlockSize / kTactileDecimationFactor;

// Globals.
static uint8_t g_which_pwm_module_triggered;

static bool g_new_mic_data = false;
static bool g_tactor_processor_on = true;

// Buffer of input audio as floats in [-1, 1].
static float g_audio_input[kAdcDataSize];

static float* g_tactile_processor_output;

int16_t g_analog_mic_data[kAdcDataSize];

TactileProcessorWrapper g_tactile_processor;
PostProcessorWrapper g_post_processor;
// Channel to tactor mapping and final output gains.
ChannelMap g_channel_map;

void AdcNewData();
void FlashLeds();
void OnPwmSequenceEnd();
void OnNewSerialData();

void setup() {
  // Set the indicator led pin to output.
  nrf_gpio_cfg_output(kLedPinBlue);
  nrf_gpio_cfg_output(kLedPinGreen);

  // Initialize serial port.
  PuckSleeveSerialPort.OnSerialDataReceived(OnNewSerialData);
  PuckSleeveSerialPort.InitializeSlimBoard();

  // Initialize analog microphone.
  ExternalAnalogMic.OnAdcDataReady(AdcNewData);
  ExternalAnalogMic.Initialize();

  FlashLeds();

  // Initialize PWM.
  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  // Warning: issue only in Arduino. When using StartPlayback() it crashes.
  // Looks like NRF_PWM0 module is automatically triggered, and triggering it
  // again here crashes ISR. Temporary fix is to only use nrf_pwm_task_trigger
  // for NRF_PWM1 and NRF_PWM2. To fix might need a nRF52 driver update.
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);

  // Initialize tactile processor
  g_tactile_processor.Init(kSaadcSampleRateHz, kCarlBlockSize,
                           kTactileDecimationFactor);
  const float kDefaultGain = 20.0f;
  g_post_processor.Init(g_tactile_processor.GetOutputSampleRate(),
                        g_tactile_processor.GetOutputBlockSize(),
                        g_tactile_processor.GetOutputNumberTactileChannels(),
                        kDefaultGain);
  ChannelMapInit(&g_channel_map, kTactileProcessorNumTactors);

  FlashLeds();
}

void loop() {
  if (g_new_mic_data) {
    // Convert to float, the mic is already zeroed.
    for (int i = 0; i < kAdcDataSize; ++i) {
      g_audio_input[i] = (float)(g_analog_mic_data[i] / 2048.0f);
    }
    // Process samples.
    g_tactile_processor_output =
        g_tactile_processor.ProcessSamples(g_audio_input);
    g_post_processor.PostProcessSamples(g_tactile_processor_output);
    g_new_mic_data = false;
    nrf_gpio_pin_toggle(kLedPinGreen);
  }
}

void AdcNewData() {
  ExternalAnalogMic.GetData(g_analog_mic_data);
  g_new_mic_data = true;
}

void OnPwmSequenceEnd() {
  g_which_pwm_module_triggered = SleeveTactors.GetEvent();

  if (g_tactor_processor_on && g_which_pwm_module_triggered == 0) {
    constexpr int kNumChannels = 12;
    // Hardware channel `c` plays logical channel kHwToLogical[c]. Value -1
    // means that nothing is played on that channel.
    constexpr static int kHwToLogical[kNumChannels] = {6, 7, -1, -1, 1, 0,
                                                       8, 2, 4, 5, 9, 3};

    // There are two levels of mapping:
    // 1. Hardware channel `c` plays logical channel kHwToLogical[c],
    //      hw[c] = logical[kHwToLogical[c]].
    // 2. g_channel_map maps tactile channels to logical channels,
    //      logical[c] = channel_map.gains[c] * tactile[channel_map.sources[c]].
    //
    // We compose this into a single step of copying as
    //   hw[c] = channel_map.gains[logical_c]
    //           * tactile[channel_map.sources[logical_c]].
    // with logical_c = kHwToLogical[c].
    for (int c = 0; c < kNumChannels; ++c) {
      const int logical_c = kHwToLogical[c];
      // Skip unused channel. The compiler should be able to unroll this loop
      // and make this iteration a no-op.
      if (logical_c == -1) { continue; }
      const float gain = g_channel_map.gains[logical_c];
      const float* src = g_tactile_processor_output +
          g_channel_map.sources[logical_c];
      SleeveTactors.UpdateChannelWithGain(c, gain, src,
                                          kTactileProcessorNumTactors);
    }
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

void OnNewSerialData() {}
