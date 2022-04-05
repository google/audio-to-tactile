// Copyright 2022 Google LLC
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
// This is an example Arduino app for slim board that demonstrates tactile
// patterns use. The patterns are played by pressing different keys on Arduino
// serial terminal: '1' to play simple connect pattern '2' to play a custom
// simple pattern '3' to play a custom extended pattern.
// There are two types of patterns: 1) Simple, that plays on all actuators.
// 2) Extended, that allows fine control of each actuator.

// Note: the pdm microphone module is used to provide 4.096 ms timer for tactile
// playback.

#include "flash_settings.h"
#include "pdm.h"
#include "post_processor_cpp.h"
#include "pwm_sleeve.h"
#include "tactile/tactile_pattern.h"
#include "tactile_processor_cpp.h"

using namespace audio_tactile;

// Compile time constants.
constexpr int kTactileDecimationFactor = 8;
constexpr int kSaadcSampleRateHz = 15625;
constexpr int kCarlBlockSize = 64;
constexpr int kPwmSamplesAllChannels = kNumPwmValues * kNumTotalPwm;
constexpr int kTactileFramesPerCarlBlock =
    kCarlBlockSize / kTactileDecimationFactor;

static uint8_t g_which_pwm_module_triggered;

static bool g_new_mic_data = false;

// Buffer of input audio as floats in [-1, 1].
static float g_audio_input[kAdcDataSize];

TactileProcessorWrapper g_tactile_processor;
PostProcessorWrapper g_post_processor;

// Device name, tuning, and channel gain and assignment.
Settings g_settings;

// Tactile pattern synthesizer.
TactilePattern g_tactile_pattern;
// True when a tactile pattern is active.
bool g_tactile_pattern_active = false;

// Pointer to the tactile output buffer of g_tactile_processor.
static float* g_tactile_output;

void FlashLeds();
void OnPwmSequenceEnd();
void OnPdmNewData();

const uint8_t extended_pattern[] = {
    // Play 60 Hz tone on channels 1 and channel 5, for 80 ms.
    kTactilePatternOpSetWaveform + 1,
    kTactilePatternWaveformSin60Hz,
    kTactilePatternOpSetWaveform + 5,
    kTactilePatternWaveformSin60Hz,
    TACTILE_PATTERN_OP_PLAY_MS(80),
    // Play 50 Hz tone on channels 1 and 5.
    kTactilePatternOpSetWaveform + 1,
    kTactilePatternWaveformSin50Hz,
    kTactilePatternOpSetWaveform + 5,
    kTactilePatternWaveformSin50Hz,
    TACTILE_PATTERN_OP_PLAY_MS(80),
    // Play 35 Hz tone on channel 5.
    kTactilePatternOpSetWaveform + 5,
    kTactilePatternWaveformSin35Hz,
    TACTILE_PATTERN_OP_PLAY_MS(160),
    // Pause for 80 ms.
    TACTILE_PATTERN_OP_PLAY_MS(80),
    // Play 90 Hz tone on channel 1.
    kTactilePatternOpSetWaveform + 1,
    kTactilePatternWaveformSin90Hz,
    TACTILE_PATTERN_OP_PLAY_MS(100),
    // Pause for 80 ms.
    TACTILE_PATTERN_OP_PLAY_MS(80),
    // Play 90 Hz tone on channel 5.
    kTactilePatternOpSetWaveform + 5,
    kTactilePatternWaveformSin90Hz,
    TACTILE_PATTERN_OP_PLAY_MS(100),
    kTactilePatternOpEnd,
};

void setup() {
  // Set the indicator led pin to output.
  nrf_gpio_cfg_output(kLedPinBlue);
  nrf_gpio_cfg_output(kLedPinGreen);

  // Initialize pdm mic. We are just using it as a 4.096 ms timer for tactile
  // patterns.
  nrf_gpio_cfg_output(kPdmSelectPin);
  nrf_gpio_pin_write(kPdmSelectPin, 0);
  OnBoardMic.Initialize(kPdmClockPin, kPdmDataPin);
  OnBoardMic.OnPdmDataReady(OnPdmNewData);
  OnBoardMic.Enable();

  // Initialize tactile processor.
  g_tactile_processor.Init(kSaadcSampleRateHz, kCarlBlockSize,
                           kTactileDecimationFactor);
  constexpr float kDefaultGain = 10.0f;
  g_post_processor.Init(g_tactile_processor.GetOutputSampleRate(),
                        g_tactile_processor.GetOutputBlockSize(),
                        g_tactile_processor.GetOutputNumberTactileChannels(),
                        kDefaultGain);

  TactilePatternInit(&g_tactile_pattern,
                     g_tactile_processor.GetOutputSampleRate(),
                     g_tactile_processor.GetOutputNumberTactileChannels());

  // Initialize PWM.
  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  // Turn the tactor amplifiers off initially, otherwise they buzz arbitrarly
  // while the system is still starting up.
  SleeveTactors.DisableAmplifiers();
  // Warning: issue only in Arduino. When using StartPlayback() it crashes.
  // Looks like NRF_PWM0 module is automatically triggered, and triggering it
  // again here crashes ISR. Temporary fix is to only use nrf_pwm_task_trigger
  // for NRF_PWM1 and NRF_PWM2. To fix might need a nRF52 driver update.
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);

  FlashLeds();
  SleeveTactors.EnableAmplifiers();
}

void loop() {
  // Play different patterns depending on the keyboard presses.
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    switch (c) {
      case '1':
        Serial.println("Playing kTactilePatternConnect tactile pattern");
        TactilePatternStart(&g_tactile_pattern, kTactilePatternConnect);
        g_tactile_pattern_active = true;
        break;
      case '2':
        Serial.println("Playing custom simple tactile pattern");
        TactilePatternStart(&g_tactile_pattern, "A66   888   888");
        g_tactile_pattern_active = true;
        break;
      case '3':
        Serial.println("Playing custom extended tactile pattern");
        TactilePatternStartEx(&g_tactile_pattern, extended_pattern);
        g_tactile_pattern_active = true;
        break;
      default:
        // Do nothing if first character is not '1','2' or '3'.
        break;
    }
  }

  if (g_new_mic_data) {
    // Play synthesized tactile pattern if a pattern is active.
    if (g_tactile_pattern_active) {
      g_tactile_pattern_active = TactilePatternSynthesize(
          &g_tactile_pattern, g_tactile_processor.GetOutputBlockSize(),
          g_tactile_output);
    } else {
      // Process samples.
      g_tactile_output = g_tactile_processor.ProcessSamples(g_audio_input);
    }
    g_post_processor.PostProcessSamples(g_tactile_output);

    g_new_mic_data = false;
    nrf_gpio_pin_toggle(kLedPinGreen);
  }
}

void OnPdmNewData() { g_new_mic_data = true; }

void OnPwmSequenceEnd() {
  g_which_pwm_module_triggered = SleeveTactors.GetEvent();

  if (g_which_pwm_module_triggered == 0) {
    constexpr int kNumChannels = 12;
    // Hardware channel `c` plays logical channel kHwToLogical[c]. Value -1
    // means that nothing is played on that channel.
    constexpr static int kHwToLogical[kNumChannels] = {6, 7, -1, -1, 1, 0,
                                                       8, 2, 4,  5,  3, 9};
    // There are two levels of mapping:
    // 1. Hardware channel `c` plays logical channel kHwToLogical[c],
    //      hw[c] = logical[kHwToLogical[c]].
    // 2. g_settings.channel_map maps tactile channels to logical channels,
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
      if (logical_c == -1) {
        continue;
      }
      const float gain = g_settings.channel_map.gains[logical_c];
      const float* src =
          g_tactile_output + g_settings.channel_map.sources[logical_c];
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
