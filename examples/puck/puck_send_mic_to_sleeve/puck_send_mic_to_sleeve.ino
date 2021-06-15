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
// This is example code for the puck that takes microphone data and pipes it to
// the sleeve. The navigation switch can be used to control the gain, denoising
// and compression. Button 1 press will mute the tactors. Pressing it again will
// unmute them. Button 2 press will cause the system to go to test mode. In this
// mode the tactors are buzzed one by one. Pressing it again will go back to the
// microphone mode.

#include "analog_external_mic.h"
#include "ble_com.h"
// #include "bmi270.h"
#include "dsp/channel_map.h"
#include "dsp/serialize.h"
#include "serial_com.h"
#include "tactile/envelope_tracker.h"
#include "tactile/tactile_pattern.h"
#include "tactile/tuning.h"
#include "two_wire.h"
#include "ui.h"

using namespace audio_tactile;

int16_t g_analog_mic_data[kAdcDataSize];
bool g_new_mic_data;
constexpr uint32_t kLedPin = 19;

static uint16_t g_sin_wave_downsample[64] = {};
static uint16_t g_no_sound[64] = {};

int g_which_tactor_to_play = 1;  // Tactor count starts at 1.
bool g_amps_enabled = true;
bool g_ble_connected = false;
bool g_test_mode_enabled = false;
int g_config_mode = 0;
EnvelopeTracker g_envelope_tracker;
TuningKnobs g_tuning_knobs = kDefaultTuningKnobs;
char g_tactile_pattern[kMaxTactilePatternLength + 1];
ChannelMap g_channel_map;

Message g_pending_message;

constexpr int kStepSizeNavSwitchTune = 10;

void RunEnvelopeTracker();
void LoopTestMode();
void LoopMicMode();
void AdcNewData();
void TouchEvent();
void OnBleEvent();

void setup() {
  // Set custom pwm values for testing with sin wave.
  // The sleeve expects a 64 point array.
  g_sin_wave_downsample[0] = 256;
  g_sin_wave_downsample[1] = 437;
  g_sin_wave_downsample[2] = 512;
  g_sin_wave_downsample[3] = 437;
  g_sin_wave_downsample[4] = 256;
  g_sin_wave_downsample[5] = 75;
  g_sin_wave_downsample[6] = 0;
  g_sin_wave_downsample[7] = 75;

  ChannelMapInit(&g_channel_map, 10);

  // SAADC sample rate in Hz.
  constexpr float kSaadcSampleRateHz = 15625.0f;
  EnvelopeTrackerInit(&g_envelope_tracker, kSaadcSampleRateHz);

  // Set LED as output.
  nrf_gpio_cfg_output(kLedPin);

  // Set I2C pins as output.
  pinMode(20, OUTPUT);
  pinMode(3, OUTPUT);

  // Initialize the drivers.
  // Imu.Initialize();
  // Imu.SetToPerformanceMode();
  PuckUi.Initialize();
  PuckUi.OnUiEventListener(TouchEvent);
  ExternalAnalogMic.Initialize();
  ExternalAnalogMic.OnAdcDataReady(AdcNewData);
  SerialCom.InitPuck(+[] {} /* Pass do-nothing callback. */);
  BleCom.Init("Audio-to-Tactile puck", OnBleEvent);

  // Tell the sleeve to turn on the amplifiers.
  g_pending_message.WriteEnableAmplifiers();

  delay(5000);
}

void HandleMessage(const Message& message) {
  switch (message.type()) {
    case MessageType::kGetTuning:
      Serial.println("Message: GetTuning.");
      BleCom.tx_message().WriteTuning(g_tuning_knobs);
      BleCom.SendTxMessage();
      g_pending_message.WriteTactilePattern(kTactilePatternConnect);
      break;
    case MessageType::kTuning:
      Serial.println("Message: Tuning.");
      message.ReadTuning(&g_tuning_knobs);
      g_pending_message = message;  // Forward this message to the sleeve.
      break;
    case MessageType::kTactilePattern:
      Serial.println("Message: TactilePattern.");
      g_pending_message = message;  // Forward this message to the sleeve.
      break;
    case MessageType::kGetChannelMap:
      Serial.println("Message: GetChannelMap.");
      BleCom.tx_message().WriteChannelMap(g_channel_map);
      BleCom.SendTxMessage();
      break;
    case MessageType::kChannelMap:
      Serial.println("Message: ChannelMap.");
      message.ReadChannelMap(&g_channel_map);
      g_pending_message = message;  // Forward this message to the sleeve.
      break;
    case MessageType::kChannelGainUpdate:
      Serial.println("Message: ChannelGainUpdate.");
      int test_channels[2];
      message.ReadChannelGainUpdate(&g_channel_map, test_channels);
      g_pending_message = message;  // Forward this message to the sleeve.
      break;

    default:
      Serial.println("Unhandled message.");
      break;
  }
}

void loop() {
  if (g_new_mic_data && (!g_test_mode_enabled)) {
    LoopMicMode();
  }

  if (g_test_mode_enabled) {
    LoopTestMode();
  }
}

void RunEnvelopeTracker() {
  float samples[kAdcDataSize];
  // Convert ADC values to floats. The raw ADC values can swing from -2048 to
  // 2048, so we scale by that value.
  const float scale = TuningGetInputGain(&g_tuning_knobs) / 2048.0f;
  for (int i = 0; i < kAdcDataSize; ++i) {
    samples[i] = scale * g_analog_mic_data[i];
  }

  if (EnvelopeTrackerProcessSamples(
        &g_envelope_tracker, samples, kAdcDataSize) && g_ble_connected) {
    BleCom.tx_message().WriteStatsRecord(g_envelope_tracker);
    BleCom.SendTxMessage();
  }
}

void LoopMicMode() {
  RunEnvelopeTracker();

  // There will be gaps in audio processing vs. other messages.
  if (g_pending_message.type() != MessageType::kNone) {
    SerialCom.tx_message() = g_pending_message;
    g_pending_message.set_type(MessageType::kNone);
  } else if (g_amps_enabled) {
    SerialCom.tx_message().WriteAudioSamples(
        Slice<const int16_t, kAdcDataSize>(g_analog_mic_data));
  } else {
    SerialCom.tx_message().WriteDisableAmplifiers();
  }

  SerialCom.SendTxMessage();
  g_new_mic_data = false;
}

void LoopTestMode() {
  if (g_pending_message.type() != MessageType::kNone) {
    SerialCom.tx_message() = g_pending_message;
    SerialCom.SendTxMessage();
    g_pending_message.set_type(MessageType::kNone);
  } else if (g_amps_enabled) {
    delay(300);
    SerialCom.tx_message().WriteSingleTactorSamples(
        g_which_tactor_to_play,
        Slice<const uint16_t, kNumPwmValues>(g_sin_wave_downsample));
    SerialCom.SendTxMessage();
    delay(300);
    SerialCom.tx_message().WriteSingleTactorSamples(
        g_which_tactor_to_play,
        Slice<const uint16_t, kNumPwmValues>(g_no_sound));
    SerialCom.SendTxMessage();
    g_which_tactor_to_play++;

    // Tactor count starts at 1.
    if (g_which_tactor_to_play > 10) {
      g_which_tactor_to_play = 1;
    }
  } else {
    SerialCom.tx_message().WriteDisableAmplifiers();
    SerialCom.SendTxMessage();
  }

  delay(100);  // needs a delay here, otherwise doesn't go in the
               // microphone mode correctly.
}

void AdcNewData() {
  ExternalAnalogMic.GetData(g_analog_mic_data);
  g_new_mic_data = true;
}

void TouchEvent() {
  switch (PuckUi.GetEvent()) {
    // Button 1 press. Disable or enable tactors.
    case 0:
      g_amps_enabled = !g_amps_enabled;
      if (g_amps_enabled) {
        g_pending_message.WriteEnableAmplifiers();
      }
      break;

    // Button 2 press. Test or Microphone mode.
    case 1:
      g_test_mode_enabled = !g_test_mode_enabled;
      break;

    // Decrease navigation switch.
    case 2:
      g_tuning_knobs.values[g_config_mode] =
          g_tuning_knobs.values[g_config_mode] + kStepSizeNavSwitchTune;
      if (g_tuning_knobs.values[g_config_mode] < 0) {
        g_tuning_knobs.values[g_config_mode] = 0;
      }
      if (g_tuning_knobs.values[g_config_mode] > 255) {
        g_tuning_knobs.values[g_config_mode] = 255;
      }
      g_pending_message.WriteTuning(g_tuning_knobs);
      break;

    // Increase nav switch.
    case 3:
      g_tuning_knobs.values[g_config_mode] =
          g_tuning_knobs.values[g_config_mode] - kStepSizeNavSwitchTune;
      if (g_tuning_knobs.values[g_config_mode] < 0) {
        g_tuning_knobs.values[g_config_mode] = 0;
      }
      if (g_tuning_knobs.values[g_config_mode] > 255) {
        g_tuning_knobs.values[g_config_mode] = 255;
      }
      g_pending_message.WriteTuning(g_tuning_knobs);
      break;

    // Press nav switch.
    case 4:
      g_config_mode = g_config_mode + 1;
      if (g_config_mode >= 3) {
        g_config_mode = 0;
      }
      break;
  }
}

void OnBleEvent() {
  switch (BleCom.event()) {
    case BleEvent::kConnect:
      Serial.println("BLE: Connected.");
      g_ble_connected = true;
      break;
    case BleEvent::kDisconnect:
      Serial.println("BLE: Disconnected.");
      g_ble_connected = false;
      break;
    case BleEvent::kInvalidMessage:
      Serial.println("BLE: Invalid message.");
      break;
    case BleEvent::kMessageReceived:
      HandleMessage(BleCom.rx_message());
      break;
  }
}
