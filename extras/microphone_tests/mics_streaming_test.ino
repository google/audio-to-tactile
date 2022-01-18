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
// This example streams pdm or analog microphone data from slim board to the PC.
// On the receiver end, I am using streaming_mic_to_pc.py script

#include "analog_external_mic.h"
#include "pdm.h"

using namespace audio_tactile;

// Choose the mic you are using.

// If audio can't keep up with real-time, buffer might need to be increased:
// kPdmDataSize in pdm.h (max 25000, default is 64)
#define USING_PDM_MIC 1

// Analog mic buffer location: kAdcDataSize in cpp/constants.h
#define USING_ANALOG_MIC 0

#if USING_PDM_MIC
constexpr int kMicBufferSize = PdmMic::kPdmDataSize;
#elif USING_ANALOG_MIC
constexpr int kMicBufferSize = kAdcDataSize;
#endif

int16_t g_mic_buffer[kMicBufferSize];

volatile bool g_mic_data_ready = false;

void NewMicData();

void setup() {
#if USING_PDM_MIC
  nrf_gpio_cfg_output(kPdmSelectPin);
  nrf_gpio_pin_write(kPdmSelectPin, 0);
  OnBoardMic.Initialize(kPdmClockPin, kPdmDataPin);
  OnBoardMic.OnPdmDataReady(NewMicData);
  OnBoardMic.Enable();
#endif

#if USING_ANALOG_MIC
  ExternalAnalogMic.OnAdcDataReady(NewMicData);
  ExternalAnalogMic.Initialize();
#endif
}

void loop() {
  if (g_mic_data_ready) {
    for (int i = 0; i < kMicBufferSize; ++i) {
      Serial.println(g_mic_buffer[i]);
    }
    g_mic_data_ready = false;
  }
}

void NewMicData() {
#if USING_PDM_MIC
  OnBoardMic.GetData(g_mic_buffer);
#elif USING_ANALOG_MIC
  ExternalAnalogMic.GetData(g_mic_buffer);
#endif
  g_mic_data_ready = true;
}
