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
// This example prints out data collected by an external microphone.
// Data is copied on adc_new_data() and displayed with a serial output.

#include "analog_external_mic.h"

using namespace audio_tactile;

int16_t analog_mic_data[kAdcDataSize];

void setup() {
  ExternalAnalogMic.Initialize();
  ExternalAnalogMic.OnAdcDataReady(adc_new_data);
}

void loop() {
  for (int i = 0; i < kAdcDataSize; ++i) {
    Serial.println(analog_mic_data[i]);
  }
  delay(15);
}

void adc_new_data() { ExternalAnalogMic.GetData(analog_mic_data); }
