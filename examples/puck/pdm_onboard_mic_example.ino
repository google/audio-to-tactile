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
// This example prints out data from the pdm buffer.
// The data is referenced with a pointer.
// Data is grabbed with GetData(), as new data interrupt is triggered.
// The buffer size is 64 signed 16-bit integers.

#include "pdm.h"

using namespace audio_tactile;

int16_t pdm_data [64];

void setup() {
  OnBoardMic.Initialize(kPdmClockPin, kPdmDataPin);
  OnBoardMic.OnPdmDataReady(pdm_new_data);
  OnBoardMic.Enable();
}

void loop() {
  for (int i = 0; i < 64; ++i) {
    Serial.println(pdm_data[i]);
  }
  delay(30);
}

void pdm_new_data() { OnBoardMic.GetData(pdm_data); }
