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
// An example of the temperature monitor.

#include "temperature_monitor.h"
const int kLed = 22;  // Led on pin P0.12

using namespace audio_tactile;

// This interrupt is triggered on overheating.
void on_overheating() { digitalWrite(kLed, 1); }

void setup() {
  // Initialize temperature monitor.
  SleeveTemperatureMonitor.StartMonitoringTemperature();
  SleeveTemperatureMonitor.OnOverheatingEventListener(on_overheating);

  pinMode(kLed, OUTPUT);
  digitalWrite(kLed, 0);
}

void loop() {}
