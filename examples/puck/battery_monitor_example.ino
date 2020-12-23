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

#include "battery_monitor.h"

using namespace audio_tactile;

void setup() {
  pinMode(27, OUTPUT);  // P.019 LED
  PuckBatteryMonitor.Initialize();
  PuckBatteryMonitor.OnLowBatteryEventListener(low_battery_warning);
}

void loop() {
  uint16_t battery = PuckBatteryMonitor.MeasureBatteryVoltage();
  float converted = PuckBatteryMonitor.ConvertBatteryVoltageToFloat(battery);
  Serial.println(converted);
  delay(100);
}

void low_battery_warning() {
  digitalToggle(27);  // Toggle the led.
  Serial.print("Low voltage trigger: ");
  Serial.println(PuckBatteryMonitor.GetEvent());
  // "0" event means that battery voltage is below reference voltage (3.5V)
  // "1" event means above.
}
