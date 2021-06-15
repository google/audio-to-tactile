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
// Arduino sketch for the BMI270 IMU.
// This sketch prints out accelerometer values to USB serial port.

#include "bmi270.h"

using namespace audio_tactile;

const int16_t* accel;

void setup() {
  pinMode(27, OUTPUT);  // LED on P.019
  Imu.Initialize();
  Imu.SetToPerformanceMode();
}

void loop() {
  accel = Imu.ReadAccelerometer();
  Serial.print(accel[0]);
  Serial.print(",");
  Serial.print(accel[1]);
  Serial.print(",");
  Serial.println(accel[2]);
  delay(10);
  digitalToggle(27);
}
