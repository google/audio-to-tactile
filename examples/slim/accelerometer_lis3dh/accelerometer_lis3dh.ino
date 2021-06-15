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
// Example app that plots accelerometer X,Y,Z axis data over USB port.
// The accelerometer is LIS3DH from STMicro.

#include "accelerometer_lis3dh.h"

#include "two_wire.h"

using namespace audio_tactile;

const int16_t* accel;

void setup() {
  Serial.begin(0);
  Accelerometer.Initialize();
}

void loop() {
  accel = Accelerometer.ReadXyzAccelerationRaw();
  Serial.print(accel[0]);
  Serial.print(",");
  Serial.print(accel[1]);
  Serial.print(",");
  Serial.println(accel[2]);
  delay(40);
}
