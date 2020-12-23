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
// This example demonstrates the serial port operation.
// The green led will toggle when serial data with microphone op code is
// received. The green led will look continuously on, if data is received at
// normal (4ms) interval.

#include "serial_puck_sleeve.h"

using namespace audio_tactile;

const int kLed = 22;  // Led on pin P0.12

void on_new_serial_data() {
  switch (PuckSleeveSerialPort.GetEvent()) {
    case kLoadMicDataOpCpde:
      digitalToggle(kLed);
      break;
    case kCommError:
      // Reset if there are errors. Serial coms are not always reliable.
      NVIC_SystemReset();
      break;
    case kTimeOutError:
      NVIC_SystemReset();
      break;
    default:
      // Handle an unknown op code event.
      break;
  }
}

void setup() {
  pinMode(kLed, OUTPUT);

  // Initialize serial port.
  PuckSleeveSerialPort.InitializeSleeve();
  PuckSleeveSerialPort.OnSerialDataReceived(on_new_serial_data);
}

void loop() {}
