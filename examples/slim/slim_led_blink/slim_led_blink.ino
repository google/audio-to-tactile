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
// Simple app to blink an LED on the slim board
// Make sure that the correct board is defined in
// board_defs.h file
//
// Check variant.cpp file for Arduino pin mapping
// https://github.com/adafruit/Adafruit_nRF52_Arduino/blob/master/variants/feather_nrf52840_express/variant.cpp
// It does not map all the nRF52 pins to arduino pins, so I am using
// nrf pin toggle approach.

#include "board_defs.h"

void setup() {
  nrf_gpio_cfg_output(kLedPinBlue);
  nrf_gpio_cfg_output(kLedPinGreen);
}

void loop() {
  nrf_gpio_pin_set(kLedPinBlue);
  nrf_gpio_pin_set(kLedPinGreen);
  delay(200);
  nrf_gpio_pin_clear(kLedPinBlue);
  nrf_gpio_pin_clear(kLedPinGreen);
  delay(200);
}
