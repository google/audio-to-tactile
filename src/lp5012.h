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
// Library for the Lp5012 12-channels I2C LED driver from Texas Instruments.
//
// The datasheet is provided here: https://www.ti.com/product/LP5012

#ifndef AUDIO_TO_TACTILE_SRC_LP5012_H_
#define AUDIO_TO_TACTILE_SRC_LP5012_H_

#include <stdint.h>

namespace audio_tactile {

// Lp5012 control registers as named in the datasheet.
enum {
  LP5012_DEVICE_CONFIG0 = 0x00,
  LP5012_DEVICE_CONFIG1 = 0x01,
  LP5012_LED_CONFIG0 = 0x02,
  LP5012_BANK_BRIGHTNESS = 0x03,
  LP5012_BANK_A_COLOR = 0x04,
  LP5012_BANK_B_COLOR = 0x05,
  LP5012_BANK_C_COLOR = 0x06,
  LP5012_LED0_BRIGHTNESS = 0x07,
  LP5012_LED1_BRIGHTNESS = 0X08,
  LP5012_LED2_BRIGHTNESS = 0x09,
  LP5012_LED3_BRIGTNESS = 0x0A,
  LP5012_OUT0_COLOR = 0x0B,
  LP5012_OUT1_COLOR = 0x0C,
  LP5012_OUT2_COLOR = 0x0D,
  LP5012_OUT3_COLOR = 0x0E,
  LP5012_OUT4_COLOR = 0x0F,
  LP5012_OUT5_COLOR = 0x10,
  LP5012_OUT6_COLOR = 0x11,
  LP5012_OUT7_COLOR = 0x12,
  LP5012_OUT8_COLOR = 0x13,
  LP5012_OUT9_COLOR = 0x14,
  LP5012_OUT10_COLOR = 0x15,
  LP5012_OUT11_COLOR = 0x16,
  LP5012_RESET = 0x17
};

// Hardware constants.
enum {
  kLp5012Address = 0x14,  // b0010100
  // Pin definitions.
  kLp5012EnablePin = 20,
  kSclPin = 22,
  kSdaPin = 24
};

class Lp5012 {
 public:
  Lp5012();

  // Number of LEDs, controlled by Lp5012.
  enum { kNumLeds = 12 };

  // Initialize the led driver.
  void Initialize();

  // Set brightness register of one led. Brigtness is from 0 (min) to 255 (max).
  // Leds are numbered 0 to 11.
  void SetOneLed(uint8_t which_led, uint8_t brightness);

  // Set one brightness for all leds.
  void TurnOnAllLeds(uint8_t brightness);

  // Disable led driver.
  void Disable();

  // Flash all leds at once with increasing then decreasing brightness. Repeated
  // n number of cycles.
  void CycleAllLeds(int cycles);

  // Flash leds one by one. Repeated n number of cycles.
  void CycleLedsOneByOne(int cycles);

  // Show a bar chart of LEDs, represented by the value from 0 to 255.
  // This is a good way to show magnitude.
  // Leds should be cleared before setting the LedBar, since this function does
  // not automatically clear them.
  // Brigtness is from 0 to 255.
  void LedBar(int value, int brightness);

  // Clear all leds by setting them to 0.
  void Clear();
};

extern Lp5012 LedArray;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_LP5012_H_
