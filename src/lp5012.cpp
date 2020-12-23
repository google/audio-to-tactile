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

#include "lp5012.h"  // NOLINT(build/include)

#include "two_wire.h"  // NOLINT(build/include)

namespace audio_tactile {

Lp5012::Lp5012() {}

void Lp5012::Initialize() {
  // Initialize the I2C bus.
  i2c_init(kSclPin, kSdaPin, kLp5012Address);

  // Set led driver enable pin as output.
  nrf_gpio_cfg_output(kLp5012EnablePin);

  // Toggle enable pin to reset the chip.
  // Datasheet doesn't provide information about how long to hold the reset
  // line. I set the delay to 5 ms, which seemed to work ok and seems
  // reasonable.
  nrf_gpio_pin_write(kLp5012EnablePin, 0);
  nrfx_coredep_delay_us(5000);  // 5 ms
  nrf_gpio_pin_write(kLp5012EnablePin, 1);
  nrfx_coredep_delay_us(5000);  // 5 ms

  // Enable the led driver by setting the register Chip_EN to 1, which
  // translates to b0100 0000 (0x40).
  i2c_write(LP5012_DEVICE_CONFIG0, 0x40);

  // Wait while chip goes online. This delay is a guess as datasheet doesn't
  // provide information.
  nrfx_coredep_delay_us(5000);  // 5 ms wait time.
}

void Lp5012::TurnOnAllLeds(uint8_t brightness) {
  for (int i = 0; i < kNumLeds; ++i) {
    SetOneLed(i, brightness);
  }
}

void Lp5012::SetOneLed(uint8_t led, uint8_t brightness) {
  i2c_write(LP5012_OUT0_COLOR + led, brightness);
}

void Lp5012::CycleAllLeds(int cycles) {
  for (int g = 0; g < cycles; ++g) {
    for (int i = 0; i < 256; ++i) {
      TurnOnAllLeds(i);
      nrfx_coredep_delay_us(2000);
    }

    for (int i = 255; i >= 0; --i) {
      TurnOnAllLeds(i);
      nrfx_coredep_delay_us(2000);
    }
  }
}

void Lp5012::CycleLedsOneByOne(int cycles) {
  for (int g = 0; g < cycles; ++g) {
    for (int h = 0; h < kNumLeds; ++h) {
      for (int i = 0; i < 256; ++i) {
        SetOneLed(h, i);
        nrfx_coredep_delay_us(300);
      }

      for (int i = 255; i >= 0; --i) {
        SetOneLed(h, i);
        nrfx_coredep_delay_us(300);
      }
    }
  }
}

void Lp5012::LedBar(int value, int brightness) {
  const int kNumLedsInBar = 10;
  for (int i = 0; i < kNumLedsInBar; ++i) {
    if ((uint8_t)value >= 25 * i) {
      LedArray.SetOneLed(i, (uint8_t)brightness);
    }
  }
}

void Lp5012::Clear() {
  for (int i = 0; i < kNumLeds; ++i) {
    SetOneLed(i, 0);
  }
}

void Lp5012::Disable() { nrf_gpio_pin_write(kLp5012EnablePin, 0); }

Lp5012 LedArray;

}  // namespace audio_tactile
