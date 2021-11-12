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

#include "max14661.h"  // NOLINT(build/include)

#include "two_wire.h"  // NOLINT(build/include)

namespace audio_tactile {

void Max14661::Initialize() {
  // Initializes the I2C bus.
  i2c_init(kSclPin, kSdaPin, kMax14661Address1);

  // Sets mux enable pin as output (both muxes are connected to same pin).
  nrf_gpio_cfg_output(kMuxEnable);

  // Enables the muxes.
  nrf_gpio_pin_write(kMuxEnable, 1);
}

void Max14661::Enable() { nrf_gpio_pin_write(kMuxEnable, 1); }

void Max14661::Disable() { nrf_gpio_pin_write(kMuxEnable, 0); }

void Max14661::DisconnectAllSwitches() {
  // Sets registers to zero for mux 1.
  NRF_TWIM0->ADDRESS = kMax14661Address1;
  i2c_write(DIR0, 0x00);
  i2c_write(DIR1, 0x00);
  i2c_write(DIR2, 0x00);
  i2c_write(DIR3, 0x00);

  // Sets registers to zero for mux 2.
  NRF_TWIM0->ADDRESS = kMax14661Address2;
  i2c_write(DIR0, 0x00);
  i2c_write(DIR1, 0x00);
  i2c_write(DIR2, 0x00);
  i2c_write(DIR3, 0x00);
}

void Max14661::ConnectChannel(int channel) {
  static const struct {
    uint8_t i2c_address;  // I2C address of which MAX14661 to configure.
    struct {
      uint8_t register_address;  // Which register in the MAX14661.
      uint8_t value;  // Value to write, representing a switch connection.
    } connections[2];
  }

  kChannelSettings[12] = {
      // Channel 0 (Schematic label: kPwmL1Pin).
      {
          kMax14661Address2,
          {{DIR0, 0x02},  // Connects switch AB02 to COMA.
           {DIR2, 0x01}}  // Connects switch AB01 to COMB.
      },
      // Channel 1 (kPwmR1Pin).
      {
          kMax14661Address2,
          {{DIR0, 0x08},  // Connects switch AB04 to COMA.
           {DIR2, 0x04}}  // Connects switch AB03 to COMB.
      },
      // Channel 2 (kPwmL2Pin).
      {
          kMax14661Address2,
          {{DIR0, 0x80},  // Connects switch AB08 to COMA.
           {DIR2, 0x40}}  // Connects switch AB07 to COMB.
      },
      // Channel 3 (kPwmR2Pin).
      {
          kMax14661Address2,
          {{DIR0, 0x20},  // Connects switch AB06 to COMA.
           {DIR2, 0x10}}  // Connects switch AB05 to COMB.
      },
      // Channel 4 (kPwmL3Pin).
      {
          kMax14661Address1,
          {{DIR1, 0x02},  // Connects switch AB10 to COMA.
           {DIR3, 0x01}}  // Connects switch AB09 to COMB.
      },
      // Channel 5 (kPwmR3Pin).
      {
          kMax14661Address1,
          {{DIR1, 0x08},  // Connects switch AB12 to COMA.
           {DIR3, 0x04}}  // Connects switch AB11 to COMB.
      },
      // Channel 6 (kPwmL4Pin).
      {
          kMax14661Address1,
          {{DIR1, 0x80},  // Connects switch AB16 to COMA.
           {DIR3, 0x40}}  // Connects switch AB15 to COMB.
      },
      // Channel 7 (kPwmR4Pin).
      {
          kMax14661Address1,
          {{DIR1, 0x20},  // Connects switch AB14 to COMA.
           {DIR3, 0x10}}  // Connects switch AB13 to COMB.
      },
      // Channel 8 (kPwmL5Pin).
      {
          kMax14661Address1,
          {{DIR0, 0x02},  // Connects switch AB02 to COMA.
           {DIR2, 0x01}}  // Connects switch AB01 to COMB.
      },
      // Channel 9 (kPwmR5Pin).
      {
          kMax14661Address1,
          {{DIR0, 0x08},  // Connects switch AB04 to COMA.
           {DIR2, 0x04}}  // Connects switch AB03 to COMB.
      },
      // Channel 10 (kPwmL6Pin).
      {
          kMax14661Address1,
          {{DIR0, 0x80},  // Connects switch AB08 to COMA.
           {DIR2, 0x40}}  // Connects switch AB07 to COMB.
      },
      // Channel 11 (kPwmR6Pin).
      {
          kMax14661Address1,
          {{DIR0, 0x20},  // Connects switch AB06 to COMA.
           {DIR2, 0x10}}  // Connects switch AB05 to COMB.
      },
  };

  if (channel < 0 || channel >= 12) {
    return;
  }

  // Disconnect all previous switches.
  DisconnectAllSwitches();

  const auto& settings = kChannelSettings[channel];
  NRF_TWIM0->ADDRESS = settings.i2c_address;
  i2c_write(settings.connections[0].register_address,
            settings.connections[0].value);
  i2c_write(settings.connections[1].register_address,
            settings.connections[1].value);
}

Max14661 Multiplexer;
}  // namespace audio_tactile
