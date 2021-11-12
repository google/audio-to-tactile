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
// Library for the Max14661 16:2 multiplexer (mux) from Maxim Integrated. This
// multiplexer has 16 inputs and 2 outputs. The connections between inputs and
// outputs can be set programmatically in any configuration.
//
// This driver is written for use with 12 vibrotactile channels in Slim Board
// for current sensing. The board has two 16:2 muxes, so for ease of
// integration, the driver behaves as there is one 32:2 mux.
//
// The datasheet is provided here:
// https://datasheets.maximintegrated.com/en/ds/MAX14661.pdf

#ifndef AUDIO_TO_TACTILE_SRC_MAX14661_H_
#define AUDIO_TO_TACTILE_SRC_MAX14661_H_

#include <stdint.h>

namespace audio_tactile {

class Max14661 {
 public:
  enum {
    // The I2C address of the multiplexer is set by A0 and A1 pins, which are
    // physical pins that can be connected to ground (low) or Vdd (high)
    // The byte address is determined by binary: 10011[A1][A0]
    kMax14661Address1 = 0x4C,  // A0: 0, A1: 0. b1001100
    kMax14661Address2 = 0x4E,  // A0: 0, A1: 1. b1001110
    // Pin definitions.
    kMuxEnable = 2,
    kSclPin = 25,
    kSdaPin = 24
  };
  // Initializes the two muxes.
  void Initialize();

  // Enables the two muxes.
  void Enable();

  // Disables the two muxes.
  void Disable();

  // Connects output to one of the vibrotactile channels from 0 to 11.
  // One mux connects to the high side of current sensing resistor, and the
  // second mux connects to the low side of current sensing resistor.
  void ConnectChannel(int channel);

  // Disconnects all switches inside the two muxes. This means inputs are not
  // connected to outputs.
  void DisconnectAllSwitches();

 private:
  // Register map constants from the mux datasheet.
  enum {
    DIR0 = 0x00,
    DIR1 = 0x01,
    DIR2 = 0x02,
    DIR3 = 0x03,
    SHDW0 = 0x10,
    SHDW1 = 0x11,
    SHDW2 = 0x12,
    SHDW3 = 0x13,
    CMD_A = 0x14,
    CMD_B = 0x15
  };
};

extern Max14661 Multiplexer;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_MAX14461_H_
