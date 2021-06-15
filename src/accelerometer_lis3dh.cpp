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

#include "two_wire.h"  // NOLINT(build/include)
#include "accelerometer_lis3dh.h"  // NOLINT(build/include)
#include "dsp/serialize.h"

namespace audio_tactile {

Lis3dh::Lis3dh() {}

bool Lis3dh::Initialize() {
  // Initialize the I2C bus.
  i2c_init(kSclPin, kSdaPin, kLis3dhAddress);

  // Check if we can read WHO_AM_I register, if not there is no point to
  // initialize.
  uint8_t id_reg = i2c_read(WHO_AM_I);
  if (id_reg != kWhoAmIResponse) {
    return true;
  }

  // Enable X,Y,Z axes in normal mode.
  // Set sampling rate to 25 Hz, which also wakes up the accelerometer.
  i2c_write(CTRL_REG1, 0x37);  // Binary: 0110111.

  // Enable block data update (BDU) to update all the accel axis at once. Also,
  // enable high resolution.
  i2c_write(CTRL_REG4, 0x88);

  return false;
}

void Lis3dh::Disable() {
  // Disable all the axis, put accelerometer to sleep.
  i2c_write(CTRL_REG1, 0x00);
}

void Lis3dh::Enable() {
  // Enable all the axis, wake up the accelerometer.
  i2c_write(CTRL_REG1, 0x37);
}

const int16_t* Lis3dh::ReadXyzAccelerationRaw() {
  static int16_t accel[3];
  uint8_t* buffer;

  // Set the most significant bit of the first read register to 1, to enable
  // auto-increment. This way we can read all 6 acceleration registers at once.
  uint8_t auto_increment_address = OUT_X_L;
  auto_increment_address |= 0x80;

  buffer = i2c_read_array(auto_increment_address, 6);

  accel[0] = LittleEndianReadS16(buffer);
  accel[1] = LittleEndianReadS16(buffer + 2);
  accel[2] = LittleEndianReadS16(buffer + 4);
  return accel;
}

const float* Lis3dh::ReadXyzAccelerationFloat() {
  const int16_t* accel_raw;
  static float accel[3];

  accel_raw = ReadXyzAccelerationRaw();

  // The conversion constant from 16-bit integer to gravity (g).
  // The constant is calculated as following: (2^16 bit / 2) / Max acceleration.
  // The converstion constant will need to be adjusted if different max
  // acceleration is set.
  const float kAccelerationRange =
      2.0f;  // Current max acceleration is set to 2 Gs.
  const float kConversionIntToFloat = 32768.0f / kAccelerationRange;

  accel[0] = (float)accel_raw[0] / kConversionIntToFloat;
  accel[1] = (float)accel_raw[1] / kConversionIntToFloat;
  accel[2] = (float)accel_raw[2] / kConversionIntToFloat;
  return accel;
}

Lis3dh Accelerometer;
}  // namespace audio_tactile
