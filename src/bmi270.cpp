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

#include "bmi270.h"  // NOLINT(build/include)

namespace audio_tactile {

Bmi270::Bmi270() {}

bool Bmi270::Initialize() {
  // Initialize the I2C bus.
  i2c_init(kPuckSclPin, kPuckSdaPin, BMI270_I2C_ADDRESS);

  // Trigger a soft reset to clear all registers.
  i2c_write(BMI2_CMD_REG_ADDR, 0xb6);

  // Add a hardware delay, since the IMU need to start up and won't respond.
  nrfx_coredep_delay_us(5000);

  // Check if we can read id, if not there is no point to initialize.
  uint8_t id_reg = i2c_read(BMI2_CHIP_ID_ADDR);
  if (id_reg != 0x24) {
    return 1;
  }

  // Perform initialization steps as described in the datasheet on page 20:
  // https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf
  i2c_write(BMI2_PWR_CONF_ADDR, 0x00);
  // Add delay for imu to startup.
  nrfx_coredep_delay_us(500);
  i2c_write(BMI2_INIT_CTRL_ADDR, 0x00);

  // Load a 8 kB file. It is hex file with unknown information,
  // but required for operation.
  // I added the address as first byte of the array, so
  // we don't need to touch the const array.
  // TODO: add bmi270_config_file to 3rd party folder
  i2c_write_array(bmi270_config_file, 8193);
  i2c_write(BMI2_INIT_CTRL_ADDR, 0x01);

  // Run until initialization is complete or time out.
  uint16_t count = 0;
  while (i2c_read(BMI2_INTERNAL_STATUS_ADDR) != 1) {
    nrfx_coredep_delay_us(10);
    count = count + 1;
    // If doesn't initialize after some time return error.
    if (count > 1000) {
      return 1;
    }
  }
  return 0;
}

void Bmi270::SetToLowPowerMode() {
  i2c_write(BMI2_PWR_CTRL_ADDR, 0x04);
  i2c_write(BMI2_ACC_CONF_ADDR, 0x17);
  i2c_write(BMI2_PWR_CONF_ADDR, 0x03);
}

void Bmi270::SetToNormalMode() {
  i2c_write(BMI2_PWR_CTRL_ADDR, 0x0E);
  i2c_write(BMI2_ACC_CONF_ADDR, 0xA8);
  i2c_write(BMI2_GYR_CONF_ADDR, 0xA9);
  i2c_write(BMI2_PWR_CONF_ADDR, 0x02);
}

void Bmi270::SetToPerformanceMode() {
  i2c_write(BMI2_PWR_CTRL_ADDR, 0x0E);
  i2c_write(BMI2_ACC_CONF_ADDR, 0xA8);
  i2c_write(BMI2_GYR_CONF_ADDR, 0xE9);
  i2c_write(BMI2_PWR_CONF_ADDR, 0x02);
}

const int16_t* Bmi270::ReadAccelerometer() {
  static int16_t accel[3];
  uint8_t* buffer;
  buffer = i2c_read_array(BMI2_ACC_X_LSB_ADDR, 6);
  accel[0] = (((uint16_t)buffer[1]) << 8) | ((uint16_t)buffer[0]);
  accel[1] = (((uint16_t)buffer[3]) << 8) | ((uint16_t)buffer[2]);
  accel[2] = (((uint16_t)buffer[5]) << 8) | ((uint16_t)buffer[4]);
  return accel;
}

const int16_t* Bmi270::ReadGyroscope() {
  static int16_t gyro[3];
  uint8_t* buffer;
  buffer = i2c_read_array(BMI2_GYR_X_LSB_ADDR, 6);
  gyro[0] = (((uint16_t)buffer[1]) << 8) | ((uint16_t)buffer[0]);
  gyro[1] = (((uint16_t)buffer[3]) << 8) | ((uint16_t)buffer[2]);
  gyro[2] = (((uint16_t)buffer[5]) << 8) | ((uint16_t)buffer[4]);
  return gyro;
}

uint8_t Bmi270::ReadId() {
  uint8_t id_reg = i2c_read(0x00);
  return id_reg;
}

void Bmi270::SetAllSensorsOff() { i2c_write(BMI2_PWR_CTRL_ADDR, 0x00); }

Bmi270 Imu;

}  // namespace audio_tactile
