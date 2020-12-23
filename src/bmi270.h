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
// Driver for the BMI270 IMU accelerometer and gyroscope.
//
// This is a library for basic functionality of the BMI270 IMU. This IMU is
// ultra low power and has an accelerometer and gyroscope. Tested in both
// Arduino and Segger embedded studio. The only difference is the delay.h
// include path.

#ifndef AUDIO_TO_TACTILE_SRC_BMI270_H_
#define AUDIO_TO_TACTILE_SRC_BMI270_H_

#include <stdint.h>

#include "delay.h"     // NOLINT(build/include)
#include "two_wire.h"  // NOLINT(build/include)

// TODO: include 3rd party config file

namespace audio_tactile {

// Pin definitions.
enum {
  kPuckSclPin = 29,  // P0.29
  kPuckSdaPin = 47   // P1.15
};

// I2C address is defined by the chip physical wiring.
// It is always 0x68 for this board.
// Chip id is always 0x24. Can be used for checking if communications are ok.
enum {
  BMI270_I2C_ADDRESS = 0x68,
  BMI270_CHIP_ID = 0x24,
};

// BMI2 register addresses in incremental order.
// Described in the datasheet:
// https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf
// Defined:
// https://github.com/BoschSensortec/BMI270-Sensor-API/blob/master/bmi2_defs.h
enum {
  BMI2_CHIP_ID_ADDR = 0x00,
  BMI2_STATUS_ADDR = 0x03,
  BMI2_AUX_X_LSB_ADDR = 0x04,
  BMI2_ACC_X_LSB_ADDR = 0x0C,
  BMI2_GYR_X_LSB_ADDR = 0x12,
  BMI2_EVENT_ADDR = 0x1B,
  BMI2_INT_STATUS_0_ADDR = 0x1C,
  BMI2_INT_STATUS_1_ADDR = 0x1D,
  BMI2_SC_OUT_0_ADDR = 0x1E,
  BMI2_SYNC_COMMAND_ADDR = 0x1E,
  BMI2_GYR_CAS_GPIO0_ADDR = 0x1E,
  BMI2_INTERNAL_STATUS_ADDR = 0x21,
  BMI2_FIFO_LENGTH_0_ADDR = 0x24,
  BMI2_FIFO_DATA_ADDR = 0x26,
  BMI2_FEAT_PAGE_ADDR = 0x2F,
  BMI2_FEATURES_REG_ADDR = 0x30,
  BMI2_ACC_CONF_ADDR = 0x40,
  BMI2_GYR_CONF_ADDR = 0x42,
  BMI2_AUX_CONF_ADDR = 0x44,
  BMI2_FIFO_DOWNS_ADDR = 0x45,
  BMI2_FIFO_WTM_0_ADDR = 0x46,
  BMI2_FIFO_WTM_1_ADDR = 0x47,
  BMI2_FIFO_CONFIG_0_ADDR = 0x48,
  BMI2_FIFO_CONFIG_1_ADDR = 0x49,
  BMI2_AUX_DEV_ID_ADDR = 0x4B,
  BMI2_AUX_IF_CONF_ADDR = 0x4C,
  BMI2_AUX_RD_ADDR = 0x4D,
  BMI2_AUX_WR_ADDR = 0x4E,
  BMI2_AUX_WR_DATA_ADDR = 0x4F,
  BMI2_INT1_IO_CTRL_ADDR = 0x53,
  BMI2_INT2_IO_CTRL_ADDR = 0x54,
  BMI2_INT_LATCH_ADDR = 0x55,
  BMI2_INT1_MAP_FEAT_ADDR = 0x56,
  BMI2_INT2_MAP_FEAT_ADDR = 0x57,
  BMI2_INT_MAP_DATA_ADDR = 0x58,
  BMI2_INIT_CTRL_ADDR = 0x59,
  BMI2_INIT_ADDR_0 = 0x5B,
  BMI2_INIT_ADDR_1 = 0x5C,
  BMI2_INIT_DATA_ADDR = 0x5E,
  BMI2_AUX_IF_TRIM = 0x68,
  BMI2_GYR_CRT_CONF_ADDR = 0x69,
  BMI2_NVM_CONF_ADDR = 0x6A,
  BMI2_IF_CONF_ADDR = 0x6B,
  BMI2_ACC_SELF_TEST_ADDR = 0x6D,
  BMI2_GYR_SELF_TEST_AXES_ADDR = 0x6E,
  BMI2_SELF_TEST_MEMS_ADDR = 0x6F,
  BMI2_NV_CONF_ADDR = 0x70,
  BMI2_ACC_OFF_COMP_0_ADDR = 0x71,
  BMI2_GYR_OFF_COMP_3_ADDR = 0x74,
  BMI2_GYR_OFF_COMP_6_ADDR = 0x77,
  BMI2_GYR_USR_GAIN_0_ADDR = 0x78,
  BMI2_PWR_CONF_ADDR = 0x7C,
  BMI2_PWR_CTRL_ADDR = 0x7D,
  BMI2_CMD_REG_ADDR = 0x7E
};

class Bmi270 {
 public:
  Bmi270();

  // Load 8Kb initialization file.
  // This should only be performed once every reset.
  bool Initialize();

  // Enable accelerometer only.
  void SetToLowPowerMode();

  // Enable gyro, accelerometer and temperature sensor.
  void SetToNormalMode();

  // Enable gyro, accelerometer and temperature sensor.
  // Accelerometer is the same in performance and normal mode.
  // However, gyroscope noise is reduced in performance mode.
  void SetToPerformanceMode();

  // Should always return 0x24 to check basic communications.
  uint8_t ReadId();

  // Returns the accelerometer x,y,z values, signed.
  // Default range is +/-8g, 4096 LSB per g (register 0x41).
  // LSB stands for Least Significant Bit. In this case it is a unit for the
  // digital 16-bit values from the sensor.
  // For example, LSB value of 4096 represents 1 g acceleration.
  // LSB can change depending on the range.
  const int16_t* ReadAccelerometer();

  // Returns the gyro x,y,z values, signed. No gyro in low power mode.
  // Default range is +/- 2000 dps (degrees-per-second) and 16.4 LSB per dps
  // (reg 0x43).
  const int16_t* ReadGyroscope();

  // Disable all sensors (gyro, accel, temperature).
  void SetAllSensorsOff();
};

extern Bmi270 Imu;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_BMI270_H_
