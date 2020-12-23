/* Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * Hardware level (registers) library for the I2C controller.
 *
 * The I2C controller is also called TWIM in nRF. It is mostly intended to
 * communicate with chips (e.g. IMU, sensors). This is a simple library so it
 * does not implement callbacks and peripheral functionality. A nice thing is
 * that it does not have any dependencies. The module is described here:
 * https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Ftwim.html
 */

#ifndef AUDIO_TO_TACTILE_SRC_TWO_WIRE_H_
#define AUDIO_TO_TACTILE_SRC_TWO_WIRE_H_

#include <nrf.h>
#include <stdbool.h>
#include <stdint.h>

#include "nrf_gpio.h"  // NOLINT(build/include)

#ifdef __cplusplus
extern "C" {
#endif

/* Start the I2C bus. The address to which i2c communicates
 * is specified here, and automatically used in read/write.
 */
void i2c_init(uint8_t scl_pin, uint8_t sda_pin,
              uint8_t i2c_peripheral_device_address);

/* Write a byte to a specific address. */
void i2c_write(uint8_t register_to_write, uint8_t data_to_write);

/* Read a byte from a specified address. */
uint8_t i2c_read(uint8_t register_to_read);

/* Read an array, which starts at a specific address. */
uint8_t* i2c_read_array(uint8_t first_register_to_read, uint8_t read_data_size);

/* Write an array. Array starts at data[0]. */
void i2c_write_array(const uint8_t data_to_write[], uint16_t write_data_size);

#ifdef __cplusplus
}
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_TWO_WIRE_H_ */
