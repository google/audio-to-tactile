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
 */

#include "two_wire.h"

static volatile uint32_t *pincfg_reg(uint32_t pin) {
  NRF_GPIO_Type *port = nrf_gpio_pin_port_decode(&pin);
  return &port->PIN_CNF[pin];
}

void i2c_write(uint8_t addr, uint8_t data) {
  static uint8_t tx_buf[2];
  NRF_TWIM0->SHORTS = TWIM_SHORTS_LASTTX_STOP_Msk;

  tx_buf[0] = addr;
  tx_buf[1] = data;
  NRF_TWIM0->TXD.MAXCNT = sizeof(tx_buf);
  NRF_TWIM0->TXD.PTR = (uint32_t)&tx_buf[0];

  NRF_TWIM0->EVENTS_STOPPED = 0;
  NRF_TWIM0->TASKS_STARTTX = 1;
  while (NRF_TWIM0->EVENTS_STOPPED == 0) {
  }
}

uint8_t i2c_read(uint8_t addr) {
  /* Arrays are static since the hardware uses them after the function exits. */
  static uint8_t tx_buf[1];
  static uint8_t rx_buf[1];
  /* Enable shortcuts that starts a read right after a write and sends a stop
   * condition after last TWI read.
   */
  NRF_TWIM0->SHORTS =
      TWIM_SHORTS_LASTTX_STARTRX_Msk | TWIM_SHORTS_LASTRX_STOP_Msk;
  /* Transmit the address. */
  tx_buf[0] = addr;
  NRF_TWIM0->TXD.MAXCNT = sizeof(tx_buf);
  NRF_TWIM0->TXD.PTR = (uint32_t)&tx_buf[0];

  NRF_TWIM0->RXD.MAXCNT = 1;  /* Max number of bytes per transfer. */
  NRF_TWIM0->RXD.PTR = (uint32_t)&rx_buf[0];  /* Point to RXD buffer. */

  NRF_TWIM0->EVENTS_STOPPED = 0;
  NRF_TWIM0->TASKS_STARTTX = 1;
  while (NRF_TWIM0->EVENTS_STOPPED == 0) {
  }

  return rx_buf[0];
}

uint8_t *i2c_read_array(uint8_t addr, uint8_t size) {
  /* Arrays are static since they are used after the function exits. */
  static uint8_t tx_buf[1];
  static uint8_t buffer[8];
  /* Enable shortcuts that starts a read right after a write and sends a stop
   * condition after last TWI read.
   */
  NRF_TWIM0->SHORTS =
      TWIM_SHORTS_LASTTX_STARTRX_Msk | TWIM_SHORTS_LASTRX_STOP_Msk;

  tx_buf[0] = addr;
  NRF_TWIM0->TXD.MAXCNT = sizeof(tx_buf);
  NRF_TWIM0->TXD.PTR = (uint32_t)&tx_buf[0];

  /* Load the data pointer into the TWI registers. */
  NRF_TWIM0->RXD.MAXCNT = size; /* Max number of bytes per transfer. */
  NRF_TWIM0->RXD.PTR = (uint32_t)&buffer; /* Point to RXD buffer. */

  /* Start read sequence. Note that it uses starttx, not start RX. */
  NRF_TWIM0->EVENTS_STOPPED = 0;
  NRF_TWIM0->TASKS_STARTTX = 1;

  /* Wait for the device to finish up. Currently, there is no time out so this
   * can go forever if somthing is wrong.
   */
  while (NRF_TWIM0->EVENTS_STOPPED == 0) {
  }

  return buffer;
}

void i2c_init(uint8_t scl, uint8_t sda, uint8_t device_addr) {
  *pincfg_reg(scl) =
      ((uint32_t)GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos) |
      ((uint32_t)GPIO_PIN_CNF_INPUT_Connect << GPIO_PIN_CNF_INPUT_Pos) |
      ((uint32_t)GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
      ((uint32_t)GPIO_PIN_CNF_DRIVE_S0D1 << GPIO_PIN_CNF_DRIVE_Pos) |
      ((uint32_t)GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);

  *pincfg_reg(sda) =
      ((uint32_t)GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos) |
      ((uint32_t)GPIO_PIN_CNF_INPUT_Connect << GPIO_PIN_CNF_INPUT_Pos) |
      ((uint32_t)GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
      ((uint32_t)GPIO_PIN_CNF_DRIVE_S0D1 << GPIO_PIN_CNF_DRIVE_Pos) |
      ((uint32_t)GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);

  NRF_TWIM0->PSEL.SCL = scl;
  NRF_TWIM0->PSEL.SDA = sda;

  NRF_TWIM0->ADDRESS = device_addr;
  NRF_TWIM0->FREQUENCY = TWIM_FREQUENCY_FREQUENCY_K400
                         << TWIM_FREQUENCY_FREQUENCY_Pos;
  NRF_TWIM0->SHORTS = 0;

  NRF_TWIM0->ENABLE = TWIM_ENABLE_ENABLE_Enabled << TWIM_ENABLE_ENABLE_Pos;
}

void i2c_write_array(const uint8_t *data, uint16_t size) {
  NRF_TWIM0->SHORTS = TWIM_SHORTS_LASTTX_STOP_Msk;

  NRF_TWIM0->TXD.MAXCNT = size;
  NRF_TWIM0->TXD.PTR = (uint32_t)data;

  NRF_TWIM0->EVENTS_STOPPED = 0;
  NRF_TWIM0->TASKS_STARTTX = 1;
  while (NRF_TWIM0->EVENTS_STOPPED == 0) {
  }
}
