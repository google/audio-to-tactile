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

#include "serial_puck_sleeve.h"  // NOLINT(build/include)

namespace audio_tactile {

SerialCom::SerialCom() {}

void SerialCom::InitializeSleeve() { initialize(kRxSleevePin, kTxSleevePin); }

void SerialCom::InitializePuck() { initialize(kTxPuckPin, kRxPuckPin); }

void SerialCom::Disable() {
  nrf_uarte_disable(NRF_UARTE0);
  NVIC_DisableIRQ(UARTE0_UART0_IRQn);
}

void SerialCom::Enable() {
  nrf_uarte_enable(NRF_UARTE0);
  NVIC_EnableIRQ(UARTE0_UART0_IRQn);
}

void SerialCom::SendDataRaw(uint8_t* data_to_send_buffer, uint8_t size) {
  nrf_uarte_tx_buffer_set(NRF_UARTE0, data_to_send_buffer, size);
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}

void SerialCom::SendMicrophoneData(int16_t* mic_data, uint8_t size) {
  // TODO In board_defs or main app make sure size is less or equal the
  // buffer size. Calculate how many bytes needs to be transmitted.
  uint8_t send_array_size = size * sizeof(int16_t) + kExtraPayload;

  // Set the packet header for microphone data transmission.
  tx_buffer_[0] = kPacketStart;
  tx_buffer_[1] = kDeviceId;
  tx_buffer_[2] = kLoadMicDataOpCode;
  tx_buffer_[3] = (uint8_t)size;

  // Convert 16 bit signed integers into byte array.
  memcpy(tx_buffer_ + kExtraPayload, mic_data, size * sizeof(int16_t));

  // Set the TX buffer.
  nrf_uarte_tx_buffer_set(NRF_UARTE0, tx_buffer_, send_array_size);

  // Start transmitting immediately.
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}

void SerialCom::GetRawData(uint8_t* destination_array, uint8_t size_in_bytes) {
  memcpy(destination_array, rx_buffer_[which_buffer_ready_], size_in_bytes);
}

void SerialCom::GetMicrophoneData(int16_t* destination_array, uint32_t size) {
  memcpy(destination_array, rx_buffer_[which_buffer_ready_] + kExtraPayload,
         size * sizeof(int16_t));
}

float SerialCom::GetThermistorTemperature() {
  float temperature;
  memcpy(&temperature, rx_buffer_[which_buffer_ready_] + kExtraPayload,
         sizeof(float));
  return temperature;
}

void SerialCom::SendThermistorTemperature(float temperature) {
  // Calculate how many bytes needs to be transmitted.
  uint8_t send_array_size = 1 * sizeof(float) + kExtraPayload;

  // Set the packet header for thermistor data transmission.
  tx_buffer_[0] = kPacketStart;
  tx_buffer_[1] = kDeviceId;
  tx_buffer_[2] = kLoadThermistor;
  tx_buffer_[3] = 2;

  // Convert float to byte array.
  memcpy(tx_buffer_ + kExtraPayload, &temperature, sizeof(float));

  // Set the TX buffer.
  nrf_uarte_tx_buffer_set(NRF_UARTE0, tx_buffer_, send_array_size);

  // Start transmitting immediately.
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}

void SerialCom::GetPlayOneTactorData(uint16_t* destination_array,
                                     uint32_t size) {
  memcpy(destination_array, rx_buffer_[which_buffer_ready_] + kExtraPayload,
         sizeof(uint16_t) * size);
}

void SerialCom::SendPlayOneTactorData(uint8_t pwm_channel, uint8_t size,
                                      uint16_t* pwm_values) {
  // Calculate how many bytes needs to be transmitted.
  uint8_t send_array_size = size * sizeof(int16_t) + kExtraPayload;

  // Set the packet header for microphone data transmission.
  tx_buffer_[0] = kPacketStart;
  tx_buffer_[1] = kDeviceId;
  tx_buffer_[2] = pwm_channel;
  tx_buffer_[3] = (uint8_t)size;

  // Convert 16 bit unsigned integers into byte array.
  memcpy(tx_buffer_ + kExtraPayload, pwm_values, sizeof(uint16_t) * size);

  // Set the TX buffer.
  nrf_uarte_tx_buffer_set(NRF_UARTE0, tx_buffer_, send_array_size);

  // Start transmitting immediately.
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}

// TODO const uint8_t* doesn't work in arduino Adafruit Feather. Figure
// out why it happens.
void SerialCom::SendPlayAllTactorsData(uint8_t* pwm_data, uint32_t size) {
  // Calculate how many bytes needs to be transmitted.
  uint8_t send_array_size = size * sizeof(uint8_t) + kExtraPayload;
  // Set the packet header for play all tactors data transmission.
  tx_buffer_[0] = kPacketStart;
  tx_buffer_[1] = kDeviceId;
  tx_buffer_[2] = kPlayAllChannels;
  tx_buffer_[3] = (uint8_t)size;
  // Copy the data into the transmit buffer.
  memcpy(tx_buffer_ + kExtraPayload, pwm_data, sizeof(uint8_t) * size);
  // Set the TX buffer.
  nrf_uarte_tx_buffer_set(NRF_UARTE0, tx_buffer_, send_array_size);
  // Start transmitting immediately.
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}
void SerialCom::GetPlayAllTactorsData(uint8_t* destination_array,
                                      uint32_t size) {
  memcpy(destination_array, rx_buffer_[which_buffer_ready_] + kExtraPayload,
         sizeof(uint8_t) * size);
}

void SerialCom::SendOutputGain(int value) {
  if (value < 0) {
    value = 0;
  }
  if (value > 255) {
    value = 255;
  }

  // Copy the data byte to the serial buffer.
  tx_buffer_[4] = value;
  send_serial_packet(kSetOutputGain, 1);
}

void SerialCom::SendDenoising(int value) {
  if (value < 0) {
    value = 0;
  }
  if (value > 255) {
    value = 255;
  }

  // Copy the data byte to the serial buffer.
  tx_buffer_[4] = value;
  send_serial_packet(kSetDenoising, 1);
}

void SerialCom::SendCompression(int value) {
  if (value < 0) {
    value = 0;
  }
  if (value > 255) {
    value = 255;
  }

  // Copy the data byte to the serial buffer.
  tx_buffer_[4] = value;
  send_serial_packet(kSetCompression, 1);
}

int SerialCom::GetTuningParameters() {
  int tuning_value;
  memcpy(&tuning_value, rx_buffer_[which_buffer_ready_] + kExtraPayload,
         sizeof(uint8_t));
  return tuning_value;
}

void SerialCom::SendDisableAmplifiers() {
  // There is no data in this case, just opcode.
  send_serial_packet(kTurnOffAmplifiers, 0);
}

void SerialCom::SendEnableAmplifiers() {
  // There is no data in this case, just opcode.
  send_serial_packet(kTurnOnAmplifiers, 0);
}

SerialCom PuckSleeveSerialPort;

void SerialCom::OnSerialDataReceived(void (*function)(void)) {
  callback_ = function;
}

void SerialCom::initialize(uint32_t tx_pin, uint32_t rx_pin) {
  nrf_uarte_enable(NRF_UARTE0);
  nrf_uarte_baudrate_set(NRF_UARTE0, NRF_UARTE_BAUDRATE_1000000);
  nrf_uarte_txrx_pins_set(NRF_UARTE0, tx_pin, rx_pin);
  nrf_uarte_configure(NRF_UARTE0, NRF_UARTE_PARITY_EXCLUDED,
                      NRF_UARTE_HWFC_DISABLED);

  // Set the receive buffer.
  nrf_uarte_rx_buffer_set(NRF_UARTE0, rx_buffer_[0],
                          kRxDataSize + kExtraPayload);

  // Enable Serial (UARTE) global interrupt.
  NVIC_DisableIRQ(UARTE0_UART0_IRQn);
  NVIC_ClearPendingIRQ(UARTE0_UART0_IRQn);
  NVIC_SetPriority(UARTE0_UART0_IRQn, kUarteIrqPriority);
  NVIC_EnableIRQ(UARTE0_UART0_IRQn);

  // Enable the specific interrupts.
  nrf_uarte_int_enable(NRF_UARTE0, NRF_UARTE_INT_ENDRX_MASK);
  nrf_uarte_int_enable(NRF_UARTE0, NRF_UARTE_INT_ENDTX_MASK);
  nrf_uarte_int_enable(NRF_UARTE0, NRF_UARTE_INT_ERROR_MASK);
  nrf_uarte_int_enable(NRF_UARTE0, NRF_UARTE_INT_RXSTARTED_MASK);
  nrf_uarte_int_enable(NRF_UARTE0, NRF_UARTE_INT_TXSTARTED_MASK);

  // This short automatically restarts receiver (RX) after a transmission is
  // done. Otherwise, we would need to set TASK_STARTRX = 1 after every time
  // after EVENT_ENDRX.
  nrf_uarte_shorts_enable(NRF_UARTE0, NRF_UARTE_SHORT_ENDRX_STARTRX);

  // Start the receiver.
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTRX);
}

void SerialCom::send_serial_packet(uint8_t op_code, uint8_t size_data) {
  // Set the packet size to the size of the serial buffer.
  uint8_t send_array_size = kTxDataSize + kExtraPayload;
  // Set the packet header with opcode.
  tx_buffer_[0] = kPacketStart;
  tx_buffer_[1] = kDeviceId;
  tx_buffer_[2] = op_code;
  tx_buffer_[3] = size_data;

  // Set the TX buffer.
  nrf_uarte_tx_buffer_set(NRF_UARTE0, tx_buffer_, send_array_size);

  // Start transmitting immediately.
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}

// Interrupt handler for the serial port.
extern "C" {
void UARTE0_UART0_IRQHandler() { PuckSleeveSerialPort.IrqHandler(); }
}

void SerialCom::IrqHandler() {
  // Triggered when new serial data is received into Easy DMA buffer.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_ENDRX)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_ENDRX);
    bytes_received_ = nrf_uarte_rx_amount_get(NRF_UARTE0);

    // Keep count of arriving data. If data keeps coming without a correct
    // header, trigger an error. Serial port is not always reliable.
    rx_counter++;
    if (rx_counter > kTimeOutLoops) {
      event_ = kTimeOutError;
      callback_();
    }

    // check if the packet byteID is correct, if so trigger callback.
    if (rx_buffer_[which_buffer_ready_][1] == kDeviceId) {
      event_ = rx_buffer_[which_buffer_ready_][2];
      callback_();
      rx_counter = 0;
    }
  }

  // Triggered when data transmission is started.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_ENDTX)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_ENDTX);
  }

  // Triggered when there is an error.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_ERROR)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_ERROR);
    event_ = kCommError;
    callback_();
  }

  // Triggered when receive is initiated.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_RXSTARTED)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_RXSTARTED);

    // RX is naturally double-buffered, so we need to take care of that.
    // See where the buffer pointer is. If it is at first buffer, set next
    // pointer to second buffer and vice-versa. Take the ready buffer and start
    // preparing it for next transmission.
    if (which_buffer_ready_ == 0) {
      nrf_uarte_rx_buffer_set(NRF_UARTE0, rx_buffer_[0],
                              kTxDataSize + kExtraPayload);
      which_buffer_ready_ = 1;
    } else if (which_buffer_ready_ == 1) {
      nrf_uarte_rx_buffer_set(NRF_UARTE0, rx_buffer_[1],
                              kTxDataSize + kExtraPayload);
      which_buffer_ready_ = 0;
    }
  }

  // Triggered when transmit is initiated.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_TXSTARTED)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_TXSTARTED);
  }
}

}  // namespace audio_tactile
