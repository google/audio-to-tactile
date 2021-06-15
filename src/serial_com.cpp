// Copyright 2020-2021 Google LLC
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

#include "serial_com.h"  // NOLINT(build/include)

#include <string.h>

#include "nrf_gpio.h"   // NOLINT(build/include)
#include "nrf_uarte.h"  // NOLINT(build/include)

namespace audio_tactile {

AudioTactileSerialCom SerialCom;

AudioTactileSerialCom::AudioTactileSerialCom() {}

void AudioTactileSerialCom::InitSleeve(void (*event_fun)()) {
  InitInternal(kRxSleevePin, kTxSleevePin, event_fun);
}

void AudioTactileSerialCom::InitPuck(void (*event_fun)()) {
  InitInternal(kTxPuckPin, kRxPuckPin, event_fun);
}

void AudioTactileSerialCom::InitSlimBoard(void (*event_fun)()) {
  InitInternal(kTxSlimPin, kRxSlimPin, event_fun);
}

void AudioTactileSerialCom::Disable() {
  nrf_uarte_disable(NRF_UARTE0);
  NVIC_DisableIRQ(UARTE0_UART0_IRQn);
}

void AudioTactileSerialCom::Enable() {
  nrf_uarte_enable(NRF_UARTE0);
  NVIC_EnableIRQ(UARTE0_UART0_IRQn);
}

void AudioTactileSerialCom::SendTxMessage() {
  tx_message_.SetHeader(MessageRecipient::kSleeve);
  // To simplify the protocol, always transfer a full buffer.
  SendRaw({tx_message_.data(), Message::kMaxMessageSize});
}

void AudioTactileSerialCom::SendRaw(Slice<const uint8_t> buffer) {
  nrf_uarte_tx_buffer_set(NRF_UARTE0, buffer.data(), buffer.size());
  nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTTX);
}

void AudioTactileSerialCom::GetRaw(Slice<uint8_t> buffer) {
  buffer.CopyFrom(Slice<const uint8_t>(rx_message().data(), buffer.size()));
}

void AudioTactileSerialCom::InitInternal(uint32_t tx_pin, uint32_t rx_pin,
                                         void (*event_fun)()) {
  event_fun_ = event_fun;

  nrf_uarte_enable(NRF_UARTE0);
  nrf_uarte_baudrate_set(NRF_UARTE0, NRF_UARTE_BAUDRATE_1000000);
  nrf_uarte_txrx_pins_set(NRF_UARTE0, tx_pin, rx_pin);

  // Configure the uarte.
  nrf_uarte_config_t uarte_config;
  uarte_config.hwfc = NRF_UARTE_HWFC_DISABLED;
  uarte_config.parity = NRF_UARTE_PARITY_EXCLUDED;
  uarte_config.stop = NRF_UARTE_STOP_ONE;
  nrf_uarte_configure(NRF_UARTE0, &uarte_config);
  SetRxBuffer(0);

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

void AudioTactileSerialCom::SetRxBuffer(int i) {
  nrf_uarte_rx_buffer_set(NRF_UARTE0, rx_message_[i].data(),
                          Message::kMaxMessageSize);
}

// Interrupt handler for the serial port.
extern "C" {
void UARTE0_UART0_IRQHandler() { SerialCom.IrqHandler(); }
}

void AudioTactileSerialCom::IrqHandler() {
  // Triggered when new serial data is received into Easy DMA buffer.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_ENDRX)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_ENDRX);
    bytes_received_ = nrf_uarte_rx_amount_get(NRF_UARTE0);

    // Keep count of arriving data. If data keeps coming without a correct
    // header, trigger an error. Serial port is not always reliable.
    rx_counter_++;
    if (rx_counter_ > kTimeOutLoops) {
      event_ = SerialEvent::kTimeOutError;
      event_fun_();
    }

    // check if the packet byteID is correct, if so trigger callback.
    if (rx_message_[which_buffer_ready_].data()[0] == Message::kPacketStart) {
      event_ = SerialEvent::kMessageReceived;
      event_fun_();
      rx_counter_ = 0;
    }
  }

  // Triggered when data transmission is started.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_ENDTX)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_ENDTX);
  }

  // Triggered when there is an error.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_ERROR)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_ERROR);
    event_ = SerialEvent::kCommError;
    event_fun_();
  }

  // Triggered when receive is initiated.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_RXSTARTED)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_RXSTARTED);

    // RX is naturally double-buffered, so we need to take care of that.
    // See where the buffer pointer is. If it is at first buffer, set next
    // pointer to second buffer and vice-versa. Take the ready buffer and start
    // preparing it for next transmission.
    SetRxBuffer(which_buffer_ready_);
    which_buffer_ready_ ^= 1;
  }

  // Triggered when transmit is initiated.
  if (nrf_uarte_event_check(NRF_UARTE0, NRF_UARTE_EVENT_TXSTARTED)) {
    nrf_uarte_event_clear(NRF_UARTE0, NRF_UARTE_EVENT_TXSTARTED);
  }
}

}  // namespace audio_tactile
