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
// BLE communication interface.
//
// Example use:
//   BleCom.Init("Tactile device", OnBleEvent);
//
//   void OnBleEvent() {
//     switch (BleCom.event()) {
//       case BleEvent::kMessageReceived:
//         HandleMessage(BleCom.rx_message());
//         break;
//       case BleEvent::kConnect:
//         Serial.println("BLE connected.");
//         break;
//     }
//   }

#ifndef AUDIO_TO_TACTILE_SRC_BLE_COM_H_
#define AUDIO_TO_TACTILE_SRC_BLE_COM_H_

#include <bluefruit.h>

#include "cpp/message.h"  // NOLINT(build/include)

namespace audio_tactile {

enum class BleEvent {
  kNone,
  kMessageReceived,
  kInvalidMessage,
  kConnect,
  kDisconnect,
};

class AudioTactileBleCom {
 public:
  AudioTactileBleCom(): event_fun_(nullptr), event_(BleEvent::kNone) {}

  // Initializes and begins BLE advertising.
  void Init(const char* device_name, void (*event_fun)());

  // Gets the most recent event.
  BleEvent event() const { return event_; }

  // Gets Message that will be transmitted.
  Message& tx_message() { return tx_message_; }
  // Sends tx_message over BLE UART.
  void SendTxMessage();

  // Gets Message that was most recently received.
  Message& rx_message() { return rx_message_; }

  friend void OnBleConnect(uint16_t connection_handle);
  friend void OnBleDisconnect(uint16_t connection_handle, uint8_t reason);
  friend void OnBleUartRx(uint16_t connection_handle);

 private:
  // Reads a message from ble_uart_ into rx_message_ and updates event_.
  void ReadFromBleUart();

  BLEUart ble_uart_;
  Message rx_message_;
  Message tx_message_;
  void (*event_fun_)();
  BleEvent event_;
};

extern AudioTactileBleCom BleCom;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_BLE_COM_H_
