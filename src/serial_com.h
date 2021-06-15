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
//
//
// Arduino-compatible library for serial interface.
//
// This driver provides Arduino-compatabile library for serial interface, also
// called universal serial transmitter-receiver with Easy DMA (UARTE). HAL
// (Hardware abstraction layer) functions are used. The Nordic driver is not
// used. This driver allows to send and receive raw data, as well as predefined
// packets such as a microphone or thermistor data.
//
// The following protocol is used:
// [0] <kPacketStart>
// [1] <kDeviceId> - Multiple serial devices can be connected.
// [2] <op code>  - The code tells sleeve what function to perform.
// [3] <data size n> - The size of the data packet.
// [4]....[n] <data> - data as byte array (uint8_t).
//
// The buffer size is fixed, thus if we are sending less data, the rest has to
// be padded with zeros. The fixed buffer avoids dynamic allocation and
// simplifies communication protocol.
//
// For the fastest performance data switch statement and an interrupt should be
// used. The switch statement determines what action to take based on the data.
// The interrupt should be handled something like this:
//
// // Initialize serial port.
// SerialCom.InitSleeve(OnSerialEvent);
//
// void OnSerialEvent() {
//   switch (SerialCom.event()) {
//     case kMessageReceived:
//       HandleMessage(SerialCom.rx_message());
//       break;
//     case kTimeOutError:
//       // Handle error.
//       break;
//   }
// }

#ifndef AUDIO_TO_TACTILE_SRC_SERIAL_PUCK_SLEEVE_H_
#define AUDIO_TO_TACTILE_SRC_SERIAL_PUCK_SLEEVE_H_

#include "cpp/message.h"  // NOLINT(build/include)

namespace audio_tactile {

enum class SerialEvent {
  kNone,
  kMessageReceived,
  kTimeOutError,
  kCommError,
};

class AudioTactileSerialCom {
 public:
  enum {
    kTimeOutLoops = 4,
    // Hardware constants.
    kUarteIrqPriority = 7,  // lowest priority
    kRxPuckPin = 40,        // P1.8,
    kTxPuckPin = 8,         // P0.8
    kRxSleevePin = 46,      // P1.14
    kTxSleevePin = 47,      // P1.15
    kRxSlimPin = 26,        // P0.04
    kTxSlimPin = 4          // P0.26
  };

  AudioTactileSerialCom();

  // Initializes serial comunications for the sleeve. The sleeve and the puck
  // use different pins this different initalizations. This function starts the
  // lister (interrupt handler) as well.
  void InitSleeve(void (*event_fun)());

  // Initializes serial communications for the puck.
  void InitPuck(void (*event_fun)());

  // Initializes serial communications for the slim board.
  void InitSlimBoard(void (*event_fun)());

  // Stops the callbacks, disables the serial port.
  void Disable();

  // Starts the callbacks, enable the serial port.
  void Enable();

  // Gets the most recent event.
  SerialEvent event() const { return event_; }

  // Number of bytes received in the most recent message.
  int bytes_received() const { return bytes_received_; }

  // Gets Message that will be transmitted.
  Message& tx_message() { return tx_message_; }
  // Sends tx_message over serial UART.
  void SendTxMessage();

  // Gets Message that was most recently received.
  Message& rx_message() { return rx_message_[which_buffer_ready_]; }

  // Sends byte array without any formatting.
  void SendRaw(Slice<const uint8_t> buffer);

  // Retrieves the raw received byte array. This byte array includes the
  // header as well. The size is limited to Message::kMaxMessageSize which is
  // currently 132 bytes. The maximum is 255 bytes, as limited by RXD.MAXCNT
  // uint8_t register.
  void GetRaw(Slice<uint8_t> buffer);

  // This function is called when new data arrived over a serial port.
  // The callback is only triggered if the header id is correct.
  void IrqHandler();

 private:
  // Internal initialization helper.
  void InitInternal(uint32_t tx_pin, uint32_t rx_pin, void (*event_fun)());

  // Sets the receiving buffer to rx_message_[i].
  void SetRxBuffer(int i);

  // UARTE buffers for EasyDMA. RX is double buffered.
  Message rx_message_[2];
  Message tx_message_;

  // Callback for the interrupt.
  void (*event_fun_)();

  // Storing interrupt event. The events are described in the op code enum
  // table.
  SerialEvent event_;

  // Which buffer is currently ready to read.
  int which_buffer_ready_;

  // Stores number of received bytes.
  int bytes_received_;

  // Serial timeout error counter. If serial data that can't be parsed keeps
  // coming, something is wrong, unless send/getDataRaw() is used.
  int rx_counter_ = 0;
};

extern AudioTactileSerialCom SerialCom;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_SERIAL_PUCK_SLEEVE_H_
