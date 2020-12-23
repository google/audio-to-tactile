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
// PuckSleeveSerialPort.InitializeSleeve();
// PuckSleeveSerialPort.OnSerialDataReceived(on_new_serial_data);
//
// void on_new_serial_data() {
//   switch (PuckSleeveSerialPort.GetEvent()) {
//     case kLoadMicDataOpCpde:
//      // Trigger the audio processing task.
//       break;
//     case kLoadTactorL1:
//       PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, 8);
//       SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 0, 0);
//       break;
//     case kTurnOffAmplifiers:
//       SleeveTactors.DisableAmplifiers();
//       break;
//   }
// }

#ifndef AUDIO_TO_TACTILE_SRC_SERIAL_PUCK_SLEEVE_H_
#define AUDIO_TO_TACTILE_SRC_SERIAL_PUCK_SLEEVE_H_

#include <stdint.h>
#include <string.h>

#include "nrf_gpio.h"   // NOLINT(build/include)
#include "nrf_uarte.h"  // NOLINT(build/include)

namespace audio_tactile {

// Hardware constants.
enum {
  kUarteIrqPriority = 7,  // lowest priority
  kRxPuckPin = 40,        // P1.8,
  kTxPuckPin = 8,         // P0.8
  kRxSleevePin = 46,      // P1.14
  kTxSleevePin = 47       // P1.15
};

// Communication constants.
enum {
  kPacketStart = 200,
  kDeviceId = 201,
  kExtraPayload = 4,
  kTimeOutLoops = 400,
  // Operational (Op) codes.
  kNoOpCode = 0,
  kLoadTactorL1 = 1,
  kLoadTactorR1 = 2,
  kLoadTactorL2 = 3,
  kLoadTactorR2 = 4,
  kLoadTactorL3 = 5,
  kLoadTactorR3 = 6,
  kLoadTactorL4 = 7,
  kLoadTactorR4 = 8,
  kLoadTactorL5 = 9,
  kLoadTactorR5 = 10,
  kLoadTactorL6 = 11,
  kLoadTactorR6 = 12,
  kLoadMicDataOpCode = 13,
  kTurnOffAmplifiers = 14,
  kTurnOnAmplifiers = 15,
  kLoadThermistor = 16,
  kPlayAllChannels = 17,
  kSetOutputGain = 18,
  kSetDenoising = 19,
  kSetCompression = 20,
  kStreamDataStart = 21,
  kStreamDataStop = 22,
  kRequestMoreData = 23,
  kTimeOutError = 200,
  kCommError = 201,
};

// Easy DMA rx and tx buffer size in bytes without the header.
// The Rx/Tx buffer sizes (and vice versa) on the puck and sleeve should be the
// same size.
enum { kTxDataSize = 128, kRxDataSize = 128 };

class SerialCom {
 public:
  SerialCom();

  // Initialize the serial comunications for the sleeve.
  // The sleeve and the puck use different pins this different
  // initalizations. This function starts the lister (interrupt handler)
  // as well.
  void InitializeSleeve();

  // Initialize serial communications for the puck.
  void InitializePuck();

  // Stop the callbacks, disables the serial port.
  void Disable();

  // Start the callbacks, enable the serial port.
  void Enable();

  // This function is called when new data arrived over a serial port.
  // The callback is only triggered if the header id is correct.
  void IrqHandler();

  // Retrieve the raw received byte array. This byte array includes the
  // header as well. The size is limited to the kRxDataSize + kExtraPayload
  // Currently it is set to 132 bytes. The maximum is 255 bytes, as limited by
  // RXD.MAXCNT uint8_t register.
  void GetRawData(uint8_t* destination_array, uint8_t size_in_bytes);

  // Send the byte array without any formatting.
  void SendDataRaw(uint8_t* data_to_send_buffer, uint8_t size);

  // Sets a callback function when new data is received.
  void OnSerialDataReceived(void (*function)(void));

  // Retrieve the microphone data. The data is automatically coverted to
  // signed integer array. Header is stripped off the data.
  // The rx_buffer_ holds 64 microphone samples. Larger number will cause buffer
  // overflow, and undefined Easy DMA behavior. The maximum number of samples
  // is 125. This can be set by increasing kRxDataSize or/and kTxDataSize.
  void GetMicrophoneData(int16_t* destination_array,
                         uint32_t mic_number_of_samples);

  // format the data and send as a microphone array.
  // The 16-bit microphone array is converted into byte array.
  // A header is added to the array (ID byte, op code, and data size);
  void SendMicrophoneData(int16_t* mic_data, uint8_t mic_number_of_samples);

  // Check how many bytes are received.
  uint32_t GetNumberReceivedBytes() { return bytes_received_; }

  // Send data, which will be played on a specific tactor.
  void SendPlayOneTactorData(uint8_t pwm_channel, uint8_t size, uint16_t* data);

  // Receive the tactor data.
  void GetPlayOneTactorData(uint16_t* destination_array, uint32_t size);

  // Send data to 10 channels to play each independently.
  // The data is received as a byte array and multiplied by 2. This allows
  // sending all tactors data in one serial packet, but data is truncated to
  // 8-bit. The size is number of pwm samples in byte format. Currently, 8 pwm
  // samples are sent for each channel for the total of 80 samples. Each of
  // the 8 samples plays after previous one is finished.
  void SendPlayAllTactorsData(uint8_t* pwm_data, uint32_t size);

  // Get the data for all tactors.
  void GetPlayAllTactorsData(uint8_t* destination_array, uint32_t size);

  // Send output gain parameter. The value is a byte: 0 to 255.
  void SendOutputGain(int value);

  // Send denoising parameter. The value is a byte: 0 to 255.
  void SendDenoising(int value);

  // Send compression parameter. The value is a byte: 0 to 255.
  void SendCompression(int value);

  // Get tuning parameter value from 0 to 255.
  // The state machine in the main loop, should keep track of which parameter is
  // received. The packet's event type indicates which parameter (e.g.
  // kSetOutputGain) is associated with the returned value.
  int GetTuningParameters();

  // Send the thermistor temperature measurement.
  void SendThermistorTemperature(float temperature);

  // Receive and parse the thermistor temeprture measrument.
  float GetThermistorTemperature();

  // Send a command to disable all amplifiers on the sleeve.
  // This is a convenient way to silence tactors without changing the flow of
  // the firmware. The pwm signals are still continue to be send to amplifiers.
  void SendDisableAmplifiers();

  // Send a command to enable all amplifiers.
  void SendEnableAmplifiers();

  // Get the retrieved op code from the packet.
  uint8_t GetEvent() const { return event_; }

 private:
  // Callback for the interrupt.
  void (*callback_)(void);

  // Storing interrupt event. The events are described in the op code enum
  // table.
  uint8_t event_;

  // Internal initialization helper.
  void initialize(uint32_t tx_pin, uint32_t rx_pin);

  // UARTE buffers for EasyDMA. RX is double buffered.
  uint8_t rx_buffer_[2][kRxDataSize + kExtraPayload];
  uint8_t tx_buffer_[kTxDataSize + kExtraPayload];

  // Which buffer is currently ready to read.
  uint8_t which_buffer_ready_;

  // Stores number of received bytes.
  uint32_t bytes_received_;

  // Serial timeout error counter. If serial data that can't be parsed keep
  // coming, something is wrong, unless send/getDataRaw() is used.
  int rx_counter = 0;

  // Helper function to set the packet header and start serial transmission.
  // The size is the amount of data bytes. The overall packet size is set to the
  // buffer size. Unused data will be sent as zeros. The size argument helps to
  // parse data on the other size.
  void send_serial_packet(uint8_t op_code, uint8_t size);
};

extern SerialCom PuckSleeveSerialPort;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_SERIAL_PUCK_SLEEVE_H_
