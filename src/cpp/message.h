// Copyright 2021-2022 Google LLC
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
// Message class for representing serializable commands.
//
// `Message` is a unified representation for sending commands and information
// between devices (puck, sleeve, web app, ...) and also useful within a device
// to communicate between event handlers and the main loop.
//
// Although Messages may interact with hardware, this library should itself be
// hardware agnostic. This improves compatibility across devices and enables
// this code to be reused in other contexts, like emscripten and Android NDK.

#ifndef AUDIO_TO_TACTILE_SRC_CPP_MESSAGE_H_
#define AUDIO_TO_TACTILE_SRC_CPP_MESSAGE_H_

#include <stdint.h>

#include "cpp/constants.h"
#include "cpp/slice.h"
#include "cpp/settings.h"
#include "dsp/channel_map.h"
#include "tactile/envelope_tracker.h"
#include "tactile/tuning.h"

namespace audio_tactile {

// Types of messages. Append new op codes without changing old ones, as they
// ensure compatibility when communicating with external devices. Type values
// must be between 0 and 255.
enum class MessageType {
  kNone = 0,
  kTactor1Samples = 1,
  kTactor2Samples = 2,
  kTactor3Samples = 3,
  kTactor4Samples = 4,
  kTactor5Samples = 5,
  kTactor6Samples = 6,
  kTactor7Samples = 7,
  kTactor8Samples = 8,
  kTactor9Samples = 9,
  kTactor10Samples = 10,
  kTactor11Samples = 11,
  kTactor12Samples = 12,
  kAudioSamples = 13,
  kDisableAmplifiers = 14,
  kEnableAmplifiers = 15,
  kTemperature = 16,
  kAllTactorsSamples = 17,
  kTuning = 18,
  kTactilePattern = 19,
  kGetTuning = 20,
  kStreamDataStart = 21,
  kStreamDataStop = 22,
  kRequestMoreData = 23,  // Reserved for serial IO.
  kChannelMap = 24,
  kGetChannelMap = 25,
  kStatsRecord = 26,
  kChannelGainUpdate = 27,
  kBatteryVoltage = 28,
  kDeviceName = 29,
  kGetDeviceName = 30,
  kPrepareForBluetoothBootloading = 31,
  kFlashWriteStatus = 32,
  kOnConnectionBatch = 33,
  kGetOnConnectionBatch = 34,
  kCalibrateChannel = 35,
  kTactileExPattern = 36,
};

// Recipients of messages.
enum class MessageRecipient {
  kNone,
  kAll,
  kPuck,
  kSleeve,
  kSlimBoard,
  kConnectedBleDevice,
};

class Message {
 public:
  enum {
    // Number of header bytes.
    kHeaderSize = 4,
    // Max number of payload bytes.
    kMaxPayloadSize = 128,
    // Max message size = header size + max payload size.
    kMaxMessageSize = kHeaderSize + kMaxPayloadSize,
    // Start-of-frame code for first byte of serial header.
    kPacketStart = 200,
  };

  // Gets raw data pointer to the full message, including header.
  const uint8_t* data() const { return bytes_; }
  uint8_t* data() { return bytes_; }
  // Gets number of bytes in the message, considering the payload size.
  int size() const { return kHeaderSize + payload_size(); }

  // Set a serial header with start byte and recipient.
  void SetHeader(MessageRecipient recipient) {
    bytes_[0] = kPacketStart;
    bytes_[1] = static_cast<uint8_t>(recipient);
  }

  // Computes and sets checksum header for BLE. This method should be called
  // after all other data is set.
  void SetBleHeader();

  // Verifies the checksum for a message with BLE header. Returns true if valid.
  bool VerifyChecksum();

  // The message recipient. Note, this field is only valid if it has been set
  // with SetHeader().
  MessageRecipient recipient() const {
    return static_cast<MessageRecipient>(bytes_[1]);
  }

  // The message type, e.g. kTurnOnAmplifiers.
  MessageType type() const { return static_cast<MessageType>(bytes_[2]); }
  void set_type(MessageType type) { bytes_[2] = static_cast<uint8_t>(type); }

  // The message payload.
  Slice<const uint8_t> payload() const {
    return {bytes_ + kHeaderSize, payload_size()};
  }
  Slice<uint8_t> payload() { return {bytes_ + kHeaderSize, payload_size()}; }
  template <int kSize>
  void set_payload(Slice<const uint8_t, kSize> new_payload) {
    static_assert(kSize == kDynamic || kSize <= kMaxPayloadSize,
                  "Payload size must be less than kMaxPayloadSize");
    bytes_[3] = static_cast<uint8_t>(new_payload.size());
    payload().CopyFrom(new_payload);
  }

  // Methods for writing and reading messages of predefined types.
  // The Write* methods set the type and payload, but not the first two header
  // bytes. SetHeader() or SetBleHeader() should be called after one of these to
  // form a complete message.
  //
  // The Read* methods assume the type has already been checked, and only look
  // at the payload. In all cases they return true on success, false on failure.

  // Writes a kAudioSamples message of int16 audio samples.
  void WriteAudioSamples(Slice<const int16_t, kAdcDataSize> samples);
  // Reads the samples from a kAudioSamples message.
  bool ReadAudioSamples(Slice<int16_t, kAdcDataSize> samples) const;

  // Writes a kTemperature message to send thermistor temperature measurement.
  void WriteTemperature(float temperature_c);
  // Reads the temperature from a kTemperature message.
  bool ReadTemperature(float* temperature_c) const;

  // Writes a kBatteryVoltage message to send battery voltage.
  void WriteBatteryVoltage(float voltage);
  // Reads the voltage from a kBatteryVoltage message.
  bool ReadBatteryVoltage(float* voltage) const;

  // Writes a kTactor*Samples message, where `channel` is a one-based channel
  // index and `samples` is an array of PWM samples.
  void WriteSingleTactorSamples(
      int channel, Slice<const uint16_t, kNumPwmValues> samples);
  // Reads the samples and one-based channel from a kTactor*Samples message.
  bool ReadSingleTactorSamples(int* channel,
                               Slice<uint16_t, kNumPwmValues> samples) const;

  // Writes a kAllTactorsSamples message to send samples to all kNumTotalPwm
  // (= 12) channels to play each independently. The data is received as a byte
  // array and multiplied by 2. This allows sending all tactors data in one
  // serial packet, but data is truncated to 8-bit.
  void WriteAllTactorsSamples(
      Slice<const uint8_t, kNumTotalPwm * kNumPwmValues> samples);
  // Reads the samples from a kAllTactorsSamples message.
  bool ReadAllTactorsSamples(
      Slice<uint8_t, kNumTotalPwm * kNumPwmValues> samples) const;

  // Writes kTuning message of settings for all tuning knobs.
  void WriteTuning(const TuningKnobs& knobs);
  // Reads the tuning knobs from a kTuning message.
  bool ReadTuning(TuningKnobs* knobs) const;

  // Writes kTactilePattern message. The `pattern` arg must be a null-terminate
  // string with length <= kMaxTactilePatternLength (= 15).
  void WriteTactilePattern(const char* pattern);
  // Reads the tactile pattern from a kTactilePattern message.
  bool ReadTactilePattern(char* pattern) const;

  // Writes kChannelMap message. Supports up to 16 input channels and 12 output
  // channels. Each gain is encoded as 6 bits with value 0 representing gain = 0
  // and 1-63 mapping linearly to -18 to 0 dB.
  void WriteChannelMap(const ChannelMap& channel_map);
  // Reads the ChannelMap from a kChannelMap message. Checks that the mapping's
  // number of inputs and outputs agrees with `expected_inputs` and
  // `expected_outputs`. Set these args to -1 to allow any number of channels.
  bool ReadChannelMap(ChannelMap* channel_map, int expected_inputs = -1,
                      int expected_outputs = -1) const;

  // Writes a kChannelGainUpdate message, which both updates the channel gains
  // and plays calibration test tones on two channels. Gains are encoded with 6
  // bits per channel as done in the kChannelMap message.
  //
  // NOTE: kChannelGainUpdate sends the gains, but not the source mapping. A
  // kChannelMap message needs to be sent to update the source mapping.
  void WriteChannelGainUpdate(const ChannelMap& channel_map,
                              int test_channels[2]);
  // Reads a kChannelGainUpdate message. As in ReadChannelMap, it checks that
  // the number of inputs and outputs agrees with `expected_inputs` and
  // `expected_outputs`. Set these args to -1 to allow any number of channels.
  bool ReadChannelGainUpdate(ChannelMap* channel_map, int test_channels[2],
                             int expected_inputs = -1,
                             int expected_outputs = -1) const;

  // Reads a kCalibrateChannel message. Requires two calibration channels
  // and an amplitude for calibration playback.
  bool ReadCalibrateChannel(ChannelMap* channel_map,
                            int calibration_channels[2],
                            float* calibration_amplitude) const;

  // Writes a kStatsRecord message.
  void WriteStatsRecord(const EnvelopeTracker& envelope_tracker);

  // Writes a kDeviceName message. `device_name` must be a string with
  // length <= kMaxDeviceNameLength (= 16), not counting the null terminator.
  void WriteDeviceName(const char* device_name);
  // Reads a kDeviceName message into `device_name`. `device_name` must have
  // space for at least kMaxDeviceNameLength + 1 chars.
  bool ReadDeviceName(char* device_name) const;

  // Writes an on-connection batch message. This message is sent from the device
  // to app on connection, batching firmware build date, battery voltage,
  // temperature, and Settings.
  void WriteOnConnectionBatch(int firmware_build_date,
                              float battery_v,
                              float temperature_c,
                              const Settings& settings);
  // Reads an on-connection batch message, reading firmware build date, battery
  // voltage, temperature, and Settings. As in `ReadChannelMap`, the function
  // checks whether the message's number of inputs and outputs agree with
  // `expected_inputs` and `expected_outputs`. Setting these args to -1 allows
  // any number of channels. Returns true if all fields are read successfully.
  bool ReadOnConnectionBatch(int* firmware_build_date,
                             float* battery_v,
                             float* temperature_c,
                             Settings* settings,
                             int expected_inputs = -1,
                             int expected_outputs = -1);

  // Writes flash memory status after a write, where:
  // 0 - successful write
  // 1 - unknown error
  // 2 - flash is not formatted.
  void WriteFlashWriteStatus(int status);

  // Reads flash memory status after a write.
  bool ReadFlashWriteStatus(int* status) const;

  // Reads the extended-format pattern from a kTactileExPattern message.
  bool ReadTactileExPattern(Slice<uint8_t> pattern) const;

  // For the following messages, the payload is empty and are read simply by
  // checking `type()`.

  // Writes a kDisableAmplifiers message to send a command to disable all
  // amplifiers. This is a convenient way to silence tactors without changing
  // the flow of the firmware and reduce power consumption.
  void WriteDisableAmplifiers() {
    SetTypeAndPayload(MessageType::kDisableAmplifiers, {});
  }

  // Writes a kEnableAmplifiers message to enable the amplifiers.
  void WriteEnableAmplifiers() {
    SetTypeAndPayload(MessageType::kEnableAmplifiers, {});
  }

  // Writes kGetTuning message request to get the tuning knobs.
  void WriteGetTuning() {
    SetTypeAndPayload(MessageType::kGetTuning, {});
  }

  // Writes kGetChannelMap message request to get the channel map.
  void WriteGetChannelMap() {
    SetTypeAndPayload(MessageType::kGetChannelMap, {});
  }

  // Writes kGetDeviceName message request to get the device name.
  void WriteGetDeviceName() {
    SetTypeAndPayload(MessageType::kGetDeviceName, {});
  }

  // Writes a kStreamDataStart message.
  void WriteStreamDataStart() {
    SetTypeAndPayload(MessageType::kStreamDataStart, {});
  }

  // Writes a kStreamDataStop message.
  void WriteStreamDataStop() {
    SetTypeAndPayload(MessageType::kStreamDataStop, {});
  }

 private:
  int payload_size() const { return bytes_[3]; }

  void SetTypeAndPayload(MessageType type, Slice<const uint8_t> payload) {
    set_type(type);
    set_payload(payload);
  }

  uint16_t ComputeChecksum() const;

  uint8_t bytes_[kHeaderSize + kMaxPayloadSize];
};

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_CPP_MESSAGE_H_
