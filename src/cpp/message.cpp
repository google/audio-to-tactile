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

#include "cpp/message.h"  // NOLINT(build/include)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/serialize.h"  // NOLINT(build/include)

namespace audio_tactile {

void Message::SetBleHeader() {
  ::LittleEndianWriteU16(ComputeChecksum(), bytes_);
}

bool Message::VerifyChecksum() {
  return ::LittleEndianReadU16(bytes_) == ComputeChecksum();
}

uint16_t Message::ComputeChecksum() const {
  return ::Fletcher16(bytes_ + 2, (kHeaderSize - 2) + payload_size(),
                      /*init=*/1);
}

void Message::WriteAudioSamples(Slice<const int16_t, kAdcDataSize> samples) {
  SetTypeAndPayload(MessageType::kAudioSamples, samples.bytes());
}
bool Message::ReadAudioSamples(Slice<int16_t, kAdcDataSize> samples) const {
  return samples.bytes().CopyFrom(payload());
}

void Message::WriteTemperature(float temperature_c) {
  uint8_t bytes[4];
  ::LittleEndianWriteF32(temperature_c, bytes);
  SetTypeAndPayload(MessageType::kTemperature, Slice<uint8_t, 4>(bytes));
}
bool Message::ReadTemperature(float* temperature_c) const {
  *temperature_c = ::LittleEndianReadF32(payload().data());
  return true;
}

void Message::WriteBatteryVoltage(float voltage) {
  uint8_t bytes[4];
  ::LittleEndianWriteF32(voltage, bytes);
  SetTypeAndPayload(MessageType::kBatteryVoltage, Slice<uint8_t, 4>(bytes));
}
bool Message::ReadBatteryVoltage(float* voltage) const {
  *voltage = ::LittleEndianReadF32(payload().data());
  return true;
}

void Message::WriteSingleTactorSamples(
    int channel, Slice<const uint16_t, kNumPwmValues> samples) {
  SetTypeAndPayload(static_cast<MessageType>(channel), samples.bytes());
}
bool Message::ReadSingleTactorSamples(
    int* channel, Slice<uint16_t, kNumPwmValues> samples) const {
  *channel = static_cast<int>(type());
  return samples.bytes().CopyFrom(payload());
}

void Message::WriteAllTactorsSamples(
    Slice<const uint8_t, kNumTotalPwm * kNumPwmValues> samples) {
  SetTypeAndPayload(MessageType::kAllTactorsSamples, samples);
}
bool Message::ReadAllTactorsSamples(
      Slice<uint8_t, kNumTotalPwm * kNumPwmValues> samples) const {
  return samples.CopyFrom(payload());
}

void Message::WriteTuning(const TuningKnobs& knobs) {
  SetTypeAndPayload(MessageType::kTuning,
                    Slice<const uint8_t, kNumTuningKnobs>(knobs.values));
}
bool Message::ReadTuning(TuningKnobs* knobs) const {
  return Slice<uint8_t, kNumTuningKnobs>(knobs->values).CopyFrom(payload());
}

void Message::WriteTactilePattern(const char* pattern) {
  const int length =
      std_shim::min<int>(kMaxTactilePatternLength, strlen(pattern));
  // Null terminator is not stored.
  SetTypeAndPayload(MessageType::kTactilePattern,
                    Slice<const char>(pattern, length).bytes());
}
bool Message::ReadTactilePattern(char* pattern) const {
  const int length =
      std_shim::min<int>(kMaxTactilePatternLength, payload_size());
  // Within the Message, we know the pattern length from the payload size field,
  // so we don't store the \0 in the Message. However, outside of Message,
  // patterns are C strings, which know their length based on the \0 terminator.
  // So append a \0 when deserializing it from the Message.
  pattern[length] = '\0';  // Append null terminator.
  return Slice<char>(pattern, length).bytes().CopyFrom(payload());
}

namespace {
// Max supported number of input channels in a channel map.
constexpr int kMaxInputChannels = 16;
// Max supported number of output channels in a channel map.
constexpr int kMaxOutputChannels = 12;

uint_fast32_t EncodeChannelGain(float gain) {
  return static_cast<uint_fast32_t>(ChannelGainToControlValue(gain));
}

// Writes channel map gains to `dest`, 6 bits per channel. Returns the position
// of `dest` after writing.
uint8_t* SerializeChannelMapGains(const ChannelMap& channel_map,
                                  uint8_t* dest) {
  const int num_out = std_shim::min<int>(12, channel_map.num_output_channels);
  const float* gains = channel_map.gains;
  // Each loop iteration handles 4 channels.
  for (int c = 0; c < num_out; c += 4, dest += 3) {
    const uint_fast32_t pack24 = EncodeChannelGain(gains[c])
                               | EncodeChannelGain(gains[c + 1]) << 6
                               | EncodeChannelGain(gains[c + 2]) << 12
                               | EncodeChannelGain(gains[c + 3]) << 18;
    dest[0] = static_cast<uint8_t>(pack24);
    dest[1] = static_cast<uint8_t>(pack24 >> 8);
    dest[2] = static_cast<uint8_t>(pack24 >> 16);
  }
  return dest;
}

// Computes the payload size of a ChannelMap or ChannelGainUpdate message.
// `num_out` is the number of output channels. `is_gain_update` indicates the
// type of message: false => ChannelMap, true => ChannelGainUpdate.
//
// Size computation for a ChannelMap message =
//  one byte for the number of input and output channels
//  + `3 * ceil(num_out / 4)` bytes for the channel gains
//  + `ceil(num_out / 2)` bytes for the sources
//
// Size computation for a ChannelGainUpdate message =
//  one byte for the number of input and output channels
//  + `3 * ceil(num_out / 4)` bytes for the channel gains
//  + one byte for the test channel indices
int ComputeChannelMapOrGainUpdatePayloadSize(int num_out, bool is_gain_update) {
  return 1 + 3 * ((num_out + 3) / 4)
    + (is_gain_update ? 1 : ((num_out + 1) / 2));
}

// Read number of input and output channels and check payload size.
bool DeserializeNumChannelsAndCheckSize(Slice<const uint8_t> payload,
                                        bool is_gain_update,
                                        int expected_inputs,
                                        int expected_outputs,
                                        ChannelMap* channel_map) {
  if (payload.empty()) { return false; }
  const int num_in = 1 + ((payload[0] + 15) & 15);
  const int num_out = payload[0] >> 4;
  const int expected_size =
    ComputeChannelMapOrGainUpdatePayloadSize(num_out, is_gain_update);
  if (payload.size() != expected_size ||
      (expected_inputs >= 0 && num_in != expected_inputs) ||
      (expected_outputs >= 0 && num_out != expected_outputs)) {
    return false;
  }
  channel_map->num_input_channels = num_in;
  channel_map->num_output_channels = num_out;
  return true;
}

// Read channel map source mapping. Returns final position of `src`.
const uint8_t* DeserializeChannelMapSources(const uint8_t* src,
                                            ChannelMap* channel_map) {
  const int num_in = channel_map->num_input_channels;
  const int num_out = channel_map->num_output_channels;
  int* sources = channel_map->sources;
  for (int c = 0; c < num_out; c += 2, ++src) {
    sources[c] = std_shim::min<int>(*src & 15, num_in - 1);
    sources[c + 1] = std_shim::min<int>(*src >> 4, num_in - 1);
  }
  return src;
}

// Read channel map gains. Returns final position of `src`.
const uint8_t* DeserializeChannelMapGains(const uint8_t* src,
                                          ChannelMap* channel_map) {
  const int num_out = channel_map->num_output_channels;
  float* gains = channel_map->gains;
  for (int c = 0; c < num_out; c += 4, src += 3) {
    const uint_fast32_t pack24 =
        static_cast<uint_fast32_t>(src[0])
        | static_cast<uint_fast32_t>(src[1]) << 8
        | static_cast<uint_fast32_t>(src[2]) << 16;
    gains[c] = ChannelGainFromControlValue(pack24 & 63);
    gains[c + 1] = ChannelGainFromControlValue((pack24 >> 6) & 63);
    gains[c + 2] = ChannelGainFromControlValue((pack24 >> 12) & 63);
    gains[c + 3] = ChannelGainFromControlValue((pack24 >> 18) & 63);
  }
  return src;
}

// Writes channel map to `dest`. Returns the position of `dest` after writing.
uint8_t* SerializeChannelMap(const ChannelMap& channel_map, uint8_t* dest) {
  const int num_in =
      std_shim::min<int>(kMaxInputChannels, channel_map.num_input_channels);
  const int num_out =
      std_shim::min<int>(kMaxOutputChannels, channel_map.num_output_channels);

  // Write number of input and output channels in the first byte.
  *dest++ = static_cast<uint8_t>((num_in & 15) | num_out << 4);
  // Write source mapping, 4 bits per channel.
  for (int c = 0; c < num_out; c += 2, ++dest) {
    *dest = static_cast<uint8_t>((channel_map.sources[c] & 15)
                               | (channel_map.sources[c + 1] & 15) << 4);
  }
  // Write channel gains, 6 bits per channel.
  return SerializeChannelMapGains(channel_map, dest);
}

// Reads channel map from `payload`. Returns true on success.
bool DeserializeChannelMap(Slice<const uint8_t> payload,
                           int expected_inputs,
                           int expected_outputs,
                           ChannelMap* channel_map) {
  if (DeserializeNumChannelsAndCheckSize(payload, /*is_gain_update=*/false,
        expected_inputs, expected_outputs, channel_map)) {
    const uint8_t* src = payload.data() + 1;
    src = DeserializeChannelMapSources(src, channel_map);
    DeserializeChannelMapGains(src, channel_map);
    return true;
  }
  return false;
}

// Writes channel gain update to `dest`. Returns final position of `dest`.
uint8_t* SerializeChannelGainUpdate(
    const ChannelMap& channel_map, const int* test_channels, uint8_t* dest) {
  const int num_in =
      std_shim::min<int>(kMaxInputChannels, channel_map.num_input_channels);
  const int num_out =
      std_shim::min<int>(kMaxOutputChannels, channel_map.num_output_channels);

  // Write number of input and output channels in the first byte.
  *dest++ = static_cast<uint8_t>((num_in & 15) | num_out << 4);
  // Write the two test channel indices, 4 bits for each.
  *dest++ = static_cast<uint8_t>((test_channels[0] & 15)
                                | test_channels[1] << 4);
  // Write channel gains, 6 bits per channel.
  return SerializeChannelMapGains(channel_map, dest);
}

// Reads ChannelGainUpdate message from `payload`. Returns true on success.
bool DeserializeChannelGainUpdate(Slice<const uint8_t> payload,
                                  int expected_inputs,
                                  int expected_outputs,
                                  ChannelMap* channel_map,
                                  int* test_channels) {
  if (DeserializeNumChannelsAndCheckSize(payload, /*is_gain_update=*/true,
        expected_inputs, expected_outputs, channel_map)) {
    const uint8_t* src = payload.data() + 1;
    test_channels[0] = *src & 15;  // Read the two test channel indices.
    test_channels[1] = *src >> 4;
    DeserializeChannelMapGains(src + 1, channel_map);
    return true;
  }
  return false;
}
}  // namespace

void Message::WriteChannelMap(const ChannelMap& channel_map) {
  uint8_t* start = bytes_ + kHeaderSize;
  uint8_t* end = SerializeChannelMap(channel_map, bytes_ + kHeaderSize);
  bytes_[3] = static_cast<uint8_t>(end - start);
  set_type(MessageType::kChannelMap);
}

bool Message::ReadChannelMap(ChannelMap* channel_map, int expected_inputs,
                             int expected_outputs) const {
  return DeserializeChannelMap(
      payload(), expected_inputs, expected_outputs, channel_map);
}

void Message::WriteChannelGainUpdate(
    const ChannelMap& channel_map, int test_channels[2]) {
  uint8_t* start = bytes_ + kHeaderSize;
  uint8_t* end = SerializeChannelGainUpdate(
      channel_map, test_channels, bytes_ + kHeaderSize);
  bytes_[3] = static_cast<uint8_t>(end - start);
  set_type(MessageType::kChannelGainUpdate);
}

bool Message::ReadChannelGainUpdate(ChannelMap* channel_map,
                                    int test_channels[2], int expected_inputs,
                                    int expected_outputs) const {
  return DeserializeChannelGainUpdate(
      payload(), expected_inputs, expected_outputs, channel_map, test_channels);
}

bool Message::ReadCalibrateChannel(ChannelMap* channel_map,
                                    int calibration_channels[2],
                                    float* calibration_amplitude) const {
  if (payload_size() != 4) { return false; }
  const uint8_t* src = payload().data();

  // Read the two test channel indices.
  calibration_channels[0] = *src & 15;
  calibration_channels[1] = *src >> 4;

  // validate that these indices are < channel_map->num_output_channels
  if (calibration_channels[0] >= channel_map->num_output_channels ||
      calibration_channels[1] >= channel_map->num_output_channels) {
    return false;
  }

  // Read and set the new gain.
  channel_map->gains[calibration_channels[1]] =
      ChannelGainFromControlValue(*++src);

  // Read playback amplitude, converting uint16 value to a float in [0, 1].
  *calibration_amplitude = ::LittleEndianReadU16(++src) / 65535.0f;

  return true;
}


void Message::WriteStatsRecord(const EnvelopeTracker& envelope_tracker) {
  EnvelopeTrackerGetRecord(&envelope_tracker, bytes_ + kHeaderSize);
  bytes_[3] = kEnvelopeTrackerRecordBytes;
  set_type(MessageType::kStatsRecord);
}

void Message::WriteDeviceName(const char* device_name) {
  const int length =
      std_shim::min<int>(kMaxDeviceNameLength, strlen(device_name));
  // Null terminator is not stored.
  SetTypeAndPayload(MessageType::kDeviceName,
                    Slice<const char>(device_name, length).bytes());
}

bool Message::ReadDeviceName(char* device_name) const {
  const int length = std_shim::min<int>(kMaxDeviceNameLength, payload_size());
  // Same logic as in ReadTactilePattern: Within the Message, we know the name
  // length from the payload size field, so we don't store the \0 in the
  // Message. However, outside of Message, the name is a C string, so append a
  // \0 when deserializing it from the Message.
  device_name[length] = '\0';  // Append null terminator.
  return Slice<char>(device_name, length).bytes().CopyFrom(payload());
}

void Message::WriteFlashWriteStatus(int status) {
  uint8_t status_byte[1];
  status_byte[0] = static_cast<uint8_t>(status);
  SetTypeAndPayload(MessageType::kFlashWriteStatus,
                    Slice<uint8_t, 1>(status_byte));
}

bool Message::ReadFlashWriteStatus(int* status) const {
  *status = ::LittleEndianReadF32(payload().data());
  return true;
}

namespace {
constexpr int kOnConnectionBatchFixedFieldsSize = 12;
}  // namespace

void Message::WriteOnConnectionBatch(int firmware_build_date,
                                     float battery_v,
                                     float temperature_c,
                                     const Settings& settings) {
  uint8_t* dest = bytes_ + kHeaderSize;
  // Write firmware build date, battery voltage, and temperature, 4 bytes each.
  ::LittleEndianWriteU32(firmware_build_date, dest);
  ::LittleEndianWriteF32(battery_v, dest + 4);
  ::LittleEndianWriteF32(temperature_c, dest + 8);
  dest += kOnConnectionBatchFixedFieldsSize;

  // The following data are formatted in a block structure beginning with a byte
  // indicating the size of the block followed by the block data. This allows
  // the reader to skip individual blocks, in case of version mismatch or other
  // error, rather than failing to read the message entirely.

  // Write device name length followed by the device name.
  int block_size =
      std_shim::min<int>(kMaxDeviceNameLength, strlen(settings.device_name));
  *dest++ = block_size;
  memcpy(dest, settings.device_name, block_size);
  dest += block_size;

  // Write number of tuning knobs followed by the knob values.
  block_size = kNumTuningKnobs;
  *dest++ = block_size;
  memcpy(dest, settings.tuning.values, block_size);
  dest += block_size;

  // Write channel map serialized size followed by the channel map.
  block_size = ComputeChannelMapOrGainUpdatePayloadSize(
    settings.channel_map.num_output_channels, /*is_gain_update=*/false);
  *dest++ = block_size;
  SerializeChannelMap(settings.channel_map, dest);
  dest += block_size;

  bytes_[3] = dest - (bytes_ + kHeaderSize);
  set_type(MessageType::kOnConnectionBatch);
}

bool Message::ReadOnConnectionBatch(int* firmware_build_date,
                                    float* battery_v,
                                    float* temperature_c,
                                    Settings* settings,
                                    int expected_inputs,
                                    int expected_outputs) {
  if (payload_size() < kOnConnectionBatchFixedFieldsSize) { return false; }
  const uint8_t* src = payload().data();
  const uint8_t* end = src + payload_size();
  bool success = true;
  // Read firmware build date, battery voltage, and temperature.
  *firmware_build_date = ::LittleEndianReadU32(src);
  *battery_v = ::LittleEndianReadF32(src + 4);
  *temperature_c = ::LittleEndianReadF32(src + 8);
  src += kOnConnectionBatchFixedFieldsSize;

  // Read the device name.
  if (end - src < 1) { return false; }
  int block_size = *src++;
  if (end - src < block_size) { return false; }

  {
    const int length = std_shim::min<int>(kMaxDeviceNameLength, block_size);
    memcpy(settings->device_name, src, length);
    settings->device_name[length] = '\0';  // Append null terminator.
  }

  src += block_size;

  // Read the tuning knobs.
  if (end - src < 1) { return false; }
  block_size = *src++;
  if (end - src < block_size) { return false; }

  if (block_size == kNumTuningKnobs) {
    memcpy(settings->tuning.values, src, kNumTuningKnobs);
  } else {
    success = false;
  }

  src += block_size;

  // Read the channel map.
  if (end - src < 1) { return false; }
  block_size = *src++;
  if (end - src < block_size) { return false; }

  success &= DeserializeChannelMap(Slice<const uint8_t>(src, block_size),
                                   expected_inputs,
                                   expected_outputs,
                                   &settings->channel_map);
  src += block_size;
  return success;
}

bool Message::ReadTactileExPattern(Slice<uint8_t> pattern) const {
  if (pattern.size() < payload_size() + 1) { return false; }
  pattern.CopyFrom(payload());
  pattern[payload_size()] = 0;  // Append null terminator (end op).
  return true;
}

}  // namespace audio_tactile
