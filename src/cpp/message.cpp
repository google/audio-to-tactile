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

#include "cpp/message.h"  // NOLINT(build/include)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/fast_fun.h"  // NOLINT(build/include)
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

void Message::WriteTemperature(float temperature) {
  uint8_t bytes[4];
  ::LittleEndianWriteF32(temperature, bytes);
  SetTypeAndPayload(MessageType::kTemperature, Slice<uint8_t, 4>(bytes));
}
bool Message::ReadTemperature(float* temperature) const {
  *temperature = ::LittleEndianReadF32(payload().data());
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
int EncodeChannelGain(float gain) {
  if (gain < 0.05f) { return 0; }
  float gain_db = static_cast<float>(20.0 * M_LN2 / M_LN10) * FastLog2(gain);
  return static_cast<int>(
      std_shim::min(std_shim::max((62.0f / 18.0f) * gain_db, -62.0f), 0.0f) +
      63.5f);
}

float DecodeChannelGain(int code) {
  if (code == 0) { return 0.0f; }
  float gain_db = (18.0f / 62.0f) * (code - 63);
  return FastExp2(static_cast<float>(M_LN10 / (20.0 * M_LN2)) * gain_db);
}
}  // namespace

void Message::WriteChannelMapOrGainUpdate(
    const ChannelMap& channel_map, const int* test_channels) {
  const int num_in = std_shim::min<int>(16, channel_map.num_input_channels);
  const int num_out = std_shim::min<int>(12, channel_map.num_output_channels);
  uint8_t* dest = bytes_ + kHeaderSize;

  // Write number of input and output channels in the first byte.
  *dest = static_cast<uint8_t>((num_in & 15) | num_out << 4);
  ++dest;

  if (test_channels) {
    // Write the two test channel indices, 4 bits for each.
    *dest = static_cast<uint8_t>((test_channels[0] & 15) |
                                 test_channels[1] << 4);
    ++dest;
  } else {
    // Write source mapping, 4 bits per channel.
    for (int c = 0; c < num_out; c += 2, ++dest) {
      *dest = static_cast<uint8_t>((channel_map.sources[c] & 15) |
                                   (channel_map.sources[c + 1] & 15) << 4);
    }
  }

  // Write gains, 6 bits per channel. Each loop iteration handles 4 channels.
  const float* gains = channel_map.gains;
  for (int c = 0; c < num_out; c += 4, dest += 3) {
    const uint_fast32_t pack24 =
        static_cast<uint_fast32_t>(EncodeChannelGain(gains[c])) |
        static_cast<uint_fast32_t>(EncodeChannelGain(gains[c + 1])) << 6 |
        static_cast<uint_fast32_t>(EncodeChannelGain(gains[c + 2])) << 12 |
        static_cast<uint_fast32_t>(EncodeChannelGain(gains[c + 3])) << 18;
    dest[0] = static_cast<uint8_t>(pack24);
    dest[1] = static_cast<uint8_t>(pack24 >> 8);
    dest[2] = static_cast<uint8_t>(pack24 >> 16);
  }

  // Set payload size and message type.
  bytes_[3] = static_cast<uint8_t>(dest - (bytes_ + kHeaderSize));
  set_type(test_channels ? MessageType::kChannelGainUpdate
                         : MessageType::kChannelMap);
}

bool Message::ReadChannelMapOrGainUpdate(ChannelMap* channel_map,
                                         int* test_channels,
                                         int expected_inputs,
                                         int expected_outputs) const {
  if (payload_size() == 0) { return false; }
  const uint8_t* src = bytes_ + kHeaderSize;
  // Read number of input and output channels from the first byte.
  const int num_in = 1 + ((*src + 15) & 15);
  const int num_out = *src >> 4;
  ++src;

  const int expected_size =
      1 + 3 * ((num_out + 3) / 4) + (test_channels ? 1 : ((num_out + 1) / 2));
  if (payload_size() != expected_size ||
      (expected_inputs >= 0 && num_in != expected_inputs) ||
      (expected_outputs >= 0 && num_out != expected_outputs)) {
    return false;
  }

  channel_map->num_input_channels = num_in;
  channel_map->num_output_channels = num_out;

  if (test_channels) {
    // Read the two test channel indices, 4 bits each.
    test_channels[0] = *src & 15;
    test_channels[1] = *src >> 4;
    ++src;
  } else {
    // Read source mapping, 4 bits per channel.
    int* sources = channel_map->sources;
    for (int c = 0; c < num_out; c += 2, ++src) {
      sources[c] = std_shim::min<int>(*src & 15, num_in - 1);
      sources[c + 1] = std_shim::min<int>(*src >> 4, num_in - 1);
    }
  }

  // Read gains, 6 bits per channel.
  float* gains = channel_map->gains;
  for (int c = 0; c < num_out; c += 4, src += 3) {
    const uint_fast32_t pack24 =
        static_cast<uint_fast32_t>(src[0])
        | static_cast<uint_fast32_t>(src[1]) << 8
        | static_cast<uint_fast32_t>(src[2]) << 16;
    gains[c] = DecodeChannelGain(pack24 & 63);
    gains[c + 1] = DecodeChannelGain((pack24 >> 6) & 63);
    gains[c + 2] = DecodeChannelGain((pack24 >> 12) & 63);
    gains[c + 3] = DecodeChannelGain((pack24 >> 18) & 63);
  }

  return true;
}

void Message::WriteChannelMap(const ChannelMap& channel_map) {
  WriteChannelMapOrGainUpdate(channel_map, nullptr);
}

bool Message::ReadChannelMap(ChannelMap* channel_map, int expected_inputs,
                             int expected_outputs) const {
  return ReadChannelMapOrGainUpdate(channel_map, nullptr, expected_inputs,
                                    expected_outputs);
}

void Message::WriteChannelGainUpdate(
    const ChannelMap& channel_map, int test_channels[2]) {
  WriteChannelMapOrGainUpdate(channel_map, test_channels);
}

bool Message::ReadChannelGainUpdate(ChannelMap* channel_map,
                                    int test_channels[2], int expected_inputs,
                                    int expected_outputs) const {
  return ReadChannelMapOrGainUpdate(channel_map, test_channels, expected_inputs,
                                    expected_outputs);
}

void Message::WriteStatsRecord(const EnvelopeTracker& envelope_tracker) {
  EnvelopeTrackerGetRecord(&envelope_tracker, bytes_ + kHeaderSize);
  bytes_[3] = kEnvelopeTrackerRecordBytes;
  set_type(MessageType::kStatsRecord);
}

}  // namespace audio_tactile
