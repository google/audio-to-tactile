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

#include "src/cpp/message.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "extras/tools/channel_map_tui.h"
#include "src/cpp/constants.h"
#include "src/dsp/logging.h"
#include "src/tactile/tuning.h"

// NOLINTBEGIN(readability/check)

namespace audio_tactile {

template <typename T>
std::vector<T> RandomValues(int n, std::mt19937* rng) {
  std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                        std::numeric_limits<T>::max());
  std::vector<T> values(n);
  for (T& v : values) {
    v = dist(*rng);
  }
  return values;
}

// Test the kAudioSamples message.
void TestAudioSamples() {
  puts("TestAudioSamples");
  std::mt19937 rng(0);
  std::vector<int16_t> samples = RandomValues<int16_t>(kAdcDataSize, &rng);

  Message message;
  message.WriteAudioSamples(Slice<int16_t, kAdcDataSize>(samples.data()));
  CHECK(message.type() == MessageType::kAudioSamples);
  CHECK(message.size() ==
        Message::kHeaderSize + kAdcDataSize * sizeof(int16_t));
  auto payload = message.payload();
  CHECK(payload.data() != reinterpret_cast<uint8_t*>(samples.data()));
  CHECK(payload.size() == kAdcDataSize * sizeof(int16_t));

  int16_t recovered[kAdcDataSize];
  CHECK(message.ReadAudioSamples(Slice<int16_t, kAdcDataSize>(recovered)));
  CHECK(std::equal(recovered, recovered + kAdcDataSize, samples.begin()));
}

// Test the kTactor*Samples messages
void TestSingleTactorSamples() {
  puts("TestSingleTactorSamples");
  std::mt19937 rng(0);
  for (auto channel_and_type : std::vector<std::pair<int, MessageType>>(
           {{1, MessageType::kTactor1Samples},
            {2, MessageType::kTactor2Samples},
            {3, MessageType::kTactor3Samples},
            {12, MessageType::kTactor12Samples}})) {
    const int channel = channel_and_type.first;
    const MessageType expected_type = channel_and_type.second;

    std::vector<uint16_t> samples = RandomValues<uint16_t>(kNumPwmValues, &rng);

    Message message;
    message.WriteSingleTactorSamples(
        channel, Slice<uint16_t, kNumPwmValues>(samples.data()));
    CHECK(message.type() == expected_type);
    auto payload = message.payload();
    CHECK(payload.size() == kNumPwmValues * sizeof(uint16_t));

    int recovered_channel;
    uint16_t recovered[kNumPwmValues];
    CHECK(message.ReadSingleTactorSamples(
        &recovered_channel, Slice<uint16_t, kNumPwmValues>(recovered)));
    CHECK(recovered_channel == channel);
    CHECK(std::equal(recovered, recovered + kNumPwmValues, samples.begin()));
  }
}

// Test the kAllTactorsSamples message.
void TestAllTactorsSamples() {
  puts("TestAllTactorsSamples");
  constexpr int kSize = kNumTotalPwm * kNumPwmValues;
  std::mt19937 rng(0);
  std::vector<uint8_t> samples = RandomValues<uint8_t>(kSize, &rng);

  Message message;
  message.WriteAllTactorsSamples(Slice<uint8_t, kSize>(samples.data()));
  CHECK(message.type() == MessageType::kAllTactorsSamples);
  auto payload = message.payload();
  CHECK(payload.size() == kNumTotalPwm * kNumPwmValues);

  uint8_t recovered[kSize];
  CHECK(message.ReadAllTactorsSamples(Slice<uint8_t, kSize>(recovered)));
  CHECK(std::equal(recovered, recovered + kSize, samples.begin()));
}

// Test the kTemperature message.
void TestTemperature() {
  puts("TestTemperature");
  Message message;
  message.WriteTemperature(12.345f);
  CHECK(message.type() == MessageType::kTemperature);
  CHECK(message.payload().size() == 4);

  float recovered;
  CHECK(message.ReadTemperature(&recovered));
  CHECK(recovered == 12.345f);
}

// Test the kBatteryVoltage message.
void TestBatteryVoltage() {
  puts("TestBatteryVoltage");
  Message message;
  message.WriteBatteryVoltage(3.750f);
  CHECK(message.type() == MessageType::kBatteryVoltage);
  CHECK(message.payload().size() == 4);

  float recovered;
  CHECK(message.ReadBatteryVoltage(&recovered));
  CHECK(recovered == 3.750f);
}

// Test the kTuning message.
void TestTuning() {
  puts("TestTuning");
  std::mt19937 rng(0);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  TuningKnobs knobs;
  for (int i = 0; i < kNumTuningKnobs; ++i) {
    knobs.values[i] = dist(rng);
  }

  Message message;
  message.WriteTuning(knobs);
  CHECK(message.type() == MessageType::kTuning);
  CHECK(message.payload().size() == kNumTuningKnobs);

  TuningKnobs recovered;
  CHECK(message.ReadTuning(&recovered));
  for (int i = 0; i < kNumTuningKnobs; ++i) {
    CHECK(recovered.values[i] == knobs.values[i]);
  }
}

// Test the kTactilePattern message.
void TestTactilePattern() {
  puts("TestTactilePattern");
  std::string pattern = "abc123";
  Message message;
  message.WriteTactilePattern(pattern.c_str());
  CHECK(message.type() == MessageType::kTactilePattern);
  CHECK(message.payload().size() == static_cast<int>(pattern.size()));

  char recovered[kMaxTactilePatternLength + 1];
  CHECK(message.ReadTactilePattern(recovered));
  CHECK(recovered == pattern);

  // Test where the pattern length exceeds kMaxTactilePatternLength.
  pattern = "This pattern is much too long...";
  message.WriteTactilePattern(pattern.c_str());
  CHECK(message.type() == MessageType::kTactilePattern);
  CHECK(message.payload().size() == kMaxTactilePatternLength);

  CHECK(message.ReadTactilePattern(recovered));
  CHECK(recovered == pattern.substr(0, kMaxTactilePatternLength));
}

void TestChannelMap() {
  puts("TestChannelMap");
  ChannelMap channel_map;
  ChannelMap recovered;
  Message message;

  CHECK(ChannelMapParse(4, "1,2,3,4,0", "2,-2,-10.3,-20", &channel_map));
  message.WriteChannelMap(channel_map);
  CHECK(message.type() == MessageType::kChannelMap);

  CHECK(message.ReadChannelMap(&recovered));
  CHECK(recovered.num_input_channels == channel_map.num_input_channels);
  CHECK(recovered.num_output_channels == channel_map.num_output_channels);
  CHECK(recovered.sources[0] == 0);
  CHECK(recovered.sources[1] == 1);
  CHECK(recovered.sources[2] == 2);
  CHECK(recovered.sources[3] == 3);
  // Gain greater than 0 dB saturates.
  CHECK(recovered.gains[0] == 1.0f);
  // Check that intermediate gains are within 4% of original values.
  CHECK(fabs(recovered.gains[1] - channel_map.gains[1])
        <= 0.04f * channel_map.gains[1]);
  CHECK(fabs(recovered.gains[2] - channel_map.gains[2])
        <= 0.04f * channel_map.gains[2]);
  // Gain less than -18 dB saturates.
  CHECK(fabs(recovered.gains[3] - 0.125f) <= 0.04f * 0.125f);
  CHECK(recovered.gains[4] == 0.0);

  channel_map.num_input_channels = 16;
  std::mt19937 rng(0);
  for (int num_out = 0; num_out <= 12; ++num_out) {
    channel_map.num_output_channels = num_out;
    for (int c = 0; c < num_out; ++c) {
      channel_map.gains[c] =
          std::uniform_real_distribution<float>(0.125f, 1.0f)(rng);
      channel_map.sources[c] =
          std::uniform_int_distribution<int>(0, 15)(rng);
    }

    message.WriteChannelMap(channel_map);
    CHECK(message.ReadChannelMap(&recovered));

    CHECK(recovered.num_input_channels == channel_map.num_input_channels);
    CHECK(recovered.num_output_channels == channel_map.num_output_channels);
    for (int c = 0; c < num_out; ++c) {
      CHECK(fabs(recovered.gains[c] - channel_map.gains[c])
            <= 0.04f * channel_map.gains[c]);
      CHECK(recovered.sources[c] == channel_map.sources[c]);
    }
  }
}

void TestChannelGainUpdate() {
  puts("TestChannelGainUpdate");
  ChannelMap channel_map;
  int test_channels[2];
  ChannelMap recovered;
  int recovered_test_channels[2];
  Message message;

  channel_map.num_input_channels = 16;
  std::mt19937 rng(0);
  for (int num_out = 0; num_out <= 12; ++num_out) {
    channel_map.num_output_channels = num_out;
    for (int c = 0; c < 2; ++c) {
      test_channels[c] = std::uniform_int_distribution<int>(0, 15)(rng);
    }
    for (int c = 0; c < num_out; ++c) {
      channel_map.gains[c] =
          std::uniform_real_distribution<float>(0.125f, 1.0f)(rng);
    }

    message.WriteChannelGainUpdate(channel_map, test_channels);
    CHECK(message.ReadChannelGainUpdate(&recovered, recovered_test_channels));

    CHECK(recovered.num_input_channels == channel_map.num_input_channels);
    CHECK(recovered.num_output_channels == channel_map.num_output_channels);
    for (int c = 0; c < 2; ++c) {
      CHECK(recovered_test_channels[c] == test_channels[c]);
    }
    for (int c = 0; c < num_out; ++c) {
      CHECK(fabs(recovered.gains[c] - channel_map.gains[c])
            <= 0.04f * channel_map.gains[c]);
    }
  }
}

// Test the kDeviceName message.
void TestDeviceName() {
  puts("TestDeviceName");
  std::string name = "abc123";
  Message message;
  message.WriteDeviceName(name.c_str());
  CHECK(message.type() == MessageType::kDeviceName);
  CHECK(message.payload().size() == static_cast<int>(name.size()));

  char recovered[17];
  CHECK(message.ReadDeviceName(recovered));
  CHECK(recovered == name);

  // Test where the name length exceeds 16 chars.
  name = "This name is much too long...";
  message.WriteDeviceName(name.c_str());
  CHECK(message.type() == MessageType::kDeviceName);
  CHECK(message.payload().size() == 16);

  CHECK(message.ReadDeviceName(recovered));
  CHECK(recovered == name.substr(0, 16));
}

// Test the kOnConnectionBatch message.
void TestOnConnectionBatch() {
  puts("TestOnConnectionBatch");
  Message message;

  std::mt19937 rng(0);
  for (int trial = 0; trial < 10; ++trial) {
    const int firmware_build_date =
        std::uniform_int_distribution<int>(20000101, 21000101)(rng);
    const float battery_v =
        std::uniform_real_distribution<float>(0.0f, 5.0f)(rng);
    const float temperature_c =
        std::uniform_real_distribution<float>(0.0f, 40.0f)(rng);

    Settings settings;
    const int device_name_length =
        std::uniform_int_distribution<int>(0, kMaxDeviceNameLength)(rng);
    for (int i = 0; i < device_name_length; ++i) {
      settings.device_name[i] =
          std::uniform_int_distribution<char>('a', 'z')(rng);
    }
    settings.device_name[device_name_length] = '\0';

    for (int i = 0; i < kNumTuningKnobs; ++i) {
      settings.tuning.values[i] =
          std::uniform_int_distribution<uint8_t>(0, 255)(rng);
    }

    const int num_in = std::uniform_int_distribution<int>(4, 16)(rng);
    const int num_out = std::uniform_int_distribution<int>(1, 12)(rng);
    settings.channel_map.num_input_channels = num_in;
    settings.channel_map.num_output_channels = num_out;
    for (int c = 0; c < num_out; ++c) {
      settings.channel_map.gains[c] =
          std::uniform_real_distribution<float>(0.125f, 1.0f)(rng);
      settings.channel_map.sources[c] =
          std::uniform_int_distribution<int>(0, num_in - 1)(rng);
    }

    message.WriteOnConnectionBatch(
        /*firmware_build_date=*/firmware_build_date,
        /*battery_v=*/battery_v,
        /*temperature_c=*/temperature_c,
        /*settings=*/settings);

    int recovered_firmware_build_date;
    float recovered_battery_v;
    float recovered_temperature_c;
    Settings recovered_settings;
    CHECK(message.ReadOnConnectionBatch(
        /*firmware_build_date=*/&recovered_firmware_build_date,
        /*battery_v=*/&recovered_battery_v,
        /*temperature_c=*/&recovered_temperature_c,
        /*settings=*/&recovered_settings));

    CHECK(recovered_firmware_build_date == firmware_build_date);
    CHECK(recovered_battery_v == battery_v);
    CHECK(recovered_temperature_c == temperature_c);
    CHECK(!strcmp(recovered_settings.device_name, settings.device_name));
    for (int i = 0; i < kNumTuningKnobs; ++i) {
      CHECK(recovered_settings.tuning.values[i] == settings.tuning.values[i]);
    }
    CHECK(recovered_settings.channel_map.num_input_channels
          == settings.channel_map.num_input_channels);
    CHECK(recovered_settings.channel_map.num_output_channels
          == settings.channel_map.num_output_channels);
    for (int c = 0; c < num_out; ++c) {
      const float expected_gain = settings.channel_map.gains[c];
      CHECK(fabs(recovered_settings.channel_map.gains[c] - expected_gain)
            <= 0.04f * expected_gain);
      const int expected_source = settings.channel_map.sources[c];
      CHECK(recovered_settings.channel_map.sources[c] == expected_source);
    }
  }
}

// Test a header-only message, like kDisableAmplifiers.
void TestSimpleMessage(const char* name, MessageType expected,
                       void (Message::*WriteFun)()) {
  printf("TestSimpleMessage(%s)\n", name);
  Message message;
  (message.*WriteFun)();
  CHECK(message.type() == expected);
  CHECK(message.payload().empty());
}

void TestSimpleMessages() {
  TestSimpleMessage("DisableAmplifiers", MessageType::kDisableAmplifiers,
                    &Message::WriteDisableAmplifiers);
  TestSimpleMessage("EnableAmplifiers", MessageType::kEnableAmplifiers,
                    &Message::WriteEnableAmplifiers);
  TestSimpleMessage("GetTuning", MessageType::kGetTuning,
                    &Message::WriteGetTuning);
  TestSimpleMessage("GetChannelMap", MessageType::kGetChannelMap,
                    &Message::WriteGetChannelMap);
  TestSimpleMessage("GetDeviceName", MessageType::kGetDeviceName,
                    &Message::WriteGetDeviceName);
  TestSimpleMessage("StreamDataStart", MessageType::kStreamDataStart,
                    &Message::WriteStreamDataStart);
  TestSimpleMessage("StreamDataStop", MessageType::kStreamDataStop,
                    &Message::WriteStreamDataStop);
}

}  // namespace audio_tactile

// NOLINTEND

int main(int argc, char** argv) {
  audio_tactile::TestAudioSamples();
  audio_tactile::TestSingleTactorSamples();
  audio_tactile::TestAllTactorsSamples();
  audio_tactile::TestTemperature();
  audio_tactile::TestBatteryVoltage();
  audio_tactile::TestTuning();
  audio_tactile::TestTactilePattern();
  audio_tactile::TestChannelMap();
  audio_tactile::TestChannelGainUpdate();
  audio_tactile::TestDeviceName();
  audio_tactile::TestOnConnectionBatch();
  audio_tactile::TestSimpleMessages();

  puts("PASS");
  return EXIT_SUCCESS;
}
