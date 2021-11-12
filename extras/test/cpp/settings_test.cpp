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

#include "src/cpp/settings.h"

#include <string>

#include "src/dsp/channel_map.h"
#include "src/dsp/fast_fun.h"
#include "src/dsp/math_constants.h"
#include "src/dsp/logging.h"
#include "src/tactile/tuning.h"

// NOLINTBEGIN(readability/check)

namespace audio_tactile {
namespace {

// For testing purposes, a mock file stored in memory.
class InMemoryFile {
 public:
  InMemoryFile() {}
  explicit InMemoryFile(const char* content): content_(content) {}
  // Gets the file contents.
  const char* content() const { return content_.c_str(); }

  class Reader {
   public:
    explicit Reader(const std::string& content): s_(content.c_str()) {}

    // Reads up to buffer_size - 1 bytes from the file. Returns false on EOF.
    bool operator()(char* buffer, int buffer_size) {
      if (*s_ == '\0') { return false; }

      const char* newline = strchr(s_, '\n');
      const char* line_end = newline ? (newline + 1) : (s_ + strlen(s_));
      int size = line_end - s_;
      if (size >= buffer_size) { size = buffer_size - 1; }
      memcpy(buffer, s_, size);
      buffer[size] = '\0';

      s_ = line_end;
      return true;
    }

   private:
    const char* s_;
  };
  Reader OpenForRead() const { return Reader(content_); }

  class Writer {
   public:
    explicit Writer(std::string& content): content_(content) {}

    // Writes `line` and appends a newline.
    bool operator()(const char* line) {
      content_ += line;
      content_ += '\n';
      return true;
    }

   private:
    std::string& content_;
  };
  Writer OpenForWrite() { return Writer(content_); }

 private:
  std::string content_;  // The in-memory file content.
};

void FailTestOnError(int line_number, const char* message) {
  fprintf(stderr, "Reading error: %d: %s\n", line_number, message);
  exit(EXIT_FAILURE);
}

// Tests Settings copy construction, assignment, ==, and != operators.
void TestCopyAndCompare() {
  puts("TestCopyAndCompare");
  Settings a;
  Settings b;
  CHECK(a == b);  // Equal since `a` and `b` are both the default settings.

  strcpy(a.device_name, "Pineapple");  // NOLINT
  CHECK(a != b);  // Test that comparison considers `.device_name`.
  b = a;
  CHECK(a == b);

  a.tuning.values[kKnobInputGain] = 123;
  CHECK(a != b);  // Comparison considers tuning knobs.
  b = a;
  CHECK(a == b);

  a.channel_map.gains[3] = 0.2f;
  CHECK(a != b);  // Comparison considers channel map gains.
  b = a;
  CHECK(a == b);

  a.channel_map.sources[3] += 1;
  CHECK(a != b);  // Comparison considers channel map sources.
  b = a;
  CHECK(a == b);

  Settings c(b);  // Test copy construction.
  CHECK(a == c);
}

void TestReadBasic() {
  puts("TestReadBasic");
  InMemoryFile settings_file(R"(
# Test settings.
device_name: Pineapple
tuning:
  input_gain: 123
  output_gain: 72
channel_map:
  gains: 10,  0  , 63
  sources: 3,2,1
)");
  InMemoryFile::Reader file_reader = settings_file.OpenForRead();
  Settings settings;

  settings.ReadFile(
      [&file_reader](char* buffer, int buffer_size) {
        return file_reader(buffer, buffer_size);
      },
      FailTestOnError  // There should be no errors for this input.
      );

  CHECK(strcmp(settings.device_name, "Pineapple") == 0);
  CHECK(settings.tuning.values[kKnobInputGain] == 123);
  CHECK(settings.tuning.values[kKnobOutputGain] == 72);
  CHECK(settings.channel_map.gains[0] == ChannelGainFromControlValue(10));
  CHECK(settings.channel_map.gains[1] == ChannelGainFromControlValue(0));
  CHECK(settings.channel_map.gains[2] == ChannelGainFromControlValue(63));
  CHECK(settings.channel_map.sources[0] == 2);
  CHECK(settings.channel_map.sources[1] == 1);
  CHECK(settings.channel_map.sources[2] == 0);
}

void TestReadInvalidSyntax() {
  puts("TestReadInvalidSyntax");
  InMemoryFile settings_file(R"(
This line is invalid because it has no colon character.
device_name: Reading shouldn't reach here
)");
  InMemoryFile::Reader file_reader = settings_file.OpenForRead();
  Settings settings;
  bool error_fun_called = false;

  settings.ReadFile(
      [&file_reader](char* buffer, int buffer_size) {
        return file_reader(buffer, buffer_size);
      },
      [&](int line_number, const char* message) {
        CHECK(line_number == 2);
        CHECK(strcmp(
            message,
            "Invalid syntax: This line is invalid because it has n...") == 0);
        error_fun_called = true;
      });

  CHECK(error_fun_called);
  CHECK(strcmp(settings.device_name, "") == 0);
}

void TestReadUnknownKey() {
  puts("TestReadUnknownKey");
  InMemoryFile settings_file(R"(
abracadabra: this key should be ignored
tuning:
  magic: 123
device_name: Should read this
)");
  InMemoryFile::Reader file_reader = settings_file.OpenForRead();
  Settings settings;
  bool error_on_abracadabra = false;
  bool error_on_magic = false;

  settings.ReadFile(
      [&file_reader](char* buffer, int buffer_size) {
        return file_reader(buffer, buffer_size);
      },
      [&](int line_number, const char* message) {
        switch (line_number) {
          case 2:
            CHECK(strcmp(message, "Unknown key: \"abracadabra\"") == 0);
            error_on_abracadabra = true;
            break;
          case 4:
            CHECK(strcmp(message, "Unknown key: \"magic\"") == 0);
            error_on_magic = true;
            break;
          default:
            // Unexpected error.
            FailTestOnError(line_number, message);
        }
      });

  CHECK(error_on_abracadabra);
  CHECK(error_on_magic);
  CHECK(strcmp(settings.device_name, "Should read this") == 0);
}

void TestReadOutOfRange() {
  puts("TestReadOutOfRange");
  InMemoryFile settings_file(R"(
channel_map:
  gains: 62, 65, 63
device_name: Should read this
)");
  InMemoryFile::Reader file_reader = settings_file.OpenForRead();
  Settings settings;
  bool error_fun_called = false;

  settings.ReadFile(
      [&file_reader](char* buffer, int buffer_size) {
        return file_reader(buffer, buffer_size);
      },
      [&](int line_number, const char* message) {
        CHECK(line_number == 3);
        CHECK(strcmp(message, "Out of range: 65") == 0);
        error_fun_called = true;
      });

  CHECK(error_fun_called);
  CHECK(strcmp(settings.device_name, "Should read this") == 0);
}

void TestWriteBasic() {
  puts("TestWriteBasic");
  Settings settings;  // Make up some test settings.
  strcpy(settings.device_name, "Pineapple");  // NOLINT
  settings.tuning.values[kKnobInputGain] = 123;
  settings.tuning.values[kKnobOutputGain] = 72;
  settings.channel_map.gains[0] = ChannelGainFromControlValue(10);
  settings.channel_map.gains[1] = ChannelGainFromControlValue(0);
  settings.channel_map.sources[0] = 2;
  settings.channel_map.sources[1] = 0;
  settings.channel_map.sources[2] = 1;

  // Write `settings` to `settings_file`.
  InMemoryFile settings_file;
  InMemoryFile::Writer file_writer = settings_file.OpenForWrite();
  CHECK(settings.WriteFile(
    [&file_writer](const char* line) {
      return file_writer(line);
    }));

  // Check that file contains these expected substrings.
  CHECK(strstr(settings_file.content(), "device_name: Pineapple"));
  CHECK(strstr(settings_file.content(), "  input_gain: 123"));
  CHECK(strstr(settings_file.content(), "  output_gain: 72"));
  CHECK(strstr(settings_file.content(), "  gains: 10, 0, 63, 63, 63"));
  CHECK(strstr(settings_file.content(), "  sources: 3, 1, 2, 4, 5"));

  // Check that file is readable and recovers the settings.
  Settings recovered;
  InMemoryFile::Reader file_reader = settings_file.OpenForRead();
  recovered.ReadFile(
      [&file_reader](char* buffer, int buffer_size) {
        return file_reader(buffer, buffer_size);
      },
      FailTestOnError  // There should be no errors for this input.
      );

  CHECK(strcmp(recovered.device_name, "Pineapple") == 0);
  CHECK(recovered.tuning.values[kKnobInputGain] == 123);
  CHECK(recovered.tuning.values[kKnobOutputGain] == 72);
  CHECK(recovered.channel_map.gains[0] == ChannelGainFromControlValue(10));
  CHECK(recovered.channel_map.gains[1] == ChannelGainFromControlValue(0));
  CHECK(recovered.channel_map.gains[2] == ChannelGainFromControlValue(63));
  CHECK(recovered.channel_map.sources[0] == 2);
  CHECK(recovered.channel_map.sources[1] == 0);
  CHECK(recovered.channel_map.sources[2] == 1)
  CHECK(recovered == settings);
}
}  // namespace
}  // namespace audio_tactile

// NOLINTEND

int main(int argc, char** argv) {
  audio_tactile::TestCopyAndCompare();
  audio_tactile::TestReadBasic();
  audio_tactile::TestReadInvalidSyntax();
  audio_tactile::TestReadUnknownKey();
  audio_tactile::TestReadOutOfRange();
  audio_tactile::TestWriteBasic();

  puts("PASS");
  return EXIT_SUCCESS;
}
