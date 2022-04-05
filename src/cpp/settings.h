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
// Device Settings struct, with name, tuning, and channel map.
//
// Defines a `Settings` struct of data that should be saved to flash. The format
// of the settings file is based on NestedText, a simple human-readable format
// for structured data [https://nestedtext.org/en/latest/index.html].
//
// # An example settings file.
// device_name: My device
// input: analog mic
//
// tuning:
//   input_gain: 123
//   output_gain: 200
//   agc_strength: 191
//
// channel_map:
//   gains: 60, 63
//   sources: 3, 1
//
// == Format ==
//
// Whitespace lines and lines starting with # are ignored. All other lines have
// the form "key: value". Indentation is significant: top-level keys have no
// indentation while indented lines represent subkeys. The following keys are
// understood:
//
// * device_name: sets a user-customizable name for the device. For BLE
//   limitations, the name is currently limited to 16 characters.
//
// * input: selects where to read input audio. May be either "analog" or "pdm".
//
// * tuning: each subkey names a tuning knob. The name lookup searches over
//   `name` fields in `kTuningKnobInfo` (src/tactile/tuning.c). Case and
//   nonalphanumeric characeters are ignored so that knob subkeys may be written
//   in snake_case like "agc_strength" instead of "AGC strength". The value is
//   the knob's control value, an integer in 0-255.
//
// * channel_map
//
//    * gains: a comma-delimited list of channel gains. The gains are integer
//      control values in 0-63. Value 0 sets a gain of zero, disabling the
//      channel, and values 1-63 map to gains in the range -18 dB to 0 dB.
//
//    * sources: a comma-delimited list of base-1 source indices. Defines how
//      sources map to tactors. For instance, "3, 1" maps source 3 to the first
//      tactor and source 1 to the second tactor.
//
// == Error handling ==
//
// Generally, the reader should report errors but continue reading where
// reasonable. The format considers each line almost independently (at least
// among lines of the same indentation level), so it is plausible that lines
// following an error can be interpreted. For instance an unknown key is a
// nonfatal error since it may be a misspelling or due to version mismatch
// between settings file and the reader.
//
// Specifically for channel_map: The lengths of the "gains" and "sources" lists
// should normally match the number of tactors. But if they don't, excess
// elements are ignored, and unset elements take on default values.

#ifndef AUDIO_TO_TACTILE_SRC_CPP_SETTINGS_H_
#define AUDIO_TO_TACTILE_SRC_CPP_SETTINGS_H_

#include <string.h>

#include "cpp/constants.h"
#include "dsp/channel_map.h"
#include "tactile/tuning.h"

namespace audio_tactile {

struct Settings {
  char device_name[kMaxDeviceNameLength + 1];
  InputSelection input;
  TuningKnobs tuning;
  ChannelMap channel_map;

  Settings();

  // Settings objects can be compared.
  bool operator==(const Settings& rhs) const;
  bool operator!=(const Settings& rhs) const { return !(*this == rhs); }

  // Reads Settings from a settings file. File reading and error reporting are
  // abstracted by using callback args.
  //
  // The `read_line_fun` callback has the signature
  //
  //   bool read_line_fun(char* buffer, int buffer_size)
  //
  // Like the standard `fgets()` function, the callback reads one text line,
  // writes up to (buffer_size - 1) characters into `buffer`, and append a null
  // character at the position immediately after the last written character. The
  // callback returns true if a line was read successfully, or false if not,
  // i.e. on IO error or end of file. The reader stops when false is returned.
  //
  // The `error_fun` callback has the signature
  //
  //   void error_fun(int line_number, const char* message)
  //
  // `error_fun` is called on error. `line_number` is the text line number
  // (starting from 1 for the first line) where the error occurred and `message`
  // is a human-readable description of the error, e.g. "Out of range: 300"
  // where a value in 0-255 was expected. This way, the method of logging or
  // reporting errors can be customized.
  //
  // NOTE: Some errors are nonfatal. The reader may attempt to continue reading
  // even after calling `error_fun`.
  //
  // The `device_name`, `tuning`, and `channel_map` args should be set to valid
  // default values before calling this function. Parameters are set in these
  // args as they are read from the file. If a parameter is unspecified, the arg
  // is not changed and keeps the value it was set to initially. If a parameter
  // is specified more than once, the last read setting applies.
  template <typename ReadLineFun, typename ErrorFun>
  void ReadFile(ReadLineFun read_line_fun, ErrorFun error_fun);

  // Writes Settings to a settings file. Returns true on success, false on
  // failure. The `write_line_fun` callback has the signature
  //
  //   bool write_line_fun(const char* line)
  //
  // and writes null-terminated string `line` to the file. The callback should
  // append a newline '\n' after the line. The callback returns true if
  // successful or false on error.
  template <typename WriteLineFun>
  bool WriteFile(WriteLineFun write_line_fun) const;
};

// Implementation details only below this line. ________________________________

namespace settings_internal {

enum ErrorCode { kSuccess, kNonfatalError, kFatalError };

class SettingsFileReader {  // Helper class for reading.
 public:
  enum { kBufferSize = 256 };

  // Constructor. Sets the output args and puts reader in initial state.
  explicit SettingsFileReader(Settings* settings);
  // Gets pointer to the reader's buffer. This buffer is used both for holding
  // input text lines and for holding error messages.
  char* buffer() { return buffer_; }
  // Gets the current line number, starting from 1 for the first line.
  int line_number() const { return line_number_; }

  // Handles the line currently set in `buffer`. Returns an ErrorCode.
  // On error, `buffer` is replaced with an error message.
  ErrorCode ReadLine();

 private:
  // Sets the tuning knob with name `key` to control value `value`.
  ErrorCode ReadTuningKnob(char* key, char* value);
  // Sets the channel map gains given a comma-delimited list.
  ErrorCode ReadChannelMapGains(char* value);
  // Sets the channel map sources given a comma-delimited list.
  ErrorCode ReadChannelMapSources(char* value);

  // Parses an integer from `s`, checks that it is between `min_value` and
  // `max_value`, and if successful sets `*value` to the result.
  ErrorCode ParseInt(char* s, int min_value, int max_value, int* value);
  // Sets an "Invalid syntax" error message.
  ErrorCode ErrorInvalidSyntax(char* syntax);
  // Sets an "Unknown key" error message.
  ErrorCode ErrorUnknownKey(char* key);
  // Sets an "Out of range" error message.
  ErrorCode ErrorOutOfRangeValue(char* value);
  // Make an error message by concatenating `message` and `syntax`.
  ErrorCode Error(ErrorCode code, const char* message, char* syntax);

  char buffer_[kBufferSize];
  Settings* settings_;
  int line_number_;  // Current line number.
  int section_;      // Current "section", for tracking what subkeys refer to.
};

// The following functions are for writing settings to file.

// Writes one tuning knob to `line`.
void WriteTuningKnob(const TuningKnobs& tuning, int knob, char* line);
// Writes channel gains to `line`.
void WriteChannelGains(const ChannelMap& channel_map, char* line);
// Writes channel sources to `line`.
void WriteChannelSources(const ChannelMap& channel_map, char* line);

}  // namespace settings_internal

template <typename ReadLineFun, typename ErrorFun>
void Settings::ReadFile(ReadLineFun read_line_fun, ErrorFun error_fun) {
  settings_internal::SettingsFileReader reader(this);

  // Each iteration reads one line of the settings file.
  while (read_line_fun(reader.buffer(), decltype(reader)::kBufferSize)) {
    // The core reading logic happens within ReadLine().
    const auto error = reader.ReadLine();
    if (error) {
      // There was an error. The error message is in reader.buffer().
      error_fun(reader.line_number(), reader.buffer());
      // Abort reading on fatal error.
      if (error == settings_internal::kFatalError) { return; }
    }
  }
}

template <typename WriteLineFun>
bool Settings::WriteFile(WriteLineFun write_line_fun) const {
  char line[256];

  // Header comment.
  if (!write_line_fun("# Audio-to-Tactile device settings.")) { return false; }

  // Device name.
  constexpr const char* kDeviceName = "device_name: ";
  memcpy(line, kDeviceName, strlen(kDeviceName));
  strcpy(line + strlen(kDeviceName), device_name);  // NOLINT
  if (!write_line_fun(line)) { return false; }

  // Input.
  switch (input) {
    case InputSelection::kAnalogMic:
      if (!write_line_fun("input: analog mic")) { return false; }
      break;

    case InputSelection::kPdmMic:
      if (!write_line_fun("input: PDM mic")) { return false; }
      break;
  }

  // Tuning knobs.
  if (!write_line_fun("\ntuning:")) { return false; }
  for (int knob = 0; knob < kNumTuningKnobs; ++knob) {
    settings_internal::WriteTuningKnob(tuning, knob, line);
    if (!write_line_fun(line)) { return false; }
  }

  // Channel map.
  const int num_out = channel_map.num_output_channels;
  if (0 <= num_out && num_out <= kChannelMapMaxChannels) {
    if (!write_line_fun("\nchannel_map:")) { return false; }

    settings_internal::WriteChannelGains(channel_map, line);
    if (!write_line_fun(line)) { return false; }
    settings_internal::WriteChannelSources(channel_map, line);
    if (!write_line_fun(line)) { return false; }
  }

  return true;
}

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_CPP_SETTINGS_H_
