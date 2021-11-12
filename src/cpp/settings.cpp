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

#include "cpp/settings.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/fast_fun.h"
#include "dsp/math_constants.h"
#include "tactile/parse_key_value.h"
#include "tactile/tactile_processor.h"

namespace audio_tactile {

Settings::Settings() {
  device_name[0] = '\0';
  tuning = kDefaultTuningKnobs;
  ChannelMapInit(&channel_map, kTactileProcessorNumTactors);
}

bool Settings::operator==(const Settings& rhs) const {
  // Compare device_name strings.
  if (strcmp(device_name, rhs.device_name) ||
      // Compare tuning knobs.
      memcmp(tuning.values, rhs.tuning.values, kNumTuningKnobs)) {
    return false;
  }

  // Compare channel_maps, tacking care to only compare the first `num_out`
  // elements of `sources` and `gains`.
  const int num_out = channel_map.num_output_channels;
  const float* gains = channel_map.gains;
  const int* sources = channel_map.sources;
  if (channel_map.num_input_channels != rhs.channel_map.num_input_channels ||
      channel_map.num_output_channels != rhs.channel_map.num_output_channels ||
      memcmp(sources, rhs.channel_map.sources, num_out * sizeof(int)) ||
      memcmp(gains, rhs.channel_map.gains, num_out * sizeof(float))) {
    return false;
  }

  return true;
}

namespace settings_internal {

SettingsFileReader::SettingsFileReader(Settings* settings)
    : settings_(settings), line_number_(0), section_(0) {}

namespace {
// Truncates a string to length of at most `max_length`. Used below for
// error messages to ensure `s` fits in the buffer and is a readable length.
void TruncateToLength(char* s, int max_length) {
  const int length = strlen(s);
  if (length > max_length) {
    // Replace end of string with "..." ellipsis to indicate truncation.
    memcpy(s + max_length - 3, "...", 4);
  }
}
}  // namespace

ErrorCode SettingsFileReader::ReadLine() {
  enum {
    kSectionNone,
    kSectionTuning,
    kSectionChannelMap,
  };

  ++line_number_;

  ParsedKeyValue kv;
  if (!ParseKeyValue(buffer_, &kv)) {
    return ErrorInvalidSyntax(buffer_);
  } else if (kv.key == nullptr) {
    return kSuccess;  // Blank or comment line.
  }

  // Handle the kv. In the following, each handled path ends by returning.
  // Otherwise, execution falls to the end and sets an "unknown key error".

  if (kv.indent == 0) {  // Handle a top-level key.
    section_ = kSectionNone;
    if (strcmp(kv.key, "device_name") == 0) {
      TruncateToLength(kv.value, kMaxDeviceNameLength);
      strcpy(settings_->device_name, kv.value);  // NOLINT(runtime/printf)
      return kSuccess;
    } else  if (strcmp(kv.key, "tuning") == 0) {
      section_ = kSectionTuning;
      return kSuccess;
    } else if (strcmp(kv.key, "channel_map") == 0) {
      section_ = kSectionChannelMap;
      return kSuccess;
    }
  } else {  // Handle a subkey of some section.
    switch (section_) {
      case kSectionTuning:
        return ReadTuningKnob(kv.key, kv.value);

      case kSectionChannelMap:
        if (strcmp(kv.key, "gains") == 0) {
          return ReadChannelMapGains(kv.value);
        } else if (strcmp(kv.key, "sources") == 0) {
          return ReadChannelMapSources(kv.value);
        }

        break;
    }
  }

  return ErrorUnknownKey(kv.key);
}

namespace {
// Maps `c` to lowercase and non-alphanumeric chars to '_'.
char NormalizeChar(char c) { return isalnum(c) ? tolower(c) : '_'; }

// Finds index of the tuning knob with the same name as `key`. Comparison is
// forgiving, made after normalizing each char with NormalizeChar(). Returns -1
// if not found.
int FindTuningKnob(const char* key) {
  const int key_length = static_cast<int>(strlen(key));

  for (int knob = 0; knob < kNumTuningKnobs; ++knob) {
    const char* name = kTuningKnobInfo[knob].name;
    // Can't be a match unless lengths agree.
    if (key_length != static_cast<int>(strlen(name))) { continue; }

    int i;
    for (i = 0; i < key_length; ++i) {
      if (NormalizeChar(key[i]) != NormalizeChar(name[i])) { break; }
    }

    if (i == key_length) {
      return knob;  // Return index of the matching knob.
    }
  }

  return -1;  // key not found.
}

// Extracts and returns the next list item in a comma-delimited list. Leading
// and trailing whitespace is removed. `*list` is updated to point to the
// following item. Returns nullptr if there are no more items.
char* NextListItem(char** list) {
  *list += strspn(*list, " \t");  // Trim leading whitespace.
  if (**list == '\0') { return nullptr; }

  char* item = *list;
  char* item_end = strchr(item, ',');
  int length;
  if (!item_end) {
    length = strlen(item);
    *list = item + length;
  } else {
    length = item_end - item;
    *list = item_end + 1;
  }

  while (length && isspace(item[length - 1])) {  // Trim trailing whitespace.
    --length;
  }
  // Cut the item substring. This modifies the original string.
  item[length] = '\0';
  return item;
}
}  // namespace

ErrorCode SettingsFileReader::ReadTuningKnob(char* key, char* value) {
  const int knob = FindTuningKnob(key);
  if (!(0 <= knob && knob < kNumTuningKnobs)) { return ErrorUnknownKey(key); }
  int knob_value;
  ErrorCode error = ParseInt(value, 0, 255, &knob_value);
  if (error) { return error; }

  settings_->tuning.values[knob] = static_cast<uint8_t>(knob_value);
  return kSuccess;
}

ErrorCode SettingsFileReader::ReadChannelMapGains(char* value) {
  for (int c = 0; c < kChannelMapMaxChannels; ++c) {
    char* item;
    if (!(item = NextListItem(&value))) { break; }

    int gain_control_value;
    ErrorCode error = ParseInt(item, 0, 63, &gain_control_value);
    if (error) { return error; }

    settings_->channel_map.gains[c] =
        ChannelGainFromControlValue(gain_control_value);
  }

  return kSuccess;
}

ErrorCode SettingsFileReader::ReadChannelMapSources(char* value) {
  for (int c = 0; c < kChannelMapMaxChannels; ++c) {
    char* item;
    if (!(item = NextListItem(&value))) { break; }

    int source_base_1;
    ErrorCode error = ParseInt(item, 1, 16, &source_base_1);
    if (error) { return error; }

    settings_->channel_map.sources[c] =
        source_base_1 - 1;  // Convert to base 0.
  }

  return kSuccess;
}

ErrorCode SettingsFileReader::ParseInt(
    char* s, int min_value, int max_value, int* value) {
  char* s_end;
  const long parsed_value = strtol(s, &s_end, /*base*/ 10);  // NOLINT
  if (s_end != s + strlen(s)) {
    return ErrorInvalidSyntax(s);
  } else if (!(min_value <= parsed_value && parsed_value <= max_value)) {
    return ErrorOutOfRangeValue(s);
  }
  *value = static_cast<int>(parsed_value);
  return kSuccess;
}

ErrorCode SettingsFileReader::ErrorInvalidSyntax(char* syntax) {
  TruncateToLength(syntax, 40);
  constexpr const char* kInvalidSyntax = "Invalid syntax: ";
  memmove(buffer_ + strlen(kInvalidSyntax), syntax, strlen(syntax) + 1);
  memcpy(buffer_, kInvalidSyntax, strlen(kInvalidSyntax));
  // Abort reading. Settings are corrupt or incorrect in some basic way.
  return kFatalError;
}

ErrorCode SettingsFileReader::ErrorUnknownKey(char* key) {
  TruncateToLength(key, 40);
  const int key_length = strlen(key);
  constexpr const char* kUnknownKey = "Unknown key: \"";
  memmove(buffer_ + strlen(kUnknownKey), key, key_length);
  memcpy(buffer_, kUnknownKey, strlen(kUnknownKey));
  memcpy(buffer_ + strlen(kUnknownKey) + key_length, "\"", 2);
  // Consider an unknown key as a nonfatal error; reading should continue. An
  // unknown key might be misspelled or refer to a removed or renamed parameter.
  return kNonfatalError;
}

ErrorCode SettingsFileReader::ErrorOutOfRangeValue(char* value) {
  TruncateToLength(value, 40);
  constexpr const char* kOutOfRange = "Out of range: ";
  memmove(buffer_ + strlen(kOutOfRange), value, strlen(value) + 1);
  memcpy(buffer_, kOutOfRange, strlen(kOutOfRange));
  // The current key-value is invalid and should be discarded, but reading can
  // safely continue.
  return kNonfatalError;
}

namespace {
// Prints an integer between 0 and 255. Returns the number of digits written.
int PrintInt255(int value, char* s) {
  char digits[3];
  int num_digits = 0;
  do {
    digits[num_digits++] = '0' + value % 10;
    value /= 10;
  } while (num_digits < 3 && value > 0);

  for (int i = 0; i < num_digits; ++i) {
    s[i] = digits[num_digits - 1 - i];
  }
  s[num_digits] = '\0';
  return num_digits;
}
}  // namespace

void WriteTuningKnob(const TuningKnobs& tuning, int knob, char* line) {
  *line++ = ' ';  // Write two-space indent.
  *line++ = ' ';
  for (const char* name = kTuningKnobInfo[knob].name; *name; ++name) {
    *line++ = NormalizeChar(*name);  // Write knob name.
  }

  // Append the value as an integer 0-255.
  *line++ = ':';
  *line++ = ' ';
  PrintInt255(tuning.values[knob], line);
}

void WriteChannelGains(const ChannelMap& channel_map, char* line) {
  constexpr const char* kGainsKey = "  gains: ";
  memcpy(line, kGainsKey, strlen(kGainsKey));
  line += strlen(kGainsKey);
  for (int c = 0; c < channel_map.num_output_channels; ++c) {
    if (c > 0) {
      *line++ = ',';  // Comma delimiter between items.
      *line++ = ' ';
    }
    // Write gain as an integer control value 0-63.
    line += PrintInt255(ChannelGainToControlValue(channel_map.gains[c]), line);
  }
}

void WriteChannelSources(const ChannelMap& channel_map, char* line) {
  constexpr const char* kSourcesKey = "  sources: ";
  memcpy(line, kSourcesKey, strlen(kSourcesKey));
  line += strlen(kSourcesKey);
  for (int c = 0; c < channel_map.num_output_channels; ++c) {
    if (c > 0) {
      *line++ = ',';  // Comma delimiter between items.
      *line++ = ' ';
    }
    // Write source as a base-1 index.
    line += PrintInt255(channel_map.sources[c] + 1, line);
  }
}

}  // namespace settings_internal

}  // namespace audio_tactile
