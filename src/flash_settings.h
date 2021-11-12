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
// Library for reading and writing device settings to flash.

#ifndef AUDIO_TO_TACTILE_SRC_FLASH_SETTINGS_H_
#define AUDIO_TO_TACTILE_SRC_FLASH_SETTINGS_H_

#include "cpp/settings.h"  // NOLINT(build/include)

// Path for the settings file. Must be a valid 8.3 FAT filename.
// https://en.wikipedia.org/wiki/8.3_filename
#define kFlashSettingsFile "settings.cfg"

namespace audio_tactile {

// Flash write status.
enum {
  kFlashWriteSuccess = 0,
  kFlashWriteUnkownError = 1,
  kFlashWriteErrorNotFormatted = 2,
};

class AudioToTactileFlashSettings {
 public:
  AudioToTactileFlashSettings();

  // Initializes flash filesystem.
  void Initialize();

  // True if a FAT flash file system was found on the device.
  bool have_file_system() const { return have_file_system_; }

  // Reads from settings.cfg flash file. Returns true on success.
  bool ReadSettingsFile(Settings* settings);

  // Writes to settings.cfg flash file. The function compares `settings` to the
  // last written settings, and only writes to flash if they differ. Returns
  // true on success.
  //
  // NOTE: Calls to this function should be rate limited to prevent prematurely
  // wearing out the flash.
  bool WriteSettingsFile(const Settings& settings);

 private:
  Settings last_written_settings_;
  bool have_file_system_;
};
extern AudioToTactileFlashSettings FlashSettings;

}  // namespace audio_tactile
#endif  // AUDIO_TO_TACTILE_SRC_FLASH_SETTINGS_H_
