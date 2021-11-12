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

#include "flash_settings.h"  // NOLINT(build/include)

#include "Adafruit_SPIFlash.h"  // NOLINT(build/include)
#include "SPI.h"  // NOLINT(build/include)
#include "SdFat.h"  // NOLINT(build/include)

namespace audio_tactile {

AudioToTactileFlashSettings FlashSettings;

namespace {
Adafruit_FlashTransport_QSPI g_flash_transport;
Adafruit_SPIFlash g_flash(&g_flash_transport);
FatFileSystem g_flash_file_system;
File g_flash_file;
}  // namespace

AudioToTactileFlashSettings::AudioToTactileFlashSettings()
    : have_file_system_(false) {}

void AudioToTactileFlashSettings::Initialize() {
  // Initialize external flash.
  g_flash.begin();
  // Open FAT file system on the flash.
  have_file_system_ = g_flash_file_system.begin(&g_flash);
}

bool AudioToTactileFlashSettings::ReadSettingsFile(Settings* settings) {
  if (!have_file_system_ ||
      !(g_flash_file = g_flash_file_system.open(
          kFlashSettingsFile, FILE_READ))) {
    return false;
  }

  settings->ReadFile(
      [](char* buffer, int buffer_size) {
        // File::fgets reads one line and returns a number of bytes. IO error is
        // indicated by count < 0, and EOF by count == 0 and .available() == 0.
        const int count = g_flash_file.fgets(buffer, buffer_size);
        // Return true (success) if something was read or if it is not yet EOF.
        return count > 0 || (count == 0 && g_flash_file.available());
      },
      [](int line_number, const char* message) {
        // Print error message to serial.
        Serial.print("FlashSettings: " kFlashSettingsFile ":");
        Serial.print(line_number);
        Serial.print(": ");
        Serial.println(message);
      });

  g_flash_file.close();
  last_written_settings_ = *settings;
  Serial.println("FlashSettings: Read " kFlashSettingsFile);
  return true;
}

bool AudioToTactileFlashSettings::WriteSettingsFile(const Settings& settings) {
  if (last_written_settings_ == settings) { return true; }

  if (!have_file_system_ ||
      !(g_flash_file = g_flash_file_system.open(
          kFlashSettingsFile,
          // Open file for writing. If the file does not yet exist, create it,
          // or if it already exists, overwrite it.
          O_WRONLY | O_CREAT | O_TRUNC))) {
    Serial.println("Unknown error writing to flash");
    return false;
  }

  settings.WriteFile(
      [](const char* line) {
        return g_flash_file.write(line) >= 0 && g_flash_file.write('\n') == 1;
      });

  g_flash_file.close();
  last_written_settings_ = settings;
  Serial.println("FlashSettings: Wrote " kFlashSettingsFile);
  return true;
}

}  // namespace audio_tactile

