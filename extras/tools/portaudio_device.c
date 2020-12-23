/* Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "extras/tools/portaudio_device.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "extras/tools/util.h"
#include "portaudio.h"

static int HasEnoughChannels(const PaDeviceInfo* device_info,
                             int min_input_channels, int min_output_channels) {
  return device_info->maxInputChannels >= min_input_channels &&
      device_info->maxOutputChannels >= min_output_channels;
}

int FindPortAudioDevice(
    const char* name, int min_input_channels, int min_output_channels) {
  if (name == NULL || *name == '\0') { return -1; }
  const int num_devices = Pa_GetDeviceCount();
  if (num_devices <= 0) { return -1; }

  /* Try to parse `name` as an integer. */
  char* s_end;
  const long parsed_index = strtol(name, &s_end, /*base*/ 10);
  if (*s_end == '\0') {
    if (0 <= parsed_index && parsed_index < num_devices &&
        HasEnoughChannels(Pa_GetDeviceInfo(parsed_index),
                          min_input_channels, min_output_channels)) {
      return parsed_index;
    }
    return -1;
  }

  /* Look for an exact string match. */
  int i;
  for (i = 0; i < num_devices; ++i) {
    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
    if (StringEqualIgnoreCase(device_info->name, name)) {
      return HasEnoughChannels(
          device_info, min_input_channels, min_output_channels) ? i : -1;
    }
  }

  /* Look for a substring match. */
  for (i = 0; i < num_devices; ++i) {
    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
    if (FindSubstringIgnoreCase(device_info->name, name) &&
        HasEnoughChannels(device_info, min_input_channels,
                          min_output_channels)) {
      return i;
    }
  }

  return -1;
}

void PrintPortAudioDevices() {
  const int num_devices = Pa_GetDeviceCount();
  if (num_devices < 0) {
    fprintf(stderr, "Error: %s\n", Pa_GetErrorText(num_devices));
    return;
  } else if (num_devices == 0) {
    fprintf(stderr, "Error: PortAudio found no devices.\n");
    return;
  }

  int i;
  for (i = 0; i < num_devices; ++i) {
    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
    printf("#%-2d %-46s channels: %3d in %3d out\n",
        i, device_info->name,
        device_info->maxInputChannels,
        device_info->maxOutputChannels);
  }
}
