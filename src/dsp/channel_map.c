/* Copyright 2019, 2021 Google LLC
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

#include "dsp/channel_map.h"

void ChannelMapInit(ChannelMap* channel_map, int num_channels) {
  channel_map->num_input_channels = num_channels;
  channel_map->num_output_channels = num_channels;

  int c;
  for (c = 0; c < num_channels; ++c) {
    channel_map->gains[c] = 1.0f;
    channel_map->sources[c] = c;
  }
}

void ChannelMapApply(const ChannelMap* channel_map, const float* input,
                     int num_frames, float* output) {
  const float* gains = channel_map->gains;
  const int* sources = channel_map->sources;
  const int num_input_channels = channel_map->num_input_channels;
  const int num_output_channels = channel_map->num_output_channels;
  int i;
  for (i = 0; i < num_frames; ++i) {
    int c;
    for (c = 0; c < num_output_channels; ++c) {
      output[c] = gains[c] * input[sources[c]];
    }
    input += num_input_channels;
    output += num_output_channels;
  }
}
