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

#include "dsp/fast_fun.h"
#include "dsp/math_constants.h"

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

int ChannelGainToControlValue(float gain) {
  if (gain >= 1.0f) { return 63; }
  if (!(gain >= 0.128f)) { return (gain >= 0.05f) ? 1 : 0; }
  const float gain_db = (float)(20.0 * M_LN2 / M_LN10) * FastLog2(gain);
  return (int)((62.0f / 18.0f) * gain_db + 63.5f);
}

float ChannelGainFromControlValue(int control_value) {
  if (control_value <= 0) { return 0.0f; }
  if (control_value >= 63) { return 1.0f; }
  const float gain_db = (18.0f / 62.0f) * (control_value - 63);
  return FastExp2((float)(M_LN10 / (20.0 * M_LN2)) * gain_db);
}
