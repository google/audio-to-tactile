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
 *
 *
 * Channel map function to remap channel indices and apply gains.
 *
 * `ChannelMap` describes remapping and gains on a multichannel audio or tactile
 * signal of the form
 *
 *   output[c] = gain[c] * input[source[c]],
 *
 * where `source[c]` is the input channel index to map to output channel `c`,
 * and `gain[c]` is a multiplied gain factor.
 */

#ifndef AUDIO_TACTILE_CHANNEL_MAP_H_
#define AUDIO_TACTILE_CHANNEL_MAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#define kChannelMapMaxChannels 32

typedef struct {
  float gain;  /* Channel gain as a linear amplitude ratio. */
  int source;  /* Input source channel as a base-0 index. */
} ChannelMapEntry;

typedef struct {
  ChannelMapEntry channels[kChannelMapMaxChannels];
  int num_input_channels;
  int num_output_channels;
} ChannelMap;

/* Parses a `ChannelMap` from a comma-delimited list of base-1 channel sources
 * and a comma-delimited list of channel gains in Decibels. Supports up to
 * `kChannelMapMaxChannels` channels. Returns 1 on success, 0 on failure.
 */
int ChannelMapParse(int num_input_channels, const char* source_list,
    const char* gains_db_list, ChannelMap* channel_map);

/* Prints `channel_map` to stdout. */
void ChannelMapPrint(const ChannelMap* channel_map);

/* Applies source map and gains described by `channel_map`. `input` is the
 * source waveform with `num_input_channels * num_frames` samples. `output` is
 * the resulting waveform with `num_output_channels * num_frames` samples.
 */
void ChannelMapApply(const ChannelMap* channel_map,
    float* input, int num_frames, float* output);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TACTILE_CHANNEL_MAP_H_ */
