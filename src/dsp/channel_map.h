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
 *
 *
 * Channel map function to remap channel indices and apply gains.
 *
 * This library enables flexible playback of multichannel signals, for instance
 * playing TactileProcessor's 10-channel signal on a 24-channel device, with
 * per-channel adjustable gain.
 *
 * `ChannelMap` describes remapping and gains on a multichannel audio or tactile
 * signal of the form
 *
 *   output[c] = gains[c] * input[sources[c]],
 *
 * where `sources[c]` is the input channel index to map to output channel `c`,
 * and `gains[c]` is a multiplied gain factor. No clipping is performed.
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_CHANNEL_MAP_H_
#define AUDIO_TO_TACTILE_SRC_DSP_CHANNEL_MAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#define kChannelMapMaxChannels 32

typedef struct {
  int num_input_channels;
  int num_output_channels;
  /* Channel gains as linear amplitude ratios. */
  float gains[kChannelMapMaxChannels];
  /* Input source channels as base-0 indices. */
  int sources[kChannelMapMaxChannels];
} ChannelMap;

/* Sets identity mapping with unit gains for `num_channels` channels. */
void ChannelMapInit(ChannelMap* channel_map, int num_channels);

/* Applies source map and gains described by `channel_map`. */
void ChannelMapApply(const ChannelMap* channel_map, const float* input,
                     int num_frames, float* output);

/* Maps a control value in the range 0-63 to a linear gain. Control value 0 maps
 * to gain 0.0. Control values 1-63 map linearly in dB space to -18 to 0 dB, and
 * is converted to linear gain. This is used to serialize channel maps.
 */
float ChannelGainFromControlValue(int control_value);

/* The inverse of ChannelGainFromControlValue(). */
int ChannelGainToControlValue(float gain);


#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_DSP_CHANNEL_MAP_H_ */
