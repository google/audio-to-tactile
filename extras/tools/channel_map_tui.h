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
 * Text user interface (TUI) for channel_map.
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_TOOLS_CHANNEL_MAP_TUI_H_
#define AUDIO_TO_TACTILE_EXTRAS_TOOLS_CHANNEL_MAP_TUI_H_

#include "src/dsp/channel_map.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parses `ChannelMap` from a comma-delimited list of base-1 channel sources and
 * a comma-delimited list of channel gains in decibels. This is useful for user
 * interface, e.g. taking the two lists as command line arguments. Supports up
 * to `kChannelMapMaxChannels` channels. Returns 1 on success, 0 on failure.
 *
 * NOTE: `source_list` is base-1 indexed, while sources in the parsed ChannelMap
 * are base-0 indexed. Base 1 is preferable for user interface, since it is the
 * convention on the Motu and other audio interface hardware, while base 0 is
 * better for implementation.
 *
 * Details:
 *  - A "0" in `source_list` means the output channel is filled with zeros.
 *  - If `gains_db_list` is shorter than `source_list`, remaining channels have
 *    0 dB gain. If gains_db_list is longer, excess elements are ignored.
 *
 * Examples:
 *
 * ChannelMapParse(3, "3,1,2,2", "-1.5,-7.2,-8,-3", &channel_map)
 * defines a map from 3-channel input to 4-channel output (written in base 0) as
 *   output[0] = input[2] * 10^(-1.5/20),
 *   output[1] = input[0] * 10^(-7.2/20),
 *   output[2] = input[1] * 10^(-8/20),
 *   output[3] = input[1] * 10^(-3/20).
 *
 * ChannelMapParse(2, "1,0,2", "-5.1", &channel_map)
 * defines a map from stereo input to 3-channel output (written in base 0) as
 *   output[0] = input[0] * 10^(-5.1/20),
 *   output[1] = 0,
 *   output[2] = input[1].
 */
int ChannelMapParse(int num_input_channels, const char* source_list,
                    const char* gains_db_list, ChannelMap* channel_map);

/* Prints `channel_map` to stdout. */
void ChannelMapPrint(const ChannelMap* channel_map);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_EXTRAS_TOOLS_CHANNEL_MAP_TUI_H_ */
