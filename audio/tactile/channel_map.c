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

#include "audio/tactile/channel_map.h"

#include <stdio.h>
#include <stdlib.h>

#include "audio/tactile/util.h"

int ChannelMapParse(int num_input_channels, const char* source_list,
    const char* gains_db_list, ChannelMap* channel_map) {
  if (source_list == NULL || channel_map == NULL) {
    fprintf(stderr, "ParseChannelMap: Null argument.\n");
    return 0;
  }
  channel_map->num_input_channels = num_input_channels;
  channel_map->num_output_channels = 0;

  int* sources = NULL;
  int num_output_channels = 0;
  double* gains_db = NULL;
  int num_gains_db = 0;
  int success = 0;

  if (!(sources = ParseListOfInts(source_list, &num_output_channels)) ||
      num_output_channels > kChannelMapMaxChannels) {
    fprintf(stderr, "ParseChannelMap: Invalid sources: \"%s\"\n",
        source_list);
    goto done;
  } else if (gains_db_list && *gains_db_list &&
      !(gains_db = ParseListOfDoubles(gains_db_list, &num_gains_db))) {
    fprintf(stderr, "ParseChannelMap: Invalid gains: \"%s\"\n",
        gains_db_list);
    goto done;
  }

  ChannelMapEntry* channels = channel_map->channels;
  int c;
  for (c = 0; c < num_output_channels; ++c) {
    if (sources[c] == 0) {  /* Channel c is disabled. */
      channels[c].gain = 0.0f;
      channels[c].source = 0;
    } else if (sources[c] <= num_input_channels) {
      const float gain_db = (c < num_gains_db) ? gains_db[c] : 0.0f;
      channels[c].gain = DecibelsToAmplitudeRatio(gain_db);
      channels[c].source = sources[c] - 1;  /* Convert to base-0 index. */
    } else {
      fprintf(stderr, "ParseChannelMap: Source %d is invalid in \"%s\"\n",
          sources[c], source_list);
      goto done;
    }
  }

  channel_map->num_output_channels = num_output_channels;
  success = 1;
done:
  free(gains_db);
  free(sources);
  return success;
}

void ChannelMapPrint(const ChannelMap* channel_map) {
  if (channel_map != NULL &&
      channel_map->num_output_channels <= kChannelMapMaxChannels) {
    const ChannelMapEntry* channels = channel_map->channels;
    int c;
    for (c = 0; c < channel_map->num_output_channels; ++c) {
      if (channels[c].gain > 0.0f) {
        printf("  channel %2d: signal %2d, gain %+.1f dB\n",
            c + 1, channels[c].source + 1,
            AmplitudeRatioToDecibels(channels[c].gain));
      } else {
        printf("  channel %2d: off\n", c + 1);
      }
    }
  }
}

void ChannelMapApply(const ChannelMap* channel_map,
    const float* input, int num_frames, float* output) {
  const ChannelMapEntry* channels = channel_map->channels;
  const int num_input_channels = channel_map->num_input_channels;
  const int num_output_channels = channel_map->num_output_channels;
  int i;
  for (i = 0; i < num_frames; ++i) {
    int c;
    for (c = 0; c < num_output_channels; ++c) {
      output[c] = channels[c].gain * input[channels[c].source];
    }
    input += num_input_channels;
    output += num_output_channels;
  }
}
