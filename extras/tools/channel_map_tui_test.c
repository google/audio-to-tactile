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

#include "extras/tools/channel_map_tui.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "extras/tools/util.h"
#include "src/dsp/logging.h"

static void TestChannelMapParse() {
  puts("TestChannelMapParse");
  ChannelMap channel_map;
  int c;

  {
    CHECK(ChannelMapParse(4, "1,2,3,4,1,3", NULL, &channel_map));

    CHECK(channel_map.num_input_channels == 4);
    CHECK(channel_map.num_output_channels == 6);
    static const int kExpectedSources[6] = {0, 1, 2, 3, 0, 2};
    for (c = 0; c < 6; ++c) {
      CHECK(channel_map.sources[c] == kExpectedSources[c]);
      CHECK(fabs(channel_map.gains[c] - 1.0f) < 1e-7f);
    }
  }

  {
    CHECK(ChannelMapParse(9, "6,7,0,2", "-6.4,-2.3,-10", &channel_map));

    CHECK(channel_map.num_input_channels == 9);
    CHECK(channel_map.num_output_channels == 4);
    static const int kExpectedSources[4] = {5, 6, 0, 1};
    for (c = 0; c < 4; ++c) {
      CHECK(channel_map.sources[c] == kExpectedSources[c]);
    }

    CHECK(fabs(channel_map.gains[0] - DecibelsToAmplitudeRatio(-6.4f)) < 1e-7f);
    CHECK(fabs(channel_map.gains[1] - DecibelsToAmplitudeRatio(-2.3f)) < 1e-7f);
    CHECK(channel_map.gains[2] == 0.0f);
    CHECK(channel_map.gains[3] == 1.0f);
  }
}

static void TestChannelMapInit() {
  puts("TestChannelMapInit");
  const int kNumChannels = 4;
  const int kNumFrames = 6;

  float* input = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kNumChannels * kNumFrames));
  float* output = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kNumChannels * kNumFrames));
  int i;
  int c;
  for (i = 0; i < kNumFrames; ++i) {
    float* input_frame = input + kNumChannels * i;
    for (c = 0; c < kNumChannels; ++c) {
      input_frame[c] = i + 0.1f * c;
    }
  }

  ChannelMap channel_map;
  ChannelMapInit(&channel_map, kNumChannels);
  CHECK(channel_map.num_input_channels == kNumChannels);
  CHECK(channel_map.num_output_channels == kNumChannels);

  ChannelMapApply(&channel_map, input, kNumFrames, output);
  CHECK(memcmp(input, output, sizeof(float) * kNumChannels * kNumFrames) == 0);

  free(output);
  free(input);
}

static void TestChannelMapApply() {
  puts("TestChannelMapApply");
  const int kNumInputChannels = 5;
  const int kNumOutputChannels = 4;
  const int kNumFrames = 3;

  float* input = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kNumInputChannels * kNumFrames));
  float* output = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kNumOutputChannels * kNumFrames));
  int i;
  int c;
  for (i = 0; i < kNumFrames; ++i) {
    float* input_frame = input + kNumInputChannels * i;
    for (c = 0; c < kNumInputChannels; ++c) {
      input_frame[c] = i + 0.1f * c;
    }
  }

  ChannelMap channel_map;
  CHECK(ChannelMapParse(kNumInputChannels, "4,5,0,1", "-6.4,-2.3,0,1.1",
                        &channel_map));
  CHECK(channel_map.num_output_channels == kNumOutputChannels);

  ChannelMapApply(&channel_map, input, kNumFrames, output);

  const float* gains = channel_map.gains;
  const int* sources = channel_map.sources;
  for (i = 0; i < kNumFrames; ++i) {
    const float* input_frame = input + kNumInputChannels * i;
    const float* output_frame = output + kNumOutputChannels * i;

    for (c = 0; c < kNumOutputChannels; ++c) {
      const float expected = gains[c] * input_frame[sources[c]];
      CHECK(fabs(output_frame[c] - expected) < 1e-6f);
    }
  }

  free(output);
  free(input);
}

int main(int argc, char** argv) {
  TestChannelMapParse();
  TestChannelMapInit();
  TestChannelMapApply();

  puts("PASS");
  return EXIT_SUCCESS;
}
