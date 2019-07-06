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

#include <math.h>
#include <stdlib.h>

#include "audio/dsp/portable/logging.h"
#include "audio/tactile/util.h"

void TestChannelMapParse() {
  ChannelMap channel_map;
  int c;

  {
    CHECK(ChannelMapParse(4, "1,2,3,4,1,3", NULL, &channel_map));

    CHECK(channel_map.num_input_channels == 4);
    CHECK(channel_map.num_output_channels == 6);
    static const int kExpectedSources[6] = {0, 1, 2, 3, 0, 2};
    for (c = 0; c < 6; ++c) {
      CHECK(channel_map.channels[c].source == kExpectedSources[c]);
      CHECK(fabs(channel_map.channels[c].gain - 1.0f) < 1e-7f);
    }
  }

  {
    CHECK(ChannelMapParse(9, "6,7,0,2", "-6.4,-2.3,-10", &channel_map));

    CHECK(channel_map.num_input_channels == 9);
    CHECK(channel_map.num_output_channels == 4);
    static const int kExpectedSources[4] = {5, 6, 0, 1};
    for (c = 0; c < 4; ++c) {
      CHECK(channel_map.channels[c].source == kExpectedSources[c]);
    }

    CHECK(fabs(channel_map.channels[0].gain -
          DecibelsToAmplitudeRatio(-6.4f)) < 1e-7f);
    CHECK(fabs(channel_map.channels[1].gain -
          DecibelsToAmplitudeRatio(-2.3f)) < 1e-7f);
    CHECK(channel_map.channels[2].gain == 0.0f);
    CHECK(channel_map.channels[3].gain == 1.0f);
  }
}

void TestChannelMapApply() {
  const int kNumInputChannels = 5;
  const int kNumOutputChannels = 4;
  const int kNumFrames = 3;

  float* input = (float*)CHECK_NOTNULL(malloc(
        sizeof(float) * kNumInputChannels * kNumFrames));
  float* output = (float*)CHECK_NOTNULL(malloc(
        sizeof(float) * kNumOutputChannels * kNumFrames));
  int i;
  int c;
  for (i = 0; i < kNumFrames; ++i) {
    float* input_frame = input + kNumInputChannels * i;
    for (c = 0; c < kNumInputChannels; ++c) {
      input_frame[c] = i + 0.1f * c;
    }
  }

  ChannelMap channel_map;
  CHECK(ChannelMapParse(
        kNumInputChannels, "4,5,0,1", "-6.4,-2.3,0,1.1", &channel_map));
  CHECK(channel_map.num_output_channels == kNumOutputChannels);

  ChannelMapApply(&channel_map, input, kNumFrames, output);

  const ChannelMapEntry* channels = channel_map.channels;
  for (i = 0; i < kNumFrames; ++i) {
    const float* input_frame = input + kNumInputChannels * i;
    const float* output_frame = output + kNumOutputChannels * i;

    for (c = 0; c < kNumOutputChannels; ++c) {
      const float expected = channels[c].gain * input_frame[channels[c].source];
      CHECK(fabs(output_frame[c] - expected) < 1e-6f);
    }
  }

  free(output);
  free(input);
}

int main(int argc, char** argv) {
  TestChannelMapParse();
  TestChannelMapApply();

  puts("PASS");
  return EXIT_SUCCESS;
}
