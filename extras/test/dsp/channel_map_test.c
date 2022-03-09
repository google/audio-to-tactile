/* Copyright 2021-2022 Google LLC
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

#include "src/dsp/channel_map.h"

#include <math.h>

#include "src/dsp/decibels.h"
#include "src/dsp/logging.h"

static void TestChannelMapInit(void) {
  puts("TestChannelMapInit");
  const int kNumChannels = 5;
  ChannelMap channel_map;

  ChannelMapInit(&channel_map, kNumChannels);

  CHECK(channel_map.num_input_channels == kNumChannels);
  CHECK(channel_map.num_output_channels == kNumChannels);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    CHECK(channel_map.gains[c] == 1.0f);
    CHECK(channel_map.sources[c] == c);
  }
}

static void TestChannelMapApply(void) {
  const int kNumFrames = 20;
  const int kInputChannels = 3;
  const int kOutputChannels = 4;
  float* input = (float*)CHECK_NOTNULL(
      malloc(kNumFrames * kInputChannels * sizeof(float)));
  float* output = (float*)CHECK_NOTNULL(
      malloc(kNumFrames * kOutputChannels * sizeof(float)));
  int i;
  for (i = 0; i < kNumFrames * kInputChannels; ++i) {
    input[i] = (float) rand() / RAND_MAX; /* Generate random test input. */
  }
  ChannelMap channel_map;
  channel_map.num_input_channels = kInputChannels;
  channel_map.num_output_channels = kOutputChannels;

  channel_map.sources[0] = 2; /* Map source 2 to output 0. */
  channel_map.sources[1] = 2; /* Map source 2 to output 1. */
  channel_map.sources[2] = 0; /* Map source 0 to output 2. */
  channel_map.sources[3] = 1; /* Map source 1 to output 3. */
  channel_map.gains[0] = 0.2f;
  channel_map.gains[1] = 0.4f;
  channel_map.gains[2] = 0.6f;
  channel_map.gains[3] = 0.8f;

  ChannelMapApply(&channel_map, input, kNumFrames, output);

  for (i = 0; i < kNumFrames; ++i) {
    const float* in_frame = input + kInputChannels * i;
    const float* out_frame = output + kOutputChannels * i;
    CHECK(fabs(out_frame[0] - 0.2f * in_frame[2]) <= 1e-6f);
    CHECK(fabs(out_frame[1] - 0.4f * in_frame[2]) <= 1e-6f);
    CHECK(fabs(out_frame[2] - 0.6f * in_frame[0]) <= 1e-6f);
    CHECK(fabs(out_frame[3] - 0.8f * in_frame[1]) <= 1e-6f);
  }

  free(output);
  free(input);
}

/* Test control value to gain mapping. */
static void TestGainMapping(void) {
  puts("TestGainMapping");
  CHECK(ChannelGainFromControlValue(0) == 0.0f);
  CHECK(ChannelGainFromControlValue(63) == 1.0f);

  const float min_db = -18.0f;
  const float max_db = 0.0f;
  int control_value;
  for (control_value = 1; control_value <= 63; ++control_value) {
    float gain = ChannelGainFromControlValue(control_value);
    float expected_db = min_db + ((max_db - min_db) / 62) * (control_value - 1);
    /* `gain` has max error of 0.3%, which is an error of about 0.03 dB. */
    float tol_db = 0.03f;
    CHECK(fabs(AmplitudeRatioToDecibels(gain) - expected_db) <= tol_db);
  }
}

/* Test that control value -> gain -> control value is a round trip. */
static void TestGainControlValueRoundTrip(void) {
  puts("TestGainControlValueRoundTrip");

  int control_value;
  for (control_value = 0; control_value <= 63; ++control_value) {
    float gain = ChannelGainFromControlValue(control_value);
    int recovered = ChannelGainToControlValue(gain);
    CHECK(recovered == control_value);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestChannelMapInit();
  TestChannelMapApply();
  TestGainMapping();
  TestGainControlValueRoundTrip();

  puts("PASS");
  return EXIT_SUCCESS;
}
