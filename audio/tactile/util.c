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

#include "audio/tactile/util.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

int StartsWith(const char* s, const char* prefix) {
  while (*prefix) {
    if (*s++ != *prefix++) {
      return 0;
    }
  }
  return 1;
}

/* Counts the number of comma characters in s. */
static int CountNumCommas(const char* s) {
  int count = 0;
  while ((s = strchr(s, ',')) != NULL) {
    ++count;
    ++s;
  }
  return count;
}

int* ParseListOfInts(const char* s, int* length) {
  if (s == NULL || length == NULL) {
    return NULL;
  }
  *length = 1 + CountNumCommas(s);

  int* parsed_ints = (int*)malloc(*length * sizeof(int));
  if (parsed_ints == NULL) { return NULL; }

  int i;
  for (i = 0; i < *length; ++i) {
    char* s_end;
    /* There's no safe standard C function for parsing ints, so parse a long. */
    const long next_integer = strtol(s, &s_end, /*base*/ 10);

    /* strtol() indicates error by setting s_end = s. An edge case is that the
     * string represents a valid long [so strtol() succeeds], but would overflow
     * an int, so we check for this as well.
     */
    if (s_end == s || !(INT_MIN <= next_integer && next_integer <= INT_MAX)) {
      return 0;  /* Integer conversion failed or is out of range. */
    }
    /* Int should be followed by ',' or the null terminator. */
    const char next_char = (i < *length - 1) ? ',' : '\0';
    if (*s_end != next_char) {
      free(parsed_ints);
      return NULL;
    }

    parsed_ints[i] = (int)next_integer;
    s = s_end + 1;
  }

  return parsed_ints;
}

int RandomInt(int max_value) {
  int result = (int) (((1.0f + max_value) / (1.0f + RAND_MAX)) * rand());
  return result <= max_value ? result : max_value;
}

float TukeyWindow(float window_duration, float transition, float t) {
  if (!(0.0f <= t && t <= window_duration)) {
    return 0.0f;
  } else if (t > window_duration - transition) {
    t = window_duration - t;
  } else if (t >= transition) {
    return 1.0f;
  }
  return (1 - cos(M_PI * t / transition)) / 2;
}

void PermuteWaveformChannels(const int* permutation, float* waveform,
                             int num_frames, int num_channels) {
  float input_frame[32];
  const int kMaxChannels = sizeof(input_frame) / sizeof(*input_frame);
  (void)kMaxChannels;
  assert(1 <= num_channels && num_channels <= kMaxChannels);
  int i;
  for (i = 0; i < num_frames; ++i, waveform += num_channels) {
    memcpy(input_frame, waveform, num_channels * sizeof(float));
    int c;
    for (c = 0; c < num_channels; ++c) {
      waveform[c] = input_frame[permutation[c]];
    }
  }
}
