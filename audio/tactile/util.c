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

/* Parses a comma-delimited list by calling a function `parse_fun` on each item:
 *
 *   int parse_fun(const char* s, const char** s_end, void* out)
 *
 * `s` points to the beginning of an item. `*s_end` is set by `parse_fun` to the
 * character immediately following the item. The parsed item is written to
 * `*out`. The return value indicates success (1) or failure (0).
 */
static void* ParseListGeneric(const char* s, int* length,
    size_t item_size, int (*parse_fun)(const char*, const char**, void*)) {
  if (s == NULL || length == NULL) {
    return NULL;
  }
  *length = 1 + CountNumCommas(s);

  void* parsed_items = malloc(*length * item_size);
  if (parsed_items == NULL) { return NULL; }

  int i;
  for (i = 0; i < *length; ++i) {
    /* Item should be followed by ',' or the null terminator. */
    const char expected_next_char = (i < *length - 1) ? ',' : '\0';
    const char* s_end;
    if (!parse_fun(s, &s_end, parsed_items + item_size * i) ||
        *s_end != expected_next_char) {
      free(parsed_items);
      return NULL;  /* Parsing failed. */
    }
    s = s_end + 1;
  }

  return parsed_items;
}

static int ParseInt(const char* s, const char** s_end, void* out) {
  /* There's no safe standard C function for parsing ints, so parse a long. */
  const long parsed_value = strtol(s, (char**)s_end, /*base*/ 10);

  /* strtol() indicates error by setting s_end = s. An edge case is that the
   * string represents a valid long [so strtol() succeeds], but would overflow
   * an int, so we check for this as well.
   */
  if (*s_end == s || !(INT_MIN <= parsed_value && parsed_value <= INT_MAX)) {
    return 0;  /* Integer conversion failed or is out of range. */
  }

  *((int*)out) = (int)parsed_value;
  return 1;
}

int* ParseListOfInts(const char* s, int* length) {
  return (int*)ParseListGeneric(s, length, sizeof(int), ParseInt);
}

static int ParseDouble(const char* s, const char** s_end, void* out) {
  *((double*)out) = strtod(s, (char**)s_end);
  return *s_end != s;  /* strtod() indicates error by setting s_end = s. */
}

double* ParseListOfDoubles(const char* s, int* length) {
  return (double*)ParseListGeneric(s, length, sizeof(double), ParseDouble);
}

int RandomInt(int max_value) {
  int result = (int) (((1.0f + max_value) / (1.0f + RAND_MAX)) * rand());
  return result <= max_value ? result : max_value;
}

float DecibelsToAmplitudeRatio(float decibels) {
  return exp((M_LN10 / 20) * decibels);
}

float AmplitudeRatioToDecibels(float amplitude_ratio) {
  return (20 / M_LN10) * log(amplitude_ratio);
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
