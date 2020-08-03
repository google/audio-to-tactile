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
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "audio/dsp/portable/math_constants.h"

int StringEqualIgnoreCase(const char* s1, const char* s2) {
  for (; *s1 && *s2; ++s1, ++s2) {
    if (tolower(*s1) != tolower(*s2)) {
      return 0;
    }
  }
  return *s1 == '\0' && *s2 == '\0';
}

const char* FindSubstringIgnoreCase(const char* s, const char* substring) {
  if (strlen(s) >= strlen(substring)) {
    for (; *s; ++s) {
      if (StartsWithIgnoreCase(s, substring)) {
        return s;
      }
    }
  }
  return NULL;
}

int StartsWith(const char* s, const char* prefix) {
  while (*prefix) {
    if (*s++ != *prefix++) {
      return 0;
    }
  }
  return 1;
}

int StartsWithIgnoreCase(const char* s, const char* prefix) {
  while (*prefix) {
    if (tolower(*s++) != tolower(*prefix++)) {
      return 0;
    }
  }
  return 1;
}

int EndsWith(const char* s, const char* suffix) {
  const int suffix_len = strlen(suffix);
  if (suffix_len == 0) { return 1; }
  const int s_len = strlen(s);
  if (s_len < suffix_len) { return 0; }
  return memcmp(s + s_len - suffix_len, suffix, suffix_len) == 0;
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

int RoundUpToPowerOfTwo(int value) {
  if (value <= 0) {
    return 0;
  } else if (value > INT_MAX / 2) {
    value = INT_MAX / 2;  /* Avoid infinite loop. */
  }

  int result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

int RandomInt(int max_value, int /* bool */ is_init) {
  if (is_init == 1) {
    /* Initialize random seed */
    srand((unsigned int)time(NULL));
  }

  int result = (int) (((1.0f + max_value) / (1.0f + RAND_MAX)) * rand());
  return result <= max_value ? result : max_value;
}

float DecibelsToAmplitudeRatio(float decibels) {
  return exp((M_LN10 / 20) * decibels);
}

float AmplitudeRatioToDecibels(float amplitude_ratio) {
  return (20 / M_LN10) * log(amplitude_ratio);
}

float GammaFilterSmootherCoeff(int order,
                               float cutoff_frequency_hz,
                               float sample_rate_hz) {
  const double theta = cutoff_frequency_hz * 2.0 * M_PI / sample_rate_hz;
  /* The filter's Z transform is H(z) = ((1 - p) / (1 - p z^-1))^order. We solve
   * for the pole p such that it has half power at the specified cutoff:
   *
   *   |H(exp(i theta)|^2 = ((1 - p)^2 / (1 - 2p cos(theta) + p^2))^order = 1/2.
   *
   * This equation simplifies to (1 + p^2) / 2 = q p where
   * q = (1 - single_stage_power cos(theta)) / (1 - single_stage_power).
   */
  const double single_stage_power = pow(0.5, 1.0 / order);
  const double q = (1.0 - single_stage_power * cos(theta))
      / (1.0 - single_stage_power);
  /* Solve for p in (1 + p^2) / 2 = q p. */
  const double p = q - sqrt(q * q - 1.0);
  return (float)(1.0 - p);
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
  (void)kMaxChannels; /* Suppress unused variable warning in nondebug builds. */
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

void PrettyTextBar(int width, float fraction, char* buffer) {
  const int max_level = 8 * width;
  if (fraction < 0.0f) { fraction = 0.0f; }
  if (fraction > 1.0f) { fraction = 1.0f; }
  int level = (int)(max_level * fraction + 0.5f);

  int i;
  for (i = 0; i < width; ++i) {
    int char_fill = level;
    if (char_fill == 0) {
      *(buffer++) = ' ';  /* Print an unfilled char as a space. */
    } else {
      /* Unicode characters 2588 to 258F are partially-filled blocks in 1/8th
       * steps, where 258F is 1/8th filled and 2588 is completely filled.
       */
      if (char_fill > 8) { char_fill = 8; }
      level -= char_fill;
      *(buffer++) = '\xe2';  /* Write block character as 3-byte UTF-8 code. */
      *(buffer++) = '\x96';
      *(buffer++) = '\x90' - char_fill;
    }
  }

  *buffer = '\0';
}
