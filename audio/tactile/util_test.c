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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "audio/dsp/portable/logging.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* Check StartsWith() function. */
void TestStartsWith() {
  CHECK(StartsWith("needle", "needle"));
  CHECK(StartsWith("needleXYZ", "needle"));
  CHECK(StartsWith("XYZ", ""));  /* Always true with an empty prefix. */

  CHECK(!StartsWith("XYZneedle", "needle"));
  CHECK(!StartsWith("", "needle"));
}

/* Test ParseListOfInts() function. */
void TestParseListOfInts() {
  int num_ints;
  int* result = CHECK_NOTNULL(ParseListOfInts("756,0,-32,1", &num_ints));

  CHECK(num_ints == 4);
  CHECK(result[0] == 756);
  CHECK(result[1] == 0);
  CHECK(result[2] == -32);
  CHECK(result[3] == 1);

  CHECK(ParseListOfInts("", &num_ints) == NULL);
  CHECK(ParseListOfInts("756,0,-32,1XX", &num_ints) == NULL);
  CHECK(ParseListOfInts("756,10000000000000000000,-32,1", &num_ints) == NULL);
  CHECK(ParseListOfInts("756,0.5,-32,1", &num_ints) == NULL);
  CHECK(ParseListOfInts(NULL, &num_ints) == NULL);
  CHECK(ParseListOfInts("756,0,-32,1", NULL) == NULL);

  free(result);
}

/* Test ParseListOfDoubles() function. */
void TestParseListOfDoubles() {
  int num_doubles;
  double* result = CHECK_NOTNULL(
      ParseListOfDoubles("42,-0.01,7.2e-4", &num_doubles));

  CHECK(num_doubles == 3);
  CHECK(fabs(result[0] - 42) < 1e-16);
  CHECK(fabs(result[1] - -0.01) < 1e-16);
  CHECK(fabs(result[2] - 7.2e-4) < 1e-16);

  CHECK(ParseListOfDoubles("", &num_doubles) == NULL);
  CHECK(ParseListOfDoubles("42,-0.01,7.2e-4nonsense", &num_doubles) == NULL);
  CHECK(ParseListOfDoubles("42,nonsense,7.2e-4", &num_doubles) == NULL);
  CHECK(ParseListOfDoubles(NULL, &num_doubles) == NULL);
  CHECK(ParseListOfDoubles("42,-0.01,7.2e-4", NULL) == NULL);

  free(result);
}

/* Check RandomInt() function with chi-squared goodness-of-fit test. */
void TestRandomInt() {
  const int kNumSamples = 500;
  int hist[101];

  int i;
  static const int kMaxValuesList[] = {0, 1, 2, 10, 100};
  for (i = 0; i < sizeof(kMaxValuesList) / sizeof(*kMaxValuesList); ++i) {
    const int max_value = kMaxValuesList[i];
    memset(hist, 0, (max_value + 1) * sizeof(int));

    int j;
    for (j = 0; j < kNumSamples; ++j) {
      const int value = RandomInt(max_value);
      CHECK(0 <= value && value <= max_value);
      ++hist[value];  /* Accumulate histogram. */
    }

    /* Compute chi-squared statistic between hist and expected distribution. */
    float expected_frequency = kNumSamples / (max_value + 1.0f);
    float chi2_stat = 0.0f;
    for (j = 0; j <= max_value; ++j) {
      const float diff = hist[j] - expected_frequency;
      chi2_stat += diff * diff;
    }
    chi2_stat /= expected_frequency;

    /* Check goodness-of-fit with 0.1% significance level. */
    CHECK(chi2_stat <= 150);
  }
}

/* Check AmplitudeRatioToDecibels and DecibelsToAmplitudeRatio. */
void TestDecibelConversions() {
  CHECK(fabs(AmplitudeRatioToDecibels(10.0f) - 20.0f) < 1e-6f);
  CHECK(fabs(DecibelsToAmplitudeRatio(20.0f) - 10.0f) < 1e-6f);
  CHECK(fabs(AmplitudeRatioToDecibels(2.0f) - 6.0206f) < 1e-6f);
  CHECK(fabs(DecibelsToAmplitudeRatio(6.0f) - 1.995262f) < 1e-6f);

  int i;
  for (i = 0; i < 20; ++i) {
    float decibels = -40.0f + (80.0f * rand()) / RAND_MAX;
    float amplitude_ratio = DecibelsToAmplitudeRatio(decibels);
    CHECK(fabs(AmplitudeRatioToDecibels(amplitude_ratio) - decibels) < 1e-6f);
  }
}

/* Check TukeyWindow() with spot-checking a few samples. */
void TestTukeyWindow() {
  const float kDuration = 0.4f;
  int i;
  static const float kTransitionsList[] = {0.05f, 0.1f, 0.2f};
  for (i = 0; i < sizeof(kTransitionsList) / sizeof(*kTransitionsList); ++i) {
    const float transition = kTransitionsList[i];

    CHECK(fabs(TukeyWindow(kDuration, transition, -2.0f)) <= 1e-12f);
    CHECK(fabs(TukeyWindow(kDuration, transition, -0.1f)) <= 1e-12f);
    CHECK(fabs(TukeyWindow(kDuration, transition, 0.0f)) <= 1e-12f);

    int j;
    for (j = 1; j < 8; ++j) {
      const float x = j / 8.0f;
      const float expected = (1 - cos(M_PI * x)) / 2;

      float t = x * transition;
      CHECK(fabs(TukeyWindow(kDuration, transition, t) - expected) <= 1e-6f);
      t = kDuration - x * transition;
      CHECK(fabs(TukeyWindow(kDuration, transition, t) - expected) <= 1e-6f);
    }

    for (j = 0; j < 8; ++j) {
      const float t = transition + (kDuration - 2 * transition) * j / 8.0f;
      CHECK(fabs(TukeyWindow(kDuration, transition, t) - 1.0f) <= 1e-12f);
    }

    CHECK(fabs(TukeyWindow(kDuration, transition, 0.4f)) <= 1e-12f);
    CHECK(fabs(TukeyWindow(kDuration, transition, 0.5f)) <= 1e-12f);
    CHECK(fabs(TukeyWindow(kDuration, transition, 2.4f)) <= 1e-12f);
  }
}

/* Test permuting a 4-channel waveform to the order 3, 1, 0, 2. */
void TestPermuteWaveformChannels() {
  /* 3 frames, 4 channels. */
  float samples[3 * 4] = {
      10.0f, 20.0f, 30.0f, 40.0f,  /* Frame 0. */
      11.0f, 21.0f, 31.0f, 41.0f,  /* Frame 1. */
      12.0f, 22.0f, 32.0f, 42.0f,  /* Frame 2. */
  };

  const int kPermutation[] = {3, 1, 0, 2};
  PermuteWaveformChannels(kPermutation, samples, 3, 4);

  static const float kExpected[3 * 4] = {
      40.0f, 20.0f, 10.0f, 30.0f,  /* Frame 0. */
      41.0f, 21.0f, 11.0f, 31.0f,  /* Frame 1. */
      42.0f, 22.0f, 12.0f, 32.0f,  /* Frame 2. */
  };
  CHECK(memcmp(samples, kExpected, 3 * 4 * sizeof(float)) == 0);
}

int main(int argc, char** argv) {
  TestStartsWith();
  TestParseListOfInts();
  TestParseListOfDoubles();
  TestRandomInt();
  TestDecibelConversions();
  TestTukeyWindow();
  TestPermuteWaveformChannels();

  puts("PASS");
  return EXIT_SUCCESS;
}
