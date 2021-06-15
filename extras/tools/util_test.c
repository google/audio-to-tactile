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

#include "extras/tools/util.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

static void TestStringEqualIgnoreCase() {
  puts("TestStringEqualIgnoreCase");
  CHECK(StringEqualIgnoreCase("banana", "banana"));
  CHECK(StringEqualIgnoreCase("Banana", "baNANa"));
  CHECK(StringEqualIgnoreCase("", ""));

  CHECK(!StringEqualIgnoreCase("Banana", "xyz"));
  CHECK(!StringEqualIgnoreCase("Banana", ""));
  CHECK(!StringEqualIgnoreCase("", "Banana"));
}

static void TestFindSubstringIgnoreCase() {
  puts("TestFindSubstringIgnoreCase");
  const char* s = "banana!";
  CHECK(FindSubstringIgnoreCase(s, "nan") == s + 2);
  CHECK(FindSubstringIgnoreCase(s, "NaN") == s + 2);
  CHECK(FindSubstringIgnoreCase(s, "!") == s + 6);
  CHECK(FindSubstringIgnoreCase(s, "BANANA") == s);
  CHECK(FindSubstringIgnoreCase(s, "") == s);

  CHECK(FindSubstringIgnoreCase(s, "xyz") == NULL);
  CHECK(FindSubstringIgnoreCase(s, "banananana") == NULL);
}

/* Check StartsWith() function. */
static void TestStartsWith() {
  puts("TestStartsWith");
  CHECK(StartsWith("needle", "needle"));
  CHECK(StartsWith("needleXYZ", "needle"));
  CHECK(StartsWith("XYZ", ""));  /* Always true with an empty prefix. */

  CHECK(!StartsWith("NeedleXYZ", "needle"));
  CHECK(!StartsWith("XYZneedle", "needle"));
  CHECK(!StartsWith("", "needle"));
}

static void TestStartsWithIgnoreCase() {
  puts("TestStartsWithIgnoreCase");
  CHECK(StartsWithIgnoreCase("needle", "NEEDLE"));
  CHECK(StartsWithIgnoreCase("NeedleXYZ", "nEEdle"));
  CHECK(StartsWithIgnoreCase("XYZ", ""));

  CHECK(!StartsWithIgnoreCase("XYZneedle", "needle"));
  CHECK(!StartsWithIgnoreCase("", "needle"));
}

/* Check EndsWith() function. */
static void TestEndsWith() {
  puts("TestEndsWith");
  CHECK(EndsWith("needle", "needle"));
  CHECK(EndsWith("XYZneedle", "needle"));
  CHECK(EndsWith("XYZ", ""));  /* Always true with an empty suffix. */

  CHECK(!EndsWith("needleXYZ", "needle"));
  CHECK(!EndsWith("", "needle"));
}

/* Test ParseListOfInts() function. */
static void TestParseListOfInts() {
  puts("TestParseListOfInts");
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
static void TestParseListOfDoubles() {
  puts("TestParseListOfDoubles");
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

static void TestRoundUpToPowerOfTwo() {
  puts("TestRoundUpToPowerOfTwo");
  CHECK(RoundUpToPowerOfTwo(1) == 1);
  CHECK(RoundUpToPowerOfTwo(15) == 16);
  CHECK(RoundUpToPowerOfTwo(16) == 16);
  CHECK(RoundUpToPowerOfTwo(17) == 32);
  CHECK(RoundUpToPowerOfTwo(300) == 512);
  CHECK(RoundUpToPowerOfTwo(3000) == 4096);
  CHECK(RoundUpToPowerOfTwo(INT_MAX) > 0);
}

/* Check RandomInt() function with chi-squared goodness-of-fit test. */
static void TestRandomInt() {
  puts("TestRandomInt");
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
static void TestDecibelConversions() {
  puts("TestDecibelConversions");
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

#define kMaxFilterOrder 5

typedef struct {
  float smoother_coeff;
  float state[kMaxFilterOrder];
  int order;
} GammaFilter;

static float ApplyGammaFilter(GammaFilter* filter, float input_sample) {
  float next_stage_input = input_sample;
  const float smoother_coeff = filter->smoother_coeff;
  int k;
  for (k = 0; k < filter->order; ++k) {
    filter->state[k] += smoother_coeff * (next_stage_input - filter->state[k]);
    next_stage_input = filter->state[k];
  }
  return next_stage_input;
}

/* Check that GammaFilterSmootherCoeff() achieves the specified cutoff. */
static void TestGammaFilterSmootherCoeff() {
  puts("TestGammaFilterSmootherCoeff");
  static const float kCutoffs[] = {350.0f, 500.0f, 3200.0f};
  const float kSampleRateHz = 8000.0f;
  int i;
  for (i = 0; i < sizeof(kCutoffs) / sizeof(*kCutoffs); ++i) {
    const float cutoff_frequency_hz = kCutoffs[i];
    GammaFilter filter;
    for (filter.order = 1; filter.order <= kMaxFilterOrder; ++filter.order) {
      /* Get smoother_coeff for the specified order and cutoff. */
      filter.smoother_coeff = GammaFilterSmootherCoeff(
          filter.order, cutoff_frequency_hz, kSampleRateHz);
      int k;
      for (k = 0; k < filter.order; ++k) {
        filter.state[k] = 0.0f;
      }

      /* Estimate the filter's actual gain at cutoff_frequency_hz. */
      const float radians_per_sample =
          2 * M_PI * cutoff_frequency_hz / kSampleRateHz;
      const float period = kSampleRateHz / cutoff_frequency_hz;
      /* Number of initial burn-in samples to ignore transient effects. */
      const int kBurnInSamples = 4 * period;
      /* Number of samples to sum, close to a whole multiple of the period. */
      const int kNumSamples = (int)(period * (int)(2 + 100.0f / period) + 0.5f);
      float energy = 0.0f;
      int n;
      for (n = 0; n < kBurnInSamples + kNumSamples; ++n) {
        float input_sample = sin(radians_per_sample * n);
        float output_sample = ApplyGammaFilter(&filter, input_sample);
        if (n >= kBurnInSamples) {
          energy += output_sample * output_sample;
        }
      }
      const float estimated_gain = sqrt(2 * energy / kNumSamples);

      CHECK(fabs(estimated_gain - 1 / M_SQRT2) <= 5e-4);
    }
  }
}

/* Check TukeyWindow() with spot-checking a few samples. */
static void TestTukeyWindow() {
  puts("TestTukeyWindow");
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
static void TestPermuteWaveformChannels() {
  puts("TestPermuteWaveformChannels");
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

static void TestPrettyTextBar() {
  puts("TestPrettyTextBar");
  const int kWidth = 3;
  char* buffer = (char*)CHECK_NOTNULL(malloc(3 * kWidth + 1));

  /* Make a 3-char-wide bar filled 40%. */
  PrettyTextBar(kWidth, 0.4f, buffer);
  /* 40% fill is 3*8*0.4 = 9.6 eighth characters, so the bar should be rendered
   * as one full block, one 2/8th block, and one space.
   */
  CHECK(strcmp(buffer, "\xe2\x96\x88\xe2\x96\x8e ") == 0);

  /* Fill 0%. */
  PrettyTextBar(kWidth, 0.0f, buffer);
  CHECK(strcmp(buffer, "   ") == 0);

  /* Fill 120%. Should saturate to 100% full. */
  PrettyTextBar(kWidth, 1.2f, buffer);
  CHECK(strcmp(buffer, "\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88") == 0);

  free(buffer);
}

int main(int argc, char** argv) {
  TestStringEqualIgnoreCase();
  TestFindSubstringIgnoreCase();
  TestStartsWith();
  TestStartsWithIgnoreCase();
  TestEndsWith();
  TestParseListOfInts();
  TestParseListOfDoubles();
  TestRoundUpToPowerOfTwo();
  TestRandomInt();
  TestDecibelConversions();
  TestGammaFilterSmootherCoeff();
  TestTukeyWindow();
  TestPermuteWaveformChannels();
  TestPrettyTextBar();

  puts("PASS");
  return EXIT_SUCCESS;
}
