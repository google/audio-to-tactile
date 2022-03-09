/* Copyright 2022 Google LLC
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

#include "src/dsp/decibels.h"

#include <math.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

typedef struct {
  double power_ratio;
  double decibels;
} TestValue;
/* Table of values to test. */
static const TestValue kTestValues[] = {
  /* {power_ratio, decibels}. */
  {1e-7, -70.0},
  {1e-6, -60.0},
  {1e-5, -50.0},
  {1e-4, -40.0},
  {1e-3, -30.0},
  {1e-2, -20.0},
  {0.1, -10.0},
  {0.25, -6.020599913},
  {0.5, -3.010299957},
  {1 / M_SQRT2, -1.505149978},
  {1.0, 0.0},
  {M_SQRT2, 1.505149978},
  {2.0, 3.010299957},
  {4.0, 6.020599913},
  {10.0, 10.0},
  {1e2, 20.0},
  {1e3, 30.0},
  {1e4, 40.0},
};
static const int kNumTestValues = sizeof(kTestValues) / sizeof(TestValue);

static const int kNumRandomValues = 1000;

/* Random value uniformly in the range -60 to +60 dB. */
static double RandDb(void) { return -60.0 + (120.0 / RAND_MAX) * rand(); }

static int /*bool*/ IsClose(double actual, double expected, double tol) {
  double diff = actual - expected;
  if (!(fabs(diff) < tol)) {
    fprintf(stderr, "Error: difference %g exceeds tolerance %g\n"
                    "  actual = %.9g\n  expected = %.9g\n",
            diff, tol, actual, expected);
    return 0;
  }
  return 1;
}

static void TestPowerRatioToDecibels(void) {
  puts("TestPowerRatioToDecibels");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const double power_ratio = kTestValues[i].power_ratio;
    const double decibels = kTestValues[i].decibels;
    CHECK(IsClose(PowerRatioToDecibels(power_ratio), decibels, 1e-9));
  }
}

static void TestAmplitudeRatioToDecibels(void) {
  puts("TestAmplitudeRatioToDecibels");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const double amplitude_ratio = sqrt(kTestValues[i].power_ratio);
    const double decibels = kTestValues[i].decibels;
    CHECK(IsClose(AmplitudeRatioToDecibels(amplitude_ratio), decibels, 1e-9));
  }
}

static void TestDecibelsToPowerRatio(void) {
  puts("TestDecibelsToPowerRatio");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const double power_ratio = kTestValues[i].power_ratio;
    const double decibels = kTestValues[i].decibels;
    const double tol = 1e-9 * power_ratio;
    CHECK(IsClose(power_ratio, DecibelsToPowerRatio(decibels), tol));
  }
}

static void TestDecibelsToAmplitudeRatio(void) {
  puts("TestDecibelsToAmplitudeRatio");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const double amplitude_ratio = sqrt(kTestValues[i].power_ratio);
    const double decibels = kTestValues[i].decibels;
    const double tol = 1e-9 * amplitude_ratio;
    CHECK(IsClose(amplitude_ratio, DecibelsToAmplitudeRatio(decibels), tol));
  }
}

static void TestFastPowerRatioToDecibels(void) {
  puts("TestFastPowerRatioToDecibels");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const float power_ratio = kTestValues[i].power_ratio;
    const float decibels = kTestValues[i].decibels;
    CHECK(IsClose(FastPowerRatioToDecibels(power_ratio), decibels, 0.01f));
  }

  for (i = 0; i < kNumRandomValues; ++i) { /* Test random points. */
    const float decibels = RandDb();
    const float power_ratio = DecibelsToPowerRatio(decibels);
    CHECK(IsClose(FastPowerRatioToDecibels(power_ratio), decibels, 0.01f));
  }
}

static void TestFastAmplitudeRatioToDecibels(void) {
  puts("TestFastAmplitudeRatioToDecibels");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const float amplitude_ratio = sqrt(kTestValues[i].power_ratio);
    const float decibels = kTestValues[i].decibels;
    CHECK(IsClose(FastAmplitudeRatioToDecibels(amplitude_ratio),
                  decibels, 0.02f));
  }

  for (i = 0; i < kNumRandomValues; ++i) { /* Test random points. */
    const float decibels = RandDb();
    const float amplitude_ratio = DecibelsToAmplitudeRatio(decibels);
    CHECK(IsClose(FastAmplitudeRatioToDecibels(amplitude_ratio),
                  decibels, 0.02f));
  }
}

static void TestFastDecibelsToPowerRatio(void) {
  puts("TestFastDecibelsToPowerRatio");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const float power_ratio = kTestValues[i].power_ratio;
    const float decibels = kTestValues[i].decibels;
    const float tol = 0.003f * power_ratio;
    CHECK(IsClose(power_ratio, FastDecibelsToPowerRatio(decibels), tol));
  }

  for (i = 0; i < kNumRandomValues; ++i) { /* Test random points. */
    const float decibels = RandDb();
    const float power_ratio = DecibelsToPowerRatio(decibels);
    const float tol = 0.003f * power_ratio;
    CHECK(IsClose(power_ratio, FastDecibelsToPowerRatio(decibels), tol));
  }
}

static void TestFastDecibelsToAmplitudeRatio(void) {
  puts("TestFastDecibelsToAmplitudeRatio");
  int i;
  for (i = 0; i < kNumTestValues; ++i) {
    const float amplitude_ratio = sqrt(kTestValues[i].power_ratio);
    const float decibels = kTestValues[i].decibels;
    const float tol = 0.003f * amplitude_ratio;
    CHECK(IsClose(amplitude_ratio,
                  FastDecibelsToAmplitudeRatio(decibels), tol));
  }

  for (i = 0; i < kNumRandomValues; ++i) { /* Test random points. */
    const float decibels = RandDb();
    const float amplitude_ratio = DecibelsToAmplitudeRatio(decibels);
    const float tol = 0.003f * amplitude_ratio;
    CHECK(IsClose(amplitude_ratio,
                  FastDecibelsToAmplitudeRatio(decibels), tol));
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestPowerRatioToDecibels();
  TestAmplitudeRatioToDecibels();
  TestDecibelsToPowerRatio();
  TestDecibelsToAmplitudeRatio();

  TestFastPowerRatioToDecibels();
  TestFastAmplitudeRatioToDecibels();
  TestFastDecibelsToPowerRatio();
  TestFastDecibelsToAmplitudeRatio();

  puts("PASS");
  return EXIT_SUCCESS;
}
