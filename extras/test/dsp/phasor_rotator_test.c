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

#include "src/dsp/phasor_rotator.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/dsp/complex.h"
#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

static double RandUniform() { return (double) rand() / RAND_MAX; }

void TestBasic() {
  PhasorRotator oscillator;
  double max_error = 0.0;
  int trial;

  for (trial = 0; trial < 10; ++trial) {
    const double sample_rate_hz = 1000 + 47000 * RandUniform();
    const double frequency_hz = 0.05 + 0.4 * sample_rate_hz * RandUniform();
    const double radians_per_sample = 2 * M_PI * frequency_hz / sample_rate_hz;
    PhasorRotatorInit(&oscillator, frequency_hz, sample_rate_hz);

    /* Generate 4000 samples of a complex exponential. */
    int i;
    for (i = 0; i < 4000; ++i) {
      ComplexDouble actual = ComplexDoubleMake(
          PhasorRotatorCos(&oscillator),
          PhasorRotatorSin(&oscillator));
      PhasorRotatorNext(&oscillator);
      ComplexDouble expected = ComplexDoubleMake(
          cos(radians_per_sample * i),
          sin(radians_per_sample * i));
      double error = ComplexDoubleAbs2(ComplexDoubleSub(actual, expected));
      if (error > max_error) { max_error = error; }
    }
  }

  max_error = sqrt(max_error);
  CHECK(max_error < 0.001);
}

void TestPeriod() {
  const double kSampleRateHz = 16000.0;
  PhasorRotator oscillator;
  int trial;

  for (trial = 0; trial < 5; ++trial) {
    const double frequency_hz = 0.05 + 0.3 * kSampleRateHz * RandUniform();
    const double expected_period = kSampleRateHz / frequency_hz;
    PhasorRotatorInit(&oscillator, frequency_hz, kSampleRateHz);
    double sum = 0.0;
    int count = 0;

    double prev_crossing = 0.0;
    float prev_value = 0.0f;
    int i;
    for (i = 0; i < 10000000; ++i) {  /* Run 10M steps. */
      float value = PhasorRotatorSin(&oscillator);
      PhasorRotatorNext(&oscillator);

      /* Positive-slope zero crossing. */
      if (prev_value < 0.0f && value > 0.0f) {
        /* Linearly interpolate the subsample zero crossing position. */
        const double crossing = i + value / ((double)prev_value - value);
        /* Take difference with the previous crossing to compute the period. */
        const double actual_period = crossing - prev_crossing;

        /* Within 0.2 samples of expected_period. */
        CHECK(fabs(actual_period - expected_period) < 0.2);

        sum += actual_period;
        ++count;
        prev_crossing = crossing;
      }
      prev_value = value;
    }

    const double average_period = sum / count;
    /* The average period should match accurately, within 5e-5 samples. */
    CHECK(fabs(average_period - expected_period) < 5e-5);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestBasic();
  TestPeriod();

  puts("PASS");
  return EXIT_SUCCESS;
}
