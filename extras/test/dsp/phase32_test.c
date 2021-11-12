/* Copyright 2020-2021 Google LLC
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

#include "src/dsp/phase32.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

static double RandUniform(void) { return (double)rand() / RAND_MAX; }

static void TestSineTable(void) {
  puts("TestSineTable");
  const int table_size = 1 << kPhase32TableBits;
  CHECK(kPhase32SinTable[0] == 0.0f);
  CHECK(kPhase32SinTable[table_size / 4] == 1.0f);
  CHECK(kPhase32SinTable[table_size / 2] == 0.0f);
  CHECK(kPhase32SinTable[table_size * 3 / 4] == -1.0f);

  const double radians_per_sample = 2 * M_PI / table_size;
  int i;
  for (i = 0; i < table_size; ++i) {
    double expected = sin(radians_per_sample * i);
    CHECK(fabs(expected - kPhase32SinTable[i]) <= 2e-7);
  }
}

static void TestPhaseConversion(void) {
  puts("TestPhaseConversion");
  Phase32 p = Phase32FromFloat(0.0f);
  CHECK(p == 0);
  CHECK(Phase32ToFloat(p) == 0.0f);

  /* Phase 0.25 should be exactly represented. */
  p = Phase32FromFloat(0.25f);
  CHECK(Phase32ToFloat(p) == 0.25f);

  /* Test adding to phase. */
  p += Phase32FromFloat(0.125f);
  CHECK(Phase32ToFloat(p) == 0.375f);

  /* Check that phase wraps correctly to [0, 1). */
  p = Phase32FromFloat(-0.25f);
  CHECK(Phase32ToFloat(p) == 0.75f);
  p = Phase32FromFloat(-0.75f);
  CHECK(Phase32ToFloat(p) == 0.25f);
  p = Phase32FromFloat(1.0f);
  CHECK(Phase32ToFloat(p) == 0.0f);
  p = Phase32FromFloat(-1.0f);
  CHECK(Phase32ToFloat(p) == 0.0f);

  int i;
  for (i = 0; i < 30; ++i) {
    const double phase = RandUniform();
    p = Phase32FromFloat(phase);
    CHECK(fabs(Phase32ToFloat(p) - phase) <= 2e-7);
  }
}

static void TestOscillatorSamples(float frequency_cycles_per_sample) {
  printf("TestOscillatorSamples(%g)\n", frequency_cycles_per_sample);
  Oscillator oscillator;
  OscillatorInit(&oscillator, frequency_cycles_per_sample);

  const double radians_per_sample = 2 * M_PI * frequency_cycles_per_sample;
  int i;
  for (i = 0; i < 1000; ++i) {
    const double angle = radians_per_sample * i;
    /* The largest quantization error in phase due to the table is
     * delta = 2^-kPhase32TableBits / 2 cycles, and for any phase x,
     * |sin(2 pi (x + delta)) - sin(2 pi x)| <= 2 pi |delta|, so errors
     * should not be much larger than pi / 2^kPhase32TableBits = 0.00307.
     */
    const double kTol = 0.0035;
    CHECK(fabs(Phase32Sin(oscillator.phase) - sin(angle)) <= kTol);
    CHECK(fabs(Phase32Cos(oscillator.phase) - cos(angle)) <= kTol);
    CHECK(fabs(Phase32ComplexExp(oscillator.phase).real - cos(angle)) <= kTol);
    CHECK(fabs(Phase32ComplexExp(oscillator.phase).imag - sin(angle)) <= kTol);
    OscillatorNext(&oscillator);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestSineTable();
  TestPhaseConversion();
  TestOscillatorSamples(0.0001f);
  TestOscillatorSamples(0.001f);
  TestOscillatorSamples(0.01f);
  TestOscillatorSamples(0.1f);
  TestOscillatorSamples(0.5f);
  TestOscillatorSamples(-0.1f);

  puts("PASS");
  return EXIT_SUCCESS;
}
