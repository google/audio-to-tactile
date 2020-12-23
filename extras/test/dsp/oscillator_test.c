/* Copyright 2020 Google LLC
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

#include "dsp/oscillator.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp/logging.h"
#include "dsp/math_constants.h"

static double RandUniform() { return (double)rand() / RAND_MAX; }

void TestSineTable() {
  puts("TestSineTable");
  const int table_size = 1 << kOscillatorTableBits;
  CHECK(kOscillatorSinTable[0] == 0.0f);
  CHECK(kOscillatorSinTable[table_size / 4] == 1.0f);
  CHECK(kOscillatorSinTable[table_size / 2] == 0.0f);
  CHECK(kOscillatorSinTable[table_size * 3 / 4] == -1.0f);

  const double radians_per_sample = 2 * M_PI / table_size;
  int i;
  for (i = 0; i < table_size; ++i) {
    double expected = sin(radians_per_sample * i);
    CHECK(fabs(expected - kOscillatorSinTable[i]) <= 2e-7);
  }
}

void TestPhaseConversion() {
  puts("TestPhaseConversion");
  Oscillator oscillator;
  OscillatorSetPhase(&oscillator, 0.0f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.0f);

  /* Phase 0.25 should be exactly represented by 0.25 * 2^32. */
  OscillatorSetPhase(&oscillator, 0.25f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.25f);

  /* Test adding to phase. */
  OscillatorAddPhase(&oscillator, 0.125f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.375f);

  /* Check that phase wraps correctly to [0, 1). */
  OscillatorSetPhase(&oscillator, -0.25f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.75f);
  OscillatorSetPhase(&oscillator, -0.75f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.25f);
  OscillatorSetPhase(&oscillator, 1.0f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.0f);
  OscillatorSetPhase(&oscillator, -1.0f);
  CHECK(OscillatorGetPhase(&oscillator) == 0.0f);

  int i;
  for (i = 0; i < 30; ++i) {
    const double phase = RandUniform();
    OscillatorSetPhase(&oscillator, phase);
    CHECK(fabs(OscillatorGetPhase(&oscillator) - phase) <= 2e-7);
  }
}

void TestAddFrequency() {
  puts("TestAddFrequency");
  Oscillator oscillator;
  OscillatorSetFrequency(&oscillator, 0.375f);
  OscillatorAddFrequency(&oscillator, 0.25f);
  CHECK(OscillatorGetFrequency(&oscillator) == 0.625f);

  OscillatorAddFrequency(&oscillator, 0.75f);
  CHECK(OscillatorGetFrequency(&oscillator) == 0.375f);

  OscillatorAddFrequency(&oscillator, -0.125f);
  CHECK(OscillatorGetFrequency(&oscillator) == 0.25f);
}

void TestOscillatorSamples(float frequency_cycles_per_sample) {
  printf("TestOscillatorSamples(%g)\n", frequency_cycles_per_sample);
  Oscillator oscillator;
  OscillatorSetFrequency(&oscillator, frequency_cycles_per_sample);
  OscillatorSetPhase(&oscillator, 0.0f);

  const double radians_per_sample = 2 * M_PI * frequency_cycles_per_sample;
  int i;
  for (i = 0; i < 1000; ++i) {
    const double angle = radians_per_sample * i;
    /* The largest quantization error in phase due to the table is
     * delta = 2^-kOscillatorTableBits / 2 cycles, and for any phase x,
     * |sin(2 pi (x + delta)) - sin(2 pi x)| <= 2 pi |delta|, so errors
     * should not be much larger than pi / 2^kOscillatorTableBits = 0.00307.
     */
    CHECK(fabs(OscillatorSin(&oscillator) - sin(angle)) <= 0.0035);
    CHECK(fabs(OscillatorCos(&oscillator) - cos(angle)) <= 0.0035);
    CHECK(fabs(OscillatorComplexExp(&oscillator).real - cos(angle)) <= 0.0035);
    CHECK(fabs(OscillatorComplexExp(&oscillator).imag - sin(angle)) <= 0.0035);
    OscillatorNext(&oscillator);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestSineTable();
  TestPhaseConversion();
  TestAddFrequency();
  TestOscillatorSamples(0.0001f);
  TestOscillatorSamples(0.001f);
  TestOscillatorSamples(0.01f);
  TestOscillatorSamples(0.1f);
  TestOscillatorSamples(0.5f);
  TestOscillatorSamples(-0.1f);

  puts("PASS");
  return EXIT_SUCCESS;
}
