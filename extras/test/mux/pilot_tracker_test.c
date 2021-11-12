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

#include "src/mux/pilot_tracker.h"

#include <math.h>
#include <stdlib.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

#define kSampleRateHz 32000.0f

static double RandUniform(void) { return (double)rand() / RAND_MAX; }

/* Test tracking of a steady tone plus a little noise:
 *   amplitude * exp(2i * pi * frequency_hz * t + initial_phase) + noise.
 */
static void TestSteadyTone(float amplitude, float frequency_hz,
                           float initial_phase) {
  printf("TestSteadyTone(%g, %g, %g)\n",
         amplitude, frequency_hz, initial_phase);
  PilotTrackerCoeffs tracker_coeffs;
  PilotTrackerCoeffsInit(&tracker_coeffs, frequency_hz, kSampleRateHz);
  PilotTracker tracker;
  PilotTrackerInit(&tracker, &tracker_coeffs);

  const float cycles_per_sample = frequency_hz / kSampleRateHz;
  int n;
  for (n = 0; n < 500; ++n) {
    /* Get the next complex input sample. */
    const float expected_phase = initial_phase + cycles_per_sample * n;
    ComplexFloat sample = ComplexFloatMake(
        cos(2 * M_PI * expected_phase) + 0.2 * (RandUniform() - 0.5),
        sin(2 * M_PI * expected_phase) + 0.2 * (RandUniform() - 0.5));
    sample = ComplexFloatMulReal(sample, amplitude);

    /* Run tracker on the sample. */
    Phase32 pilot_phase = PilotTrackerProcessOneSample(
        &tracker, &tracker_coeffs, sample);

    if (n >= 120) {
      float error = expected_phase - Phase32ToFloat(pilot_phase);
      error -= floor(error + 0.5); /* Wrap phase error to [-0.5, 0.5). */
      CHECK(fabs(error) < 0.05f); /* Within 1/20th of a cycle (18 degrees). */

      CHECK(fabs(cycles_per_sample - tracker.pilot_frequency) < 0.001f);
    }
  }
}

/* Test tracking of a warbling tone that varies in frequency. */
static void TestWarblingTone(void) {
  puts("TestWarblingTone");
  const float kMinFrequency = 295.0f;
  const float kMaxFrequency = 305.0f;
  const float kWarbleHz = 60.0f;

  const float center_hz = 0.5f * (kMinFrequency + kMaxFrequency);
  const float alpha = (kMaxFrequency - kMinFrequency) / (4 * M_PI * kWarbleHz);

  PilotTrackerCoeffs tracker_coeffs;
  PilotTrackerCoeffsInit(&tracker_coeffs, center_hz, kSampleRateHz);
  PilotTracker tracker;
  PilotTrackerInit(&tracker, &tracker_coeffs);

  int n;
  for (n = 0; n < 1000; ++n) {
    /* Get the next complex input sample. */
    const float t = n / kSampleRateHz;
    const float expected_phase =
        center_hz * t - alpha * cos(2 * M_PI * kWarbleHz * t);
    ComplexFloat sample = ComplexFloatMake(
        cos(2 * M_PI * expected_phase) + 0.2 * (RandUniform() - 0.5),
        sin(2 * M_PI * expected_phase) + 0.2 * (RandUniform() - 0.5));

    /* Run tracker on the sample. */
    Phase32 pilot_phase = PilotTrackerProcessOneSample(
        &tracker, &tracker_coeffs, sample);

    if (n >= 120) {
      float error = expected_phase - Phase32ToFloat(pilot_phase);
      error -= floor(error + 0.5); /* Wrap phase error to [-0.5, 0.5). */
      CHECK(fabs(error) < 0.05f); /* Within 1/20th of a cycle (18 degrees). */

      CHECK(fabs(center_hz / kSampleRateHz - tracker.pilot_frequency) < 0.001f);
    }
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestSteadyTone(1.0f, 500.0f, 0.0f);
  TestSteadyTone(1.0f, 500.0f, 0.6f);
  TestSteadyTone(1.0f, 500.0f, 0.7f);

  TestSteadyTone(0.1f, 500.0f, 0.0f);
  TestSteadyTone(0.6f, 500.0f, 0.0f);

  TestSteadyTone(0.3f, 1000.0f, 0.2f);
  TestSteadyTone(0.4f, -1200.0f, 0.9f);

  TestWarblingTone();

  puts("PASS");
  return EXIT_SUCCESS;
}
