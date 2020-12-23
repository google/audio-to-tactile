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

#include <math.h>

#include "dsp/math_constants.h"
#include "mux/pilot_tracker.h"

/* Pilot amplitude is computed as the average of |real part| + |imag part|
 * which is an overestimate, equal to (true amplitude) * 4/pi for a pure
 * frequency. Using this amplitude, phase error is then detected in units of
 * radians as the imaginary part of
 *
 *   sample * exp(-i2 * pi * pilot_phase) / amplitude
 *
 * So we scale the phase error by 4/pi to account for the amplitude
 * overestimation and by 1 / (2 * pi) to covert radians to cycles, a combined
 * factor of 2/pi^2. This factor is absorbed into the PLL loop filter.
 */
#define kPllScale (2 / (M_PI * M_PI))

/* Loop filter coefficients for the phase-locked loop, a basic proportional
 * plus integrator design.
 */
#define kPllBandwidth 0.05f /* Units of radians / sample. */
#define kPllIntegratorCoeff ((float)(kPllScale * kPllBandwidth * kPllBandwidth))
#define kPllProportionalCoeff ((float)(kPllScale * M_SQRT2 * kPllBandwidth))

void PilotTrackerCoeffsInit(PilotTrackerCoeffs* coeffs, float pilot_hz,
                            float sample_rate_hz) {
  /* Make a complex bandpass filter centered on pilot_hz, a 2-pole gamma filter
   * with the pole rotated in the complex plane to pilot_hz. It's like the
   * Goertzel filter [https://en.wikipedia.org/wiki/Goertzel_algorithm], but
   * both poles at the same complex place instead of a conjugate pair, and they
   * have magnitude less than one to stabilize the filter.
   */
  const float kBpfBandwidthHz = 16.0f;
  const float kPoleMag =
      exp((2 * M_PI / sample_rate_hz) * -kBpfBandwidthHz / 2);
  coeffs->pilot_bpf_pole.real =
      kPoleMag * cos((2 * M_PI / sample_rate_hz) * pilot_hz);
  coeffs->pilot_bpf_pole.imag =
      kPoleMag * sin((2 * M_PI / sample_rate_hz) * pilot_hz);

  /* Set amplitude smoother with a time constant of 20 samples. This assumes
   * pilot_hz is larger than sample_rate_hz / 20 so that the smoother averages
   * over at least a cycle.
   */
  const float kAmplitudeTimeConstant = 20.0f;  /* Units of samples. */
  coeffs->pilot_amplitude_smoother = 1 - exp(-1 / kAmplitudeTimeConstant);
}

void PilotTrackerInit(PilotTracker* tracker, const PilotTrackerCoeffs* coeffs) {
  tracker->pilot[0] = ComplexFloatMake(0.0f, 0.0f);
  tracker->pilot[1] = ComplexFloatMake(0.0f, 0.0f);
  const float pilot_amplitude_init =
      1 / (pow(1 - coeffs->pilot_bpf_pole.real, 2) +
           pow(coeffs->pilot_bpf_pole.imag, 2));
  tracker->pilot_amplitude[0] = pilot_amplitude_init;
  tracker->pilot_amplitude[1] = pilot_amplitude_init;
  tracker->pilot_frequency = 0.0f;
  tracker->pilot_phase = 0;
}

Phase32 PilotTrackerProcessOneSample(PilotTracker* tracker,
                                     const PilotTrackerCoeffs* coeffs,
                                     ComplexFloat sample) {
  /* Bandpass filter to extract the pilot. */
  tracker->pilot[0] = ComplexFloatAdd(
      sample, ComplexFloatMul(coeffs->pilot_bpf_pole, tracker->pilot[0]));
  tracker->pilot[1] = ComplexFloatAdd(
      tracker->pilot[0],
      ComplexFloatMul(coeffs->pilot_bpf_pole, tracker->pilot[1]));

  /* Estimate the pilot's amplitude from abs(sample.real) + abs(sample.imag),
   * smoothed with a second-order gamma filter.
   */
  tracker->pilot_amplitude[0] +=
      coeffs->pilot_amplitude_smoother *
      (fabs(tracker->pilot[1].real) + fabs(tracker->pilot[1].imag) -
       tracker->pilot_amplitude[0]);
  tracker->pilot_amplitude[1] +=
      coeffs->pilot_amplitude_smoother *
      (tracker->pilot_amplitude[0] - tracker->pilot_amplitude[1]);

  /* Phase-locked loop to track the pilot. */
  float phase_error =
      (tracker->pilot[1].imag * Phase32Cos(tracker->pilot_phase) -
       tracker->pilot[1].real * Phase32Sin(tracker->pilot_phase)) /
      tracker->pilot_amplitude[1];
  tracker->pilot_frequency += kPllIntegratorCoeff * phase_error;
  return tracker->pilot_phase += Phase32FromFloat(
      tracker->pilot_frequency + kPllProportionalCoeff * phase_error);
}
