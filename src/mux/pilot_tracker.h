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
 *
 *
 * Phase tracking of a pilot tone, useful for time synchronization.
 *
 * This library extracts a complex-valued pilot tone c * exp(i2 * pi * f * t)
 * and uses a basic phase-locked loop to track its phase. The tracker
 * representation is split up into two types, `PilotTrackerCoeffs` and
 * `PilotTracker` for memory efficiency. Multiple trackers can share the same
 * coefficients.
 *
 * Outline:
 * 1. The pilot signal is extracted using a complex bandpass filter.
 * 2. Pilot amplitude is estimated by smoothing |real part| + |imag part|.
 * 3. Phase error is detected as
 *
 *       phase error = Im{ sample * exp(-i2 * pi * pilot_phase) } / amplitude
 *
 * 4. Phase error is filtered with a loop filter.
 */

#ifndef AUDIO_TO_TACTILE_SRC_MUX_PILOT_PHASE_TRACKER_H_
#define AUDIO_TO_TACTILE_SRC_MUX_PILOT_PHASE_TRACKER_H_

#include "dsp/complex.h"
#include "dsp/phase32.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  /* For extracting the pilot, the second-order gamma bandpass filter pole. */
  ComplexFloat pilot_bpf_pole;
  /* Smoother coefficient for estimating the pilot amplitude. */
  float pilot_amplitude_smoother;
} PilotTrackerCoeffs;

/* Initializes coefficients for pilot tracking. */
void PilotTrackerCoeffsInit(PilotTrackerCoeffs* coeffs, float pilot_hz,
                            float sample_rate_hz);

typedef struct {
  /* pilot[1] is the BPF-filtered pilot signal (pilot[0] is an intermediate). */
  ComplexFloat pilot[2];
  /* pilot_amplitude[1] is the estimated pilot amplitude. */
  float pilot_amplitude[2];
  /* The current phase of the pilot, tracked by a phase-locked loop. */
  Phase32 pilot_phase;
  /* Pilot frequency estimate in units of cycles per sample. */
  float pilot_frequency;
} PilotTracker;

/* Initializes PilotTracker based on the given coefficients. */
void PilotTrackerInit(PilotTracker* tracker, const PilotTrackerCoeffs* coeffs);

/* Processes one complex-valued sample. Returns the pilot's current phase. */
Phase32 PilotTrackerProcessOneSample(PilotTracker* tracker,
                                     const PilotTrackerCoeffs* coeffs,
                                     ComplexFloat sample);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_MUX_PILOT_PHASE_TRACKER_H_ */
