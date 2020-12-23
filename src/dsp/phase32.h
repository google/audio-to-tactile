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
 * 32-bit phase representation and numerically controlled oscillator.
 *
 * `Phase32` represents phase or angle as a uint32_t. There are 2^32 possible
 * phases so that it may be allowed to wrap modulo 2^32 on overflow.
 * Phase32Sin() and Phase32Cos() efficiently get the approximate sine or cosine
 * at a given Phase32 using a 1024-entry lookup table.
 *
 * `Oscillator` implements a numerically controlled oscillator, an efficient
 * method for sampling approximate values of a sine wave: sin(2*pi*frequency*n)
 * [http://en.wikipedia.org/wiki/Numerically_controlled_oscillator]. The phase
 * and frequency may be manipulated during wave synthesis for time-varying phase
 * or frequency.
 *
 * NOTE: Functions below are marked `static` [the C analogy for `inline`] so
 * that ideally they get inline expanded.
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_PHASE32_H_
#define AUDIO_TO_TACTILE_SRC_DSP_PHASE32_H_

#include <math.h>
#include <stdint.h>

#include "dsp/complex.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The size of kPhase32SinTable is 2^kPhase32TableBits. A value of 10 means a
 * table of 1024 entries in 4 KB of memory, a good tradeoff between accuracy and
 * memory cost.
 */
#define kPhase32TableBits 10

#define kPhase32PhasesPerCycle 4294967296.0f  /* = 2^32. */

extern const float kPhase32SinTable[1 << kPhase32TableBits];

/* Phase represented as a Q32 fixed point value:
 *
 *   p = phase32 / 2^32,
 *
 * where 0 <= p < 1 represents the phase as a fraction of a cycle.
 */
typedef uint32_t Phase32;

/* Get Phase32 from float, where phase_float is in units of cycles. Typically
 * 0 <= phase_float < 1. Values outside this range are wrapped.
 */
static Phase32 Phase32FromFloat(float phase_float) {
  phase_float -= floor(phase_float);  /* Wrap to [0, 1) to prevent overflow. */
  return (uint32_t)(phase_float * kPhase32PhasesPerCycle);
}

/* Convert Phase32 to float in units of cycles, a value in [0, 1). */
static float Phase32ToFloat(Phase32 phase32) {
  return phase32 / kPhase32PhasesPerCycle;
}

/* Sine of phase. */
static float Phase32Sin(Phase32 phase) {
  /* Convert from 32 bits down to TableBits to get a table index. */
  return kPhase32SinTable[
      (phase
       + (UINT32_C(1) << (31 - kPhase32TableBits)) /* Rounding offset. */
      ) >> (32 - kPhase32TableBits)];
}

/* Cosine of phase. */
static float Phase32Cos(Phase32 phase) {
  /* Look up cosine using the identity cos(x) = sin(x + pi/2). Add 2^30 to
   * rotate phase by a quarter cycle, then convert to table index.
   */
  return kPhase32SinTable[
      (phase
       + ((UINT32_C(1) << (31 - kPhase32TableBits)) /* Rounding offset. */
          + (UINT32_C(1) << 30))                    /* Quarter cycle. */
      ) >> (32 - kPhase32TableBits)];
}

/* Complex exponential exp(i * 2 * pi * Phase32ToFloat(phase)) of the phase. */
static ComplexFloat Phase32ComplexExp(Phase32 phase) {
  ComplexFloat result;
  result.real = Phase32Cos(phase);
  result.imag = Phase32Sin(phase);
  return result;
}

typedef struct {
  /* The oscillator's current phase.
   * Set the phase with:
   *   oscillator.phase = Phase32FromFloat(x);  // x in units of cycles.
   * Get the phase with:
   *   float x = Phase32ToFloat(oscillator.phase);
   */
  Phase32 phase;
  /* This value is added to phase_ every sample; the oscillator frequency in
   * Q32 with units of cycles per sample.
   */
  Phase32 frequency;
} Oscillator;

/* Initializes the oscillator phase to 0 and the frequency in units of cycles
 * per sample. Typically the frequency is between -0.5 and 0.5. Frequencies of
 * greater magnitude alias into this range. For a frequency in Hz, convert to
 * cycles per sample as
 *
 *   OscillatorInit(oscillator, frequency_hz / sample_rate_hz);
 */
void OscillatorInit(Oscillator* oscillator, float frequency_cycles_per_sample);

/* Advances the phase by one sample according to the oscillator frequency. */
static void OscillatorNext(Oscillator* oscillator) {
  oscillator->phase += oscillator->frequency;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_DSP_PHASE32_H_ */
