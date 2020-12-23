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
 * Numerically controlled oscillator.
 *
 * Implements a numerically controlled oscillator, efficient method for sampling
 * approximate values of a sine wave sin(2 pi frequency n), n = 0, 1, ...
 * [http://en.wikipedia.org/wiki/Numerically_controlled_oscillator]. Internally,
 * it represents phases as fixed-point values and uses a hardcoded look up table
 * of sine values.
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_OSCILLATOR_H_
#define AUDIO_TO_TACTILE_SRC_DSP_OSCILLATOR_H_

#include <stdint.h>

#include "dsp/complex.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The size of kOscillatorSinTable is 2^kOscillatorTableBits. A value of 10
 * means a table of 1024 entries in 4 KB of memory, a good tradeoff between
 * accuracy and memory cost.
 */
#define kOscillatorTableBits 10

extern const float kOscillatorSinTable[1 << kOscillatorTableBits /* = 1024 */];

typedef struct {
  /* Current oscillator phase + kRoundOffset represented in units of cycles as a
   * Q32 fixed point value:
   *
   *   phase_plus_round_offset = p / 2^32 + kRoundOffset,
   *
   * where 0 <= p < 1 represents the phase as a fraction of a cycle. Offsetting
   * by kRoundOffset = 2^TableBits / 2 saves an addition in OscillatorSin().
   */
  uint32_t phase_plus_round_offset;
  /* The step added to phase_ every sample; the oscillator frequency. */
  uint32_t delta_phase;
} Oscillator;

/* Sets the oscillator's frequency in units of cycles per sample. Typically the
 * frequency is between -0.5 and 0.5. Frequencies of greater magnitude alias
 * into this range. For a frequency in Hz, convert to cycles per sample as
 *
 *   OscillatorSetFrequency(oscillator, frequency_hz / sample_rate_hz);
 */
void OscillatorSetFrequency(Oscillator* oscillator,
                            float frequency_cycles_per_sample);
/* Gets the frequency in units of cycles per sample. */
float OscillatorGetFrequency(const Oscillator* oscillator);
/* Adjusts the frequency by adding `frequency_cycles_per_sample`. */
void OscillatorAddFrequency(Oscillator* oscillator,
                            float frequency_cycles_per_sample);

/* Sets the oscillator's current phase in units of cycles. The phase is a value
 * in [0, 1). Values outside this range are wrapped.
 */
void OscillatorSetPhase(Oscillator* oscillator, float phase_in_cycles);
/* Gets the current phase in units of cycles as a value in [0, 1). */
float OscillatorGetPhase(const Oscillator* oscillator);
/* Adjusts the phase by adding `phase_in_cycles`. */
void OscillatorAddPhase(Oscillator* oscillator, float phase_in_cycles);

/* NOTE: Functions below are marked `static` [the C analogy for `inline`] so
 * that ideally they get inline expanded.
 */

/* Sine of the current phase. */
static float OscillatorSin(const Oscillator* oscillator) {
  /* Convert from 32 bits down to TableBits to get a table index. */
  int i = oscillator->phase_plus_round_offset >> (32 - kOscillatorTableBits);
  return kOscillatorSinTable[i];
}

/* Cosine of the current phase. */
static float OscillatorCos(const Oscillator* oscillator) {
  /* Look up cosine using the identity cos(x) = sin(x + pi/2). Add 2^30 to
   * rotate phase by a quarter period, then convert to table index.
   */
  int i = (oscillator->phase_plus_round_offset + (UINT32_C(1) << 30))
      >> (32 - kOscillatorTableBits);
  return kOscillatorSinTable[i];
}

/* Complex exponential exp(i * 2 * pi * GetPhase()) of the current phase. */
static ComplexFloat OscillatorComplexExp(const Oscillator* oscillator) {
  return ComplexFloatMake(OscillatorCos(oscillator),
                          OscillatorSin(oscillator));
}

/* Advances the phase by one sample according to the oscillator frequency. */
static void OscillatorNext(Oscillator* oscillator) {
  /* This addition rolls over once each cycle of oscillation. */
  oscillator->phase_plus_round_offset += oscillator->delta_phase;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_DSP_OSCILLATOR_H_ */
