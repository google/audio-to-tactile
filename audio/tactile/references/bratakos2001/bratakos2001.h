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
 *
 *
 * Single-band envelope tactile feature of Bratakos et al.
 *
 * This library implements the single-band envelope cue described in
 *
 * Bratakos, Maroula S., et al. "A single-band envelope cue as a supplement to
 * speechreading of segmentals: A comparison of auditory versus tactual
 * presentation." Ear and hearing 22.3 (2001): 225-235.
 *
 * The method takes input audio and produces one channel of tactile output,
 * based on filtering and extracting an amplitude envelopes.
 */

#ifndef AUDIO_TACTILE_REFERENCES_BRATAKOS2001_BRATAKOS2001_H_
#define AUDIO_TACTILE_REFERENCES_BRATAKOS2001_BRATAKOS2001_H_

#include "audio/dsp/portable/biquad_filter.h"
#include "audio/dsp/portable/phasor_rotator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  BiquadFilterCoeffs bpf_biquad_coeffs[2];
  BiquadFilterCoeffs envelope_biquad_coeffs;
  PhasorRotator oscillator;

  BiquadFilterState bpf_biquad_state[2];
  BiquadFilterState envelope_biquad_state;
} Bratakos2001State;

/* Initializes state. Returns 1 on success, 0 on failure. */
int Bratakos2001Init(Bratakos2001State* state, float sample_rate_hz);

/* Processes audio in a streaming manner. The `input` pointer should point to a
 * contiguous array of `num_samples` samples. The number of output samples
 * written is `2 * num_samples`, for the two output channels written in
 * interleaved order.
 */
void Bratakos2001ProcessSamples(Bratakos2001State* state,
                                const float* input,
                                int num_samples,
                                float* output);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_REFERENCES_BRATAKOS2001_BRATAKOS2001_H_ */

