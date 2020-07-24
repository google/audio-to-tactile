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
 * Envelope onset asynchrony (EOA) tactile feature of Yuan et al.
 *
 * This library implements the envelope onset asynchrony feature described in
 *
 * Yuan, Hanfeng, Charlotte M. Reed, and Nathaniel I. Durlach. "Tactual display
 * of consonant voicing as a supplement to lipreading." The Journal of the
 * Acoustical society of America 118.2 (2005): 1003-1015.
 * https://dspace.mit.edu/bitstream/handle/1721.1/87906/54935608-MIT.pdf?sequence=2
 *
 * The method takes input audio and produces 2-channel tactile output, based on
 * filtering and extracting amplitude envelopes.
 */

#ifndef AUDIO_TACTILE_REFERENCES_YUAN2005_YUAN2005_H_
#define AUDIO_TACTILE_REFERENCES_YUAN2005_YUAN2005_H_

#include "audio/dsp/portable/biquad_filter.h"
#include "audio/dsp/portable/phasor_rotator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  /* Threshold under which signal is set to zero. */
  float denoising_threshold;
  /* Offset added before modulating, the absolute sensory threshold. */
  float sensory_offset;
} Yuan2005ChannelParams;

typedef struct {
  /* Sample rate in Hz. */
  float sample_rate_hz;
  /* Parameters for the lowpassed channel (channel 0). */
  Yuan2005ChannelParams low_channel;
  /* Parameters for the highpassed channel (channel 1). */
  Yuan2005ChannelParams high_channel;
} Yuan2005Params;

/* Sets all parameters to default values. */
void Yuan2005SetDefaultParams(Yuan2005Params* params);

typedef struct {
  Yuan2005ChannelParams channel_params;
  BiquadFilterCoeffs band_biquad_coeffs;
  BiquadFilterCoeffs envelope_biquad_coeffs;
  PhasorRotator oscillator;

  BiquadFilterState band_biquad_state;
  BiquadFilterState envelope_biquad_state;
} Yuan2005ChannelState;

typedef struct {
  Yuan2005ChannelState low_channel;
  Yuan2005ChannelState high_channel;
} Yuan2005State;

/* Initializes state. Returns 1 on success, 0 on failure. */
int Yuan2005Init(Yuan2005State* state, const Yuan2005Params* params);

/* Processes audio in a streaming manner. The `input` pointer must point to a
 * contiguous array of `num_samples` samples. The number of output samples
 * written is `2 * num_samples`, for the two output channels written in
 * interleaved order.
 */
void Yuan2005ProcessSamples(Yuan2005State* state,
                            const float* input,
                            int num_samples,
                            float* output);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_REFERENCES_YUAN2005_YUAN2005_H_ */

