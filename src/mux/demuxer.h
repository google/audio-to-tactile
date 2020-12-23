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

#ifndef AUDIO_TO_TACTILE_SRC_MUX_DEMUXER_H_
#define AUDIO_TO_TACTILE_SRC_MUX_DEMUXER_H_

#include "dsp/biquad_filter.h"
#include "dsp/complex.h"
#include "dsp/phase32.h"
#include "mux/mux_common.h"
#include "mux/pilot_tracker.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  Oscillator down_converter;
  PilotTracker pilot_tracker;
  BiquadFilterState weaver_lpf_state_real;
  BiquadFilterState weaver_lpf_state_imag;
  Oscillator up_converter;
} DemuxerChannel;

typedef struct {
  DemuxerChannel channels[kMuxChannels];
  PilotTrackerCoeffs pilot_tracker_coeffs;
  BiquadFilterCoeffs weaver_lpf_coeffs;
} Demuxer;

/* Initializes a Demuxer. */
void DemuxerInit(Demuxer* demuxer);

/* Processes samples in a streaming manner.
 *
 * `muxed_input` is an array of `num_samples` elements of received muxed samples
 * at kMuxMuxedRate sample rate.
 * NOTE: num_samples must be a multiple of kMuxRateFactor.
 *
 * `tactile_output` is an array with space for `kMuxChannels *
 * num_samples / kMuxRateFactor` elements, where demuxed output is
 * written. `tactile_output[i * kMuxChannels + c]` is frame i, channel c
 * of the tactile output, and has sample rate kMuxTactileRate.
 */
void DemuxerProcessSamples(Demuxer* demuxer, const float* muxed_input,
                           int num_samples, float* tactile_output);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_MUX_DEMUXER_H_ */
