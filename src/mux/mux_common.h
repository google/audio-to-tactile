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
 * Constants and definitions common to both the muxer and demuxer.
 */

#ifndef AUDIO_TO_TACTILE_SRC_MUX_MUX_COMMON_H_
#define AUDIO_TO_TACTILE_SRC_MUX_MUX_COMMON_H_

#include "dsp/biquad_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sample rate in Hz of the muxed signal. */
#define kMuxMuxedRate 31250.0f
/* Sample rate factor between muxed signal and constituent tactile signals. */
#define kMuxRateFactor 16
/* Sample rate in Hz of the tactile signals, equal to 31250/16 = 1953.125 Hz. */
#define kMuxTactileRate (kMuxMuxedRate / kMuxRateFactor)
/* Number of tactile channels. */
#define kMuxChannels 12
/* Range of tactile frequencies that the FDM scheme preserves. */
#define kMuxTactileMinHz 10.0f
#define kMuxTactileMaxHz 500.0f
/* Pilot frequency in Hz at baseband. */
#define kMuxPilotHzAtBaseband (-110.0f)

/* Constants for Weaver modulation: the band midpoint frequency and lowpass
 * filtering cutoff frequency.
 */
#define kMuxMidpointHz (0.5f * (kMuxTactileMaxHz + kMuxTactileMinHz))
#define kMuxWeaverLpfCutoffHz (0.5f * (kMuxTactileMaxHz - kMuxTactileMinHz))

/* Carrier frequency in Hz for channel c. */
static float MuxCarrierFrequency(int c) { return 1000.0f * (1 + c); }

/* Designs the demuxer's Weaver lowpass filter, a single biquad. */
void DemuxerDesignWeaverLpf(BiquadFilterCoeffs* coeffs);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_MUX_MUX_COMMON_H_ */
