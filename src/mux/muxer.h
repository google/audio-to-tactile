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
 * A frequency-division multiplexing (FDM) encoder, combining
 * kMuxChannels channels of tactile signals into a single channel.
 */

#ifndef AUDIO_TO_TACTILE_SRC_MUX_MUXER_H_
#define AUDIO_TO_TACTILE_SRC_MUX_MUXER_H_

#include "mux/mux_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Muxer; /* Forward declaration. See muxer.c for definition. */
typedef struct Muxer Muxer;

/* Makes a Muxer. Returns NULL on failure. */
Muxer* MuxerMake();

/* Frees a Muxer. */
void MuxerFree(Muxer* muxer);

/* Resets Muxer to initial state. */
void MuxerReset(Muxer* muxer);

/* Gets the number of output samples that will be written for a given number of
 * input frames by the next call to MuxerProcessSamples. The output size never
 * exceeds `kMuxRateFactor * num_input_frames`.
 */
int MuxerNextOutputSize(Muxer* muxer, int num_input_frames);

/* Processes samples in a streaming manner.
 *
 * `tactile_input` is an array with `kMuxChannels * num_frames` elements
 * in which `tactile_input[i * kMuxChannels + c] is frame i, channel c of
 * the tactile input, and has sample rate kMuxTactileRate.
 *
 * `muxed_output` is an array where muxed output is written with sample rate
 * kMuxMuxedRate. Use `MuxerNextOutputSize(muxer, num_frames)` to determine the
 * number of samples that will be written. Alternatively, make muxed_output an
 * array with space for `kMuxRateFactor * num_frames` elements and get the
 * number of samples written from the return value of this function.
 */
int MuxerProcessSamples(Muxer* muxer, const float* tactile_input,
                        int num_frames, float* muxed_output);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_MUX_MUXER_H_ */
