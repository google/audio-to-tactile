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
 * Fire-and-forget playback of tactile signals.
 *
 * A tactile playback object is created with
 *
 *   TactilePlayer* player = TactilePlayerMake(num_channels, sample_rate_hz);
 *
 * If using TactilePlayer with the portaudio library, TactilePlayerFillBuffer is
 * called in the portaudio stream callback as
 *
 *   static int PortaudioStreamCallback(const void* input_buffer,
 *                                      void* output_buffer,
 *                                      uint64_t frames_per_buffer, ...) {
 *     TactilePlayerFillBuffer(player, frames_per_buffer, output_buffer);
 *     return paContinue;
 *   }
 *
 * TactilePlayerPlay can then be used from the main thread for simple
 * fire-and-forget playback of tactile signals:
 *
 *   TactilePlayerPlay(player, samples, num_frames);
 *
 * The player takes ownership of `samples`. If a tactile signal is already
 * playing, it is interrupted.
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_REFERENCES_TAPS_TACTILE_PLAYER_H_
#define AUDIO_TO_TACTILE_EXTRAS_REFERENCES_TAPS_TACTILE_PLAYER_H_

#ifdef __cplusplus
extern "C" {
#endif

struct TactilePlayer;
typedef struct TactilePlayer TactilePlayer;

/* Allocates and returns a TactilePlayer, or NULL on failure. The caller should
 * free it when done with TactilePlayerFree.
 */
TactilePlayer* TactilePlayerMake(int num_channels, float sample_rate_hz);

/* Frees a TactilePlayer. */
void TactilePlayerFree(TactilePlayer* player);

/* Plays a tactile signal in a fire-and-forget way. This function is thread
 * safe. `samples` should point to an array of `num_frames * num_channels`
 * samples in interleaved order. The `samples` array must have been allocated
 * with malloc, calloc, or realloc. The player takes ownership of `samples`.
 *
 * If a tactile signal is already playing, it is interrupted. To avoid clicks,
 * 5ms following the current read position is linearly faded out and added
 * to `samples`.
 *
 * NOTE: This function calls free(), so it shouldn't be called from the audio
 * thread.
 */
void TactilePlayerPlay(TactilePlayer* player, float* samples, int num_frames);

/* Returns 1 if playback is active, i.e. read position < end position. */
int TactilePlayerIsActive(TactilePlayer* player);

/* Fills buffer `output` with `num_frames` frames. This function is thread
 * safe. Frames remaining after playback of the current tactile signal are zero
 * filled. The number of playback frames written is returned, i.e. a return
 * value less than num_frames means playback has ended.
 */
int TactilePlayerFillBuffer(TactilePlayer* player, int num_frames,
                            float* output);

/* For tactor activity or volume meter displays, this function computes the
 * current root-mean-squared (RMS) value of the signal in each channel. The RMS
 * is computed over a window of width window_duration_s centered around the
 * current read position. RMS values are written to `rms`, which should be an
 * array of size `num_channels`.
 */
void TactilePlayerGetRms(TactilePlayer* player, float window_duration_s,
                         float* rms);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_EXTRAS_REFERENCES_TAPS_TACTILE_PLAYER_H_ */
