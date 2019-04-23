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
 * Tactophone() function that starts portaudio and ncurses and runs the game.
 */

#ifndef AUDIO_TACTILE_PHONEME_CODE_TACTOPHONE_H_
#define AUDIO_TACTILE_PHONEME_CODE_TACTOPHONE_H_

#ifdef __cplusplus
extern "C" {
#endif

struct TactophoneState;

struct TactophoneParams {
  /* Path of the Tactophone lessons file [described in tactophone_lesson.h]. */
  const char* lessons_file;
  /* Path for writing log file. If it already exists, file is appended. */
  const char* log_file;
  /* Portaudio output device for playing tactile signals. */
  int output_device;
  /* Permutation for mapping Purdue's channel order to the output device, an
   * array of size 24 of 0-based indices. Channels are mapped like
   *   output_frame[c] = purdue_frame[channel_permutation[c]].
   */
  const int* channel_permutation;
  /* Initial state to start Tactophone in. */
  const struct TactophoneState* initial_state;
};
typedef struct TactophoneParams TactophoneParams;

/* Runs the Tactophone training game. Returns when game is exited. */
void Tactophone(const TactophoneParams* params);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_PHONEME_CODE_TACTOPHONE_H_ */
