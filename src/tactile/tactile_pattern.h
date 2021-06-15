/* Copyright 2021 Google LLC
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
 * A library to synthesize a few simple tactile patterns. These patterns are the
 * haptic equivalent of sound effects, useful as feedback in UI. The patterns
 * are single-channel sequences of sine waves, chirps, and pauses.
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PATTERN_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PATTERN_H_

#include <stdint.h>
#include "dsp/phase32.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const char* kTactilePatternSilence;
extern const char* kTactilePatternConnect;
extern const char* kTactilePatternDisconnect;
extern const char* kTactilePatternConfirm;

typedef struct {
  /* Pattern string, encoding a sequence of segments. See .c for details. */
  const char* pattern;
  /* Sample rate in Hz. */
  float sample_rate_hz;
  /* Oscillator for synthesizing sine waves and chirps. */
  Oscillator tone;
  /* Another oscillator for fading in and out by modulating a raised cosine. */
  Oscillator fade;
  /* Used for synthesizing chirps. */
  float chirp_rate;
  /* Number of frames in a fade transition. */
  int fade_frames;
  /* Number of frames left in the current segment. */
  int segment_counter;
  /* Number of frames left in the current fade transition. */
  int fade_counter;
  /* If nonzero, start fading out when segment_counter == fade_start_index. */
  int fade_start_index;

  /* The channel that the pattern plays on, or -1 to play on all channels. */
  int active_channel;
  /* For calibration tones, the channel index for the second tone. */
  int second_channel;
} TactilePattern;

/* Initializes the synthesizer. */
void TactilePatternInit(TactilePattern* p, float sample_rate_hz);

/* Resets synthesizer to produce `pattern`. TactilePatternInit() should be
 * called first, and after that, TactilePatterStart() may be called at any time.
 */
void TactilePatternStart(TactilePattern* p, const char* pattern);

/* Resets synthesizer to produce calibration tones on channels `first_channel`
 * and `second_channel`. If `first_channel == second_channel`, a single tone is
 * played on that channel. Otherwise, the synthesizer plays: (1) a tone on
 * first_channel, (2) a pause, (3) a tone on second_channel.
 */
void TactilePatternStartCalibrationTones(TactilePattern* p, int first_channel,
                                         int second_channel);

/* Synthesizes `num_frames` of output to `num_channels` output channels in a
 * streaming manner. The `output` pointer should point to an array with space
 * for `num_frames * num_channels` samples. The pattern is generated as a
 * single-channel signal, which is then repeated to play equally over all output
 * channels. The function always synthesizes `num_frames` frames, padding the
 * output with silence once the pattern completes. Returns 1 if the pattern is
 * still playing, 0 if the pattern has completed.
 */
int /*bool*/ TactilePatternSynthesize(TactilePattern* p,
                                      int num_frames,
                                      int num_channels,
                                      float* output);

/* Returns 1 if the pattern is still actively playing, or 0 if completed. */
static int /*bool*/ TactilePatternIsActive(const TactilePattern* p) {
  return *p->pattern != '\0';
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PATTERN_H_ */
