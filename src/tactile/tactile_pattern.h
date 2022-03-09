/* Copyright 2021-2022 Google LLC
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
 * are multichannel sequences of sine waves, chirps, and pauses.
 *
 * The synthesizer takes a byte string as input, encoding the pattern to play.
 * Taking inspiration from music tracker formats, like Amiga MOD
 * https://en.wikipedia.org/wiki/MOD_(file_format), the byte string is a
 * sequence of basic ops (e.g., set a channel's waveform). These ops have a
 * compact representation and are simple to implement, yet can be composed in
 * many ways to produce a variety of patterns. Additionally, it is possible to
 * add new features to this format by defining new ops and waveforms.
 *
 * To avoid potential artifacts, the synthesizer takes care to smoothly fade
 * waveforms in and out as needed. Fading is done over a 20 ms interval using a
 * Hann window.
 *
 * Example, playing a simple format pattern:
 *   const char* simple_pattern = "555-C-8";
 *   TactilePattern p;
 *   TactilePatternInit(&p, kSampleRateHz, kNumChannels);
 *   TactilePatternStart(&p, simple_pattern);
 *
 *   float samples[kNumFrames * kNumChannels];
 *   while (TactilePatternSynthesize(&p, kNumFrames, samples)) {
 *     // ...
 *   }
 *
 * Example, playing an extended format pattern:
 *   const uint8_t ex_pattern[] = {
 *     kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformTone0,
 *     kTactilePatternOpSetGain + 1, 0xff, // Gain 1.0.
 *     TACTILE_PATTERN_OP_PLAY(100),       // Play for 100 ms.
 *     kTactilePatternOpMove, 0x01,        // Move from channel 0 to 1.
 *     TACTILE_PATTERN_OP_PLAY(100),       // Play for 100 ms.
 *     kTactilePatternOpMove, 0x12,        // Move from channel 1 to 2.
 *     TACTILE_PATTERN_OP_PLAY(100),       // Play for 100 ms.
 *     kTactilePatternOpEnd,
 *   };
 *   TactilePattern p;
 *   TactilePatternInit(&p, kSampleRateHz, kNumChannels);
 *   TactilePatternStartEx(&p, ex_pattern);
 *
 *   float samples[kNumFrames * kNumChannels];
 *   while (TactilePatternSynthesize(&p, kNumFrames, samples)) {
 *     // ...
 *   }
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PATTERN_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PATTERN_H_

#include <stdint.h>
#include "dsp/phase32.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gets the Play opcode for a duration in units of milliseconds, 20--640 ms.
 * (This is implemented as a macro rather than a regular function so that it
 * may be used within array initializers.)
 */
#define TACTILE_PATTERN_OP_PLAY_MS(duration_ms)           \
  (kTactilePatternOpPlay + ((duration_ms) < 20         \
      ? 0 : ((duration_ms) < 640                       \
          ? ((int) (duration_ms) - 10) / 20 : 31)))

enum { kTactilePatternMaxChannels = 16, kTactilePatternBufferSize = 129 };

/* Opcodes. */
enum {
  /* "End" op.
   * Indicates the end of the pattern. The synthesizer changes its state from
   * "Playing" to "Stopping" and fade out all channels.
   * Encoding: byte 0x00.
   */
  kTactilePatternOpEnd = 0x00,
  /* "Play" op.
   * Tells the synthesizer to generate samples for an indicated duration before
   * executing the next op. For instance if the SetWaveform op is followed by
   * Play, then that waveform is played for the duration.
   * Encoding: binary 100ddddd, where the 5 low bits represent the duration
   * (1 + d) * 20 ms. (This opcode likely makes up a sizable portion of the
   * pattern, so squeezing opcode and duration together into a single byte seems
   * worthwhile.)
   *
   * NOTE: After a Play op, all channels are silenced by default. To sustain a
   * channel instead of silencing, use SetWaveform or SetAllWaveform before the
   * next Play op.
   */
  kTactilePatternOpPlay = 0x80,
  /* "SetWaveform" op.
   * Sets the waveform for one channel. See the "Waveform indices" enum below
   * for possible waveforms.
   * Encoding: binary 1010cccc, where the 4 low bits represent the channel to
   * set, followed by a one-byte waveform index.
   */
  kTactilePatternOpSetWaveform = 0xa0,
  /* "SetGain" op.
   * Sets the gain for one channel. The gain determines the max amplitude of
   * the channel. If not set, the default gain is 0.15.
   * Encoding: binary 1011cccc, where the 4 low bits represent the channel to
   * set. This is followed by the gain, encoded as a uint8 representing a linear
   * gain between 0.0 and 1.0.
   */
  kTactilePatternOpSetGain = 0xb0,
  /* "SetAllWaveform" op.
   * The same as SetWaveform, but sets the waveform for all channels.
   * Encoding: byte 0x01 followed by a one-byte waveform index.
   */
  kTactilePatternOpSetAllWaveform = 0x01,
  /* "SetAllGain" op.
   * The same as SetGain, but sets the gain for all channels.
   * Encoding: byte 0x02 followed by 8-bit gain.
   */
  kTactilePatternOpSetAllGain = 0x02,
  /* "Move" op.
   * For creating movement patterns, this op "moves" a waveform from one
   * channel to another, copying the waveform to the destination channel and
   * silencing the source channel.
   * Encoding: byte 0x03 followed by a byte where the 4 high bits are the
   * source channel and the 4 low bits are the destination channel.
   */
  kTactilePatternOpMove = 0x03,
};

/* Waveform indices. */
enum {
  /* Pure sinusoidal tones. The tones range from 25 to 350 Hz, covering the
   * range where both hardware and tactile perception are responsive, and with a
   * median increment between successive tones of 20%, comparable to the 20-30%
   * JND for frequency change.
   */
  kTactilePatternWaveformSin25Hz,
  kTactilePatternWaveformSin30Hz,
  kTactilePatternWaveformSin35Hz,
  kTactilePatternWaveformSin45Hz,
  kTactilePatternWaveformSin50Hz,
  kTactilePatternWaveformSin60Hz,
  kTactilePatternWaveformSin70Hz,
  kTactilePatternWaveformSin90Hz,
  kTactilePatternWaveformSin100Hz,
  kTactilePatternWaveformSin125Hz,
  kTactilePatternWaveformSin150Hz,
  kTactilePatternWaveformSin175Hz,
  kTactilePatternWaveformSin200Hz,
  kTactilePatternWaveformSin250Hz,
  kTactilePatternWaveformSin300Hz,
  kTactilePatternWaveformSin350Hz,
  /* Rising exponential chirp. */
  kTactilePatternWaveformChirp,
};

/* Playback state. */
enum {
  kTactilePatternStateStopped,
  kTactilePatternStateStopping,
  kTactilePatternStatePlaying,
};

/* Synthesis state for one channel/tactor. */
typedef struct {
  /* Oscillator for synthesizing sine waves and chirps. */
  Oscillator tone;
  /* Current signal amplitude, or the final amplitude during a fade. */
  float amplitude;
  /* During a fade, `amplitude + amplitude_fade_delta` is the starting
   * amplitude and is weighted with `fade_weight`.
   */
  float amplitude_fade_delta;
  /* Linear gain for this channel. */
  float gain;
  /* Waveform currently being generated. */
  int waveform;
} TactilePatternChannel;

typedef struct {
  TactilePatternChannel channels[kTactilePatternMaxChannels];
  uint8_t buffer[kTactilePatternBufferSize];

  /* Pattern byte string. See the opcodes documentation above for details. */
  const uint8_t* pattern;
  /* Sample rate in Hz. */
  float sample_rate_hz;
  /* Number of output channels. Must be less than kTactilePatternMaxChannels. */
  int num_channels;

  /* Number of frames in a fade transition. */
  int fade_frames;
  /* Used for synthesizing chirps. */
  float chirp_rate;

  /* Synthesis playback state: stopped, stopping, or playing. */
  int playback_state;
  /* Number of frames to synthesize before executing the next op. */
  int num_frames_until_next_op;

  /* Oscillator for fading in and out by modulating a raised cosine. */
  Oscillator fade;
  float fade_weight;
  /* Number of frames left in the current fade transition. */
  int fade_counter;
} TactilePattern;

extern const char* kTactilePatternConnect;
extern const char* kTactilePatternDisconnect;
extern const char* kTactilePatternConfirm;
extern const uint8_t kTactilePatternExStartUp[];

/* Initializes the synthesizer. */
void TactilePatternInit(TactilePattern* p, float sample_rate_hz,
                        int num_channels);

/* Resets synthesizer to produce `simple_pattern`, in which each char represents
 * one "segment" of the pattern to play on all channels. '012...F' represent
 * the tones, '/' a rising exponential chirp, and '-' a pause.
 * TactilePatternInit() should be called first, then TactilePatterStart() may
 * be called at any time.
 */
int /*bool*/ TactilePatternStart(TactilePattern* p, const char* simple_pattern);

/* Translates a simple char-based pattern to the format understood by
 * `TactilePatternStartEx`. Returns 1 on success, 0 if buffer is too small.
 */
int /*bool*/ TactilePatternTranslateSimplePattern(const char* simple_pattern,
                                                  uint8_t* output,
                                                  int output_size);

/* Resets synthesizer to play an extended format ("ex") pattern, as described in
 * the opcode documentation above. TactilePatternInit() should be called first,
 * then TactilePatterStartEx() may be called at any time.
 */
void TactilePatternStartEx(TactilePattern* p, const uint8_t* ex_pattern);

/* Resets synthesizer to produce calibration tones on channels `first_channel`
 * and `second_channel`. If `first_channel == second_channel`, a single tone is
 * played on that channel. Otherwise, the synthesizer plays: (1) a tone on
 * first_channel, (2) a pause, (3) a tone on second_channel.
 */
void TactilePatternStartCalibrationTones(TactilePattern* p, int first_channel,
                                         int second_channel);

/* Resets synthesizer to produce calibration tones on channels `first_channel`
 * and `second_channel`. If `first_channel == second_channel`, a single tone is
 * played on that channel for a duration of 5 tone increments at the specified
 * amplitude. Otherwise, the synthesizer plays an A-B-A pattern: (1) a tone on
 * first_channel, (2) a pause, (3) a tone on second_channel, (4) a pause, (5)
 * a tone on first_channel.
 */
void TactilePatternStartCalibrationTonesThresholds(TactilePattern* p,
                                                   int first_channel,
                                                   int second_channel,
                                                   float amplitude);

/* Synthesizes `num_frames` of output to `num_channels` output channels in a
 * streaming manner. The `output` array must have space for `num_frames *
 * num_channels` samples. The function always synthesizes `num_frames` frames,
 * padding the output with silence once the pattern completes. Returns 1 if the
 * pattern is still playing, 0 if the pattern has completed.
 */
int /*bool*/ TactilePatternSynthesize(TactilePattern* p,
                                      int num_frames,
                                      float* output);

/* Returns 1 if the pattern is still playing, or 0 if completed. */
static int /*bool*/ TactilePatternIsActive(const TactilePattern* p) {
  return p->playback_state != kTactilePatternStateStopped;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PATTERN_H_ */
