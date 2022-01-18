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
 */

#include "tactile/tactile_pattern.h"

#include <limits.h>
#include <math.h>
#include <string.h>
#include "dsp/fast_fun.h"

/* Default gain in [0, 1]. SetGain and SetAllGain ops can override this. */
static const float kDefaultGain = 0.15f;
/* Duration of fading in or out. Must be <= 0.02 s, the min play duration. */
static const float kFadeSeconds = 0.02f;
/* Parameters for the Chirp waveform. */
static const float kChirpSeconds = 0.3f;
static const float kChirpStartHz = 40.0f;
static const float kChirpEndHz = 120.0f;

/* "Connect" pattern: low tone followed by two higher tones. */
const char* kTactilePatternConnect = "66-A-A";
/* "Disconnect" pattern: middle, high, low sequence of tones. */
const char* kTactilePatternDisconnect = "8A-6";
/* "Confirm" pattern: two quick tones. */
const char* kTactilePatternConfirm = "5-5";

/* Converts seconds to frames according to p->sample_rate_hz. */
static int SecondsToFrames(const TactilePattern* p, float seconds) {
  return (int)(seconds * p->sample_rate_hz + 0.5f);
}

/* Converts frequency in Hz to uint32 phase according to p->sample_rate_hz. */
static uint32_t FrequencyToPhase32(const TactilePattern* p, float frequency) {
  return Phase32FromFloat(frequency / p->sample_rate_hz);
}

/* Decodes a linear gain from 8-bit value. */
static float DecodeGain(uint8_t byte) {
  return (float) byte * 3.921569e-03 /* = 1.0 / 255 */;
}

/* Sets the amplitude for channel `c` to `amplitude` and starts fading. */
static void SetAmplitude(TactilePattern* p, int c, float amplitude) {
  TactilePatternChannel* channel = &p->channels[c];

  /* Start fading to the new amplitude. */
  channel->amplitude_fade_delta += channel->amplitude - amplitude;
  channel->amplitude = amplitude;
  p->fade.phase = 0;
  p->fade_counter = p->fade_frames;
}

/* Sets the waveform for channel `c` to `waveform`. */
static void SetWaveform(TactilePattern* p, int c, int waveform) {
  TactilePatternChannel* channel = &p->channels[c];
  if (channel->waveform == waveform && channel->amplitude > 0.0f) { return; }
  float frequency_hz = 0.0f;
  float amplitude = channel->gain;

  if (kTactilePatternWaveformSin25Hz <= waveform &&
      waveform <= kTactilePatternWaveformSin350Hz) {
    static const int16_t kFrequenciesHz[16] = {
        25, 30, 35, 45, 50, 60, 70, 90, 100, 125, 150, 175, 200, 250, 300, 350};
    frequency_hz = kFrequenciesHz[waveform - kTactilePatternWaveformSin25Hz];
  } else if (waveform == kTactilePatternWaveformChirp) {
    frequency_hz = kChirpStartHz;
  } else {
    amplitude = 0.0f;
  }

  channel->waveform = waveform;
  if (frequency_hz > 0.0f) {
    channel->tone.frequency = FrequencyToPhase32(p, frequency_hz);
  }
  SetAmplitude(p, c, amplitude);
}

/* Sets the linear gain for channel `c` to `gain` and starts fading. */
static void SetGain(TactilePattern* p, int c, float gain) {
  TactilePatternChannel* channel = &p->channels[c];

  if (channel->amplitude > 0.0f) {
    SetAmplitude(p, c, gain);
  }
  channel->gain = gain;
}

/* "Moves" a waveform from channel `c_from` to channel `c_to`. */
static void MoveChannel(TactilePattern* p, int c_from, int c_to) {
  TactilePatternChannel* channel_from = &p->channels[c_from];
  TactilePatternChannel* channel_to = &p->channels[c_to];

  channel_to->tone.frequency = channel_from->tone.frequency;
  SetAmplitude(p, c_to,
               channel_from->amplitude + channel_from->amplitude_fade_delta);
}

static void Stop(TactilePattern* p) {
  p->playback_state = kTactilePatternStateStopping;
  p->num_frames_until_next_op = INT_MAX;
}

/* Executes ops until the next Play or End op. */
static void ExecuteOps(TactilePattern* p) {
  if (p->playback_state != kTactilePatternStatePlaying) {  /* Pattern ended. */
    /* Set main loop to run INT_MAX times before asking again. */
    p->num_frames_until_next_op = INT_MAX;
    return;
  }

  const int num_channels = p->num_channels;
  int8_t channel_activated[kTactilePatternMaxChannels] = {0};

  while (p->num_frames_until_next_op <= 0) {
    uint_fast8_t opcode = *(p->pattern++);  /* Read next opcode. */

    /* Opcodes >= 0x80 indicate an action with the high bits and a parameter
     * (usually a channel index) with the lower bits.
     */
    if (opcode >= 0x80) {
      switch (opcode & 0xf0) {
        /* Play: synthesize for a specified duration before reading next op. */
        case kTactilePatternOpPlay:
        case kTactilePatternOpPlay + 0x10: {
          /* Convert 5-bit duration code to units of frames. */
          const float duration_s = ((opcode & 0x1f) + 1) * 0.02f;
          p->num_frames_until_next_op = SecondsToFrames(p, duration_s);
        } break;

        /* SetWaveform: set the waveform for one channel. */
        case kTactilePatternOpSetWaveform:
          SetWaveform(p, /*c=*/opcode & 0xf, /*waveform=*/*(p->pattern++));
          channel_activated[opcode & 0xf] = 1;
          break;

        /* SetGain: set the gain for one channel. */
        case kTactilePatternOpSetGain:
          SetGain(p, /*c=*/opcode & 0xf, /*gain=*/DecodeGain(*(p->pattern++)));
          break;

        /* For any other opcode, set state to Stopping. */
        default:
          Stop(p);
          memset(channel_activated, 0, kTactilePatternMaxChannels);
      }
    } else {
      switch (opcode) {
        /* SetAllWaveform: set the waveform for all channels. */
        case kTactilePatternOpSetAllWaveform: {
          const int waveform = *(p->pattern++);
          int c;
          for (c = 0; c < num_channels; ++c) {
            SetWaveform(p, c, waveform);
          }
          memset(channel_activated, 1, kTactilePatternMaxChannels);
          break;
        }

        /* SetAllGain: set the gain for all channels. */
        case kTactilePatternOpSetAllGain: {
          const float gain = DecodeGain(*(p->pattern++));
          int c;
          for (c = 0; c < num_channels; ++c) {
            SetGain(p, c, gain);
          }
          break;
        }

        /* Move: Move the waveform from one channel to another. */
        case kTactilePatternOpMove: {
          /* Read `c_from` and `c_to` from the next byte. */
          const uint_fast8_t channels = *(p->pattern++);
          const int c_from = channels >> 4;
          const int c_to = channels & 0x0f;
          MoveChannel(p, c_from, c_to);
          channel_activated[c_from] = 0;
          channel_activated[c_to] = 1;
        } break;

        /* For any other opcode, set state to Stopping. */
        default:
          Stop(p);
          memset(channel_activated, 0, kTactilePatternMaxChannels);
      }
    }
  }

  /* Set any channels that didn't set a waveform to silence. */
  int c;
  for (c = 0; c < num_channels; ++c) {
    if (!channel_activated[c]) {
      SetAmplitude(p, c, 0.0f);
    }
  }
}

/* Updates fading state when a waveform is fading in/out to a new amplitude. */
static void UpdateFadingState(TactilePattern* p) {
  if (--p->fade_counter == 0) { /* Fading just completed. */
    if (p->playback_state == kTactilePatternStateStopping) {
      p->playback_state = kTactilePatternStateStopped;
    }

    int c;
    for (c = 0; c < p->num_channels; ++c) {
      p->channels[c].amplitude_fade_delta = 0.0f;
    }
    return;
  }

  /* Fade smoothly using a Hann window. */
  OscillatorNext(&p->fade);
  p->fade_weight = 0.5f * (1.0f + Phase32Cos(p->fade.phase));
}

/* Generate the next sample for channel `c`. */
static float GenerateSample(TactilePattern* p, int c) {
  TactilePatternChannel* channel = &p->channels[c];

  OscillatorNext(&channel->tone);
  float value = Phase32Sin(channel->tone.phase);

  if (channel->waveform == kTactilePatternWaveformChirp) {
    /* Grow the oscillator frequency to make an exponential chirp. */
    channel->tone.frequency *= p->chirp_rate;
  }

  if (p->fade_counter) {
    value *= channel->amplitude
        + channel->amplitude_fade_delta * p->fade_weight;
  } else {
    value *= channel->amplitude;
  }

  return value;
}

void TactilePatternInit(TactilePattern* p, float sample_rate_hz,
                     int num_channels) {
  if (!(0 <= num_channels && num_channels <= kTactilePatternMaxChannels)) {
    num_channels = 0;
  }

  p->sample_rate_hz = sample_rate_hz;
  p->num_channels = num_channels;
  /* Compute frequency such that half a cycle is kFadeSeconds. */
  p->fade.frequency = FrequencyToPhase32(p, 0.5f / kFadeSeconds);
  p->fade_frames = SecondsToFrames(p, kFadeSeconds);
  /* Compute such that
   * kChirpEndHz = kChirpStartHz * chirp_rate^(kChirpSeconds * sample_rate_hz).
   */
  p->chirp_rate =
      pow(kChirpEndHz / kChirpStartHz, 1.0f / (kChirpSeconds * sample_rate_hz));
  TactilePatternStartEx(p, NULL);
}

void TactilePatternStartEx(TactilePattern* p, const uint8_t* ex_pattern) {
  p->pattern = ex_pattern;
  p->num_frames_until_next_op = 0;
  p->fade_counter = 0;
  p->playback_state =
      ex_pattern ? kTactilePatternStatePlaying : kTactilePatternStateStopped;

  int c;
  for (c = 0; c < p->num_channels; ++c) {
    TactilePatternChannel* channel = &p->channels[c];
    channel->tone.phase = 0;
    channel->tone.frequency = 0;
    channel->waveform = kTactilePatternWaveformSin25Hz;
    channel->amplitude = 0.0f;
    channel->amplitude_fade_delta = 0.0f;
    channel->gain = kDefaultGain;
  }
}

int TactilePatternSynthesize(TactilePattern* p, int num_frames, float* output) {
  const int num_channels = p->num_channels;

  int i;
  for (i = 0; i < num_frames; ++i, output += num_channels) {
    if (p->num_frames_until_next_op <= 0) {
      ExecuteOps(p); /* Execute ops until the next Play or End op. */
    }
    p->num_frames_until_next_op -= 1;

    if (p->fade_counter) { /* If fading, update fading state variables. */
      UpdateFadingState(p);
    }

    int c;
    for (c = 0; c < num_channels; ++c) {
      output[c] = GenerateSample(p, c);
    }
  }

  return p->playback_state != kTactilePatternStateStopped;
}

void TactilePatternStartCalibrationTones(TactilePattern* p, int first_channel,
                                         int second_channel) {
  static uint8_t kPattern[8] = {
      /* Play 125 Hz tone on the first channel for 240 ms. */
      kTactilePatternOpSetWaveform /* + first_channel (added below) */,
      kTactilePatternWaveformSin125Hz,
      TACTILE_PATTERN_OP_PLAY_MS(240),
      /* 100 ms pause. */
      TACTILE_PATTERN_OP_PLAY_MS(100),
      /* Play 125 Hz tone on the second channel for 240 ms. */
      kTactilePatternOpSetWaveform /* + second_channel (added below) */,
      kTactilePatternWaveformSin125Hz,
      TACTILE_PATTERN_OP_PLAY_MS(240),
      kTactilePatternOpEnd,
  };

  uint8_t* pattern = p->buffer;
  memcpy(pattern, kPattern, sizeof(kPattern));

  pattern[0] += first_channel;

  if (second_channel == first_channel) {
    /* If the two given channels are the same, stop after the first tone. */
    pattern[3] = kTactilePatternOpEnd;
  } else {
    pattern[4] += second_channel;
  }

  TactilePatternStartEx(p, p->buffer);
}

void TactilePatternStartCalibrationTonesThresholds(TactilePattern* p,
                                                   int first_channel,
                                                   int second_channel,
                                                   float amplitude) {
  static uint8_t kPattern[14] = {
    /* Set the gain to the specified amplitude. */
    kTactilePatternOpSetAllGain, 0xff /* Placeholder (set below). */,
    /* Play 125 Hz tone on the first channel for 400 ms. */
    kTactilePatternOpSetWaveform /* + first_channel (added below) */,
    kTactilePatternWaveformSin125Hz,
    TACTILE_PATTERN_OP_PLAY_MS(400),
    /* 300 ms pause. */
    TACTILE_PATTERN_OP_PLAY_MS(300),
    /* Play 125 Hz tone on the second channel for 400 ms. */
    kTactilePatternOpSetWaveform /* + second_channel (added below) */,
    kTactilePatternWaveformSin125Hz,
    TACTILE_PATTERN_OP_PLAY_MS(400),
    /* 300 ms pause. */
    TACTILE_PATTERN_OP_PLAY_MS(300),
    /* Play 125 Hz tone on the first channel for 400 ms. */
    kTactilePatternOpSetWaveform /* + first_channel (added below) */,
    kTactilePatternWaveformSin125Hz,
    TACTILE_PATTERN_OP_PLAY_MS(400),
    kTactilePatternOpEnd,
  };

  uint8_t* pattern = p->buffer;
  memcpy(pattern, kPattern, sizeof(kPattern));

  /* Clip amplitude to [0, 1]. */
  amplitude = (amplitude > 1.0f) ? 1.0f : (amplitude > 0.0f) ? amplitude : 0.0f;
  pattern[1] = (uint8_t) (255 * amplitude + 0.5f);
  pattern[2] += first_channel;

  if (second_channel == first_channel) {
    /* If the two given channels are the same, stop after the first tone. */
    pattern[5] = kTactilePatternOpEnd;
  } else {
    pattern[6] += second_channel;
    pattern[10] += first_channel;
  }

  TactilePatternStartEx(p, p->buffer);
}

int TactilePatternStart(TactilePattern* p, const char* simple_pattern) {
  const int /*bool*/ success = TactilePatternTranslateSimplePattern(
      simple_pattern, p->buffer, kTactilePatternBufferSize);
  TactilePatternStartEx(p, success ? p->buffer : NULL);
  return success;
}

int TactilePatternTranslateSimplePattern(const char* simple_pattern,
                                         uint8_t* output, int output_size) {
  if (!simple_pattern || !output || output_size <= 0) { return 0; }

  char segment;
  while ((segment = *(simple_pattern++)) != '\0') {
    switch (segment) {
      case '0': /* Tone. */
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case 'A':
      case 'B':
      case 'C':
      case 'D':
      case 'E':
      case 'F':
        if (output_size <= 3) { return 0; }
        *output++ = kTactilePatternOpSetAllWaveform;
        *output++ = kTactilePatternWaveformSin25Hz +
          ((segment < 'A') ? (segment - '0') : (segment - ('A' - 10)));
        *output++ = TACTILE_PATTERN_OP_PLAY_MS(80);
        output_size -= 3;
        break;

      case '/': /* Chirp. */
        if (output_size <= 3) { return 0; }
        *output++ = kTactilePatternOpSetAllWaveform;
        *output++ = kTactilePatternWaveformChirp;
        *output++ = TACTILE_PATTERN_OP_PLAY_MS(300);
        output_size -= 3;
        break;

      default: /* Pause, or an unrecognized segment. */
        if (output_size <= 1) { return 0; }
        *output++ = TACTILE_PATTERN_OP_PLAY_MS(40);
        --output_size;
        break;
    }
  }

  *output = kTactilePatternOpEnd;
  return 1;
}
