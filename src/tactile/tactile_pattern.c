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

#include <stdint.h>
#include <math.h>

#include "dsp/fast_fun.h"

/* Tactile patterns are represented as a null-terminated char* string, each char
 * indicating one "segment" of the pattern. The synthesizer understands the
 * following codes:
 *
 *   Code  Duration        Waveform
 *   '-'   kPauseSeconds   Pause (silence).
 *   '0'   kToneSeconds     25.0 Hz sine wave.
 *   '1'   kToneSeconds     29.7 Hz sine wave.
 *   '2'   kToneSeconds     35.4 Hz sine wave.
 *   '3'   kToneSeconds     42.0 Hz sine wave.
 *   '4'   kToneSeconds     50.0 Hz sine wave.
 *   '5'   kToneSeconds     59.5 Hz sine wave.
 *   '6'   kToneSeconds     70.7 Hz sine wave.
 *   '7'   kToneSeconds     84.1 Hz sine wave.
 *   '8'   kToneSeconds    100.0 Hz sine wave.
 *   '9'   kToneSeconds    118.9 Hz sine wave.
 *   'A'   kToneSeconds    141.4 Hz sine wave.
 *   'B'   kToneSeconds    168.2 Hz sine wave.
 *   'C'   kToneSeconds    200.0 Hz sine wave.
 *   'D'   kToneSeconds    237.8 Hz sine wave.
 *   'E'   kToneSeconds    282.8 Hz sine wave.
 *   'F'   kToneSeconds    336.4 Hz sine wave.
 *   '/'   kChirpSeconds   Rising exponential chirp.
 *
 * Durations for tone and pause can be set by changing the duration fields
 * in the pattern object.
 *
 * Tactile perception has little sensitivity to frequencies above 300 Hz and a
 * frequency change JND of 20-30%. With hardware constraints, tactors have
 * difficulty producing frequencies much below 30 Hz. So all frequencies are
 * well represented with 16 tones covering 4 octaves (25--336.4 Hz) with 4 tones
 * per octave (increments of 19%).
 *
 * This makes it easy to build up an interesting signal as a sequence of simple
 * pieces. For instance, "6 A A" means "tone6, pause, toneA, pause, toneA". The
 * synthesizer takes care of fading sensibly where tones start and end.
 */

const char* kTactilePatternSilence = "";
/* "Connect" pattern: low tone followed by two higher tones. */
const char* kTactilePatternConnect = "66-A-A";
/* "Disconnect" pattern: middle, high, low sequence of tones. */
const char* kTactilePatternDisconnect = "8A-6";
/* "Confirm" pattern: two quick tones. */
const char* kTactilePatternConfirm = "5-5";

/* Pattern used for calibration tones: tone, pause, tone.
 * Calibration tones are played in a special way. The first tone is played on
 * `active_channel`, StartSegment() recognizes the pause in this pattern to
 * switch to playing the second tone on `second_channel`.
 */
static const char* kCalibrationTwoTonePattern = "999---999";
static const char* kCalibrationOneTonePattern = "999";
static const char* kCalibrationIntensityAdjustmentPattern =
    "99---99---99";  /* 400 ms stimulus, 300 ms interstimulus */
static const char* kCalibrationOneToneThresholdPattern = "99"; /* 400 ms */

static const float kAmplitude = 0.15f; /* Amplitude in [0.0, 1.0]. */
static const float kPauseSeconds = 0.03f;
static const float kPauseLong = 0.1f;
static const float kFadeSeconds = 0.02f;

static const float kToneSeconds = 0.08f;
static const float kToneLong = 0.2f;

static const float kChirpSeconds = 0.3f;
static const float kChirpStartHz = 40.0f;
static const float kChirpEndHz = 120.0f;

/* Converts seconds to frames according to p->sample_rate_hz. */
static int SecondsToFrames(const TactilePattern* p, float seconds) {
  return (int)(seconds * p->sample_rate_hz + 0.5f);
}

/* Converts frequency in Hz to uint32 phase according to p->sample_rate_hz. */
static uint32_t FrequencyToPhase32(const TactilePattern* p, float frequency) {
  return Phase32FromFloat(frequency / p->sample_rate_hz);
}

/* Returns 1 if `c` represents a silent segment, i.e. the end or a pause. */
static int IsSilent(char c) {
  return c == '\0' || c == '-';
}

/* Gets tone frequency where `c` is one of "0123456789ABCDEF". */
static float GetToneFrequency(char c) {
  /* Convert hex digit to int between 0 and 15. */
  const int i = (c < 'A') ? (c - '0') : (c - ('A' - 10));
  /* Lowest tone is 25 Hz, and there are four tones per octave. */
  return 25.0f * FastExp2(0.25f * i);
}

/* Sets up variables for the duration and tone frequency for the next segment,
 * and also handling fading in and out between silent and non-silent segments.
 * The `first` arg indicates whether this is the first segment of the sequence.
 */
static void StartSegment(TactilePattern* p, int first) {
  const char segment = p->pattern[0];

  /* Get the duration and tone frequency for this segment. */
  float duration_s;
  float frequency_hz;
  switch (segment) {
    case '0':
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
      duration_s = p->tone_duration;
      frequency_hz = GetToneFrequency(segment);
      break;
    case '/':
      duration_s = kChirpSeconds;
      frequency_hz = kChirpStartHz;
      break;
    case '\0':
      return;
    default: /* Treat '-' (or anything unrecognized) as a pause. */
      duration_s = p->pause_duration;
      frequency_hz = 0.0f;
      /* When playing calibration tones, use the pause to determine when to
       * switch the active channel between the first and second channels.
       * Only switch the channel on the first pause of a continuous pause
       * segment, and only if the pause is not the first character of
       * the pattern.
       */
      if (p->active_channel >= 0 && !first && !IsSilent(p->pattern[-1])) {
        if (p->active_channel != p->second_channel) {
          p->active_channel = p->second_channel;
        } else {
          p->active_channel = p->first_channel;
        }
      }
      break;
  }
  p->segment_counter = SecondsToFrames(p, duration_s);
  p->tone.frequency = FrequencyToPhase32(p, frequency_hz);

  if (!IsSilent(segment)) { /* Current segment is not silent. */
    /* If previous segment was silent, fade in. */
    if (first || IsSilent(p->pattern[-1])) {
      p->fade.phase = 0;
      p->fade_counter = p->fade_frames;
    }
    /* If next segment will be silent, fade out at the end of this segment. */
    if (IsSilent(p->pattern[1])) {
      p->fade_start_index = p->fade_frames;
    }
  }
}

void TactilePatternInit(TactilePattern* p, float sample_rate_hz) {
  p->sample_rate_hz = sample_rate_hz;
  /* Compute frequency such that half a cycle is kFadeSeconds. */
  p->fade.frequency = FrequencyToPhase32(p, 0.5f / kFadeSeconds);
  p->fade_frames = SecondsToFrames(p, kFadeSeconds);
  /* Compute such that
   * kChirpEndHz = kChirpStartHz * chirp_rate^(kChirpSeconds * sample_rate_hz).
   */
  p->chirp_rate =
      pow(kChirpEndHz / kChirpStartHz, 1.0f / (kChirpSeconds * sample_rate_hz));
  TactilePatternStart(p, kTactilePatternSilence);
}

void TactilePatternStart(TactilePattern* p, const char* pattern) {
  p->pattern = pattern;
  p->tone.phase = 0;
  p->segment_counter = 0;
  p->fade_counter = 0;
  p->fade_start_index = 0;
  p->active_channel = -1; /* All channels are active. */
  p->amplitude = kAmplitude;
  p->tone_duration = kToneSeconds;
  p->pause_duration = kPauseSeconds;
  StartSegment(p, 1);
}

void TactilePatternStartCalibrationTones(TactilePattern* p, int first_channel,
                                         int second_channel) {
  TactilePatternStart(p, first_channel == second_channel
                             ? kCalibrationOneTonePattern
                             : kCalibrationTwoTonePattern);
  p->active_channel = first_channel;
  p->first_channel = first_channel;
  p->second_channel = second_channel;
}

void TactilePatternStartCalibrationTonesThresholds(TactilePattern* p,
                                                   int first_channel,
                                                   int second_channel,
                                                   float amplitude) {
  p->active_channel = first_channel;
  p->first_channel = first_channel;
  p->second_channel = second_channel;
  p->amplitude = amplitude;
  p->pattern = first_channel == second_channel
                             ? kCalibrationOneToneThresholdPattern
                             : kCalibrationIntensityAdjustmentPattern;
  p->tone.phase = 0;
  p->segment_counter = 0;
  p->fade_counter = 0;
  p->fade_start_index = 0;
  p->tone_duration = kToneLong;
  p->pause_duration = kPauseLong;
  StartSegment(p, 1);
}

int TactilePatternSynthesize(TactilePattern* p, int num_frames,
                             int num_channels, float* output) {
  int active_channel = p->active_channel;
  const float amplitude = p->amplitude;

  int i;
  for (i = 0; i < num_frames; ++i, output += num_channels) {
    /* Generate the next sample. */
    const char segment = p->pattern[0];
    float value;
    if (segment == '\0') { /* End of the pattern. */
      value = 0.0f;
    } else {
      if (segment == '-') { /* Pause segment. */
        value = 0.0f;
      } else {
        /* Generate sine wave. */
        OscillatorNext(&p->tone);
        value = amplitude * Phase32Sin(p->tone.phase);
        if (segment == '/') { /* Chirp segment. */
          /* Grow the oscillator frequency to make an exponential chirp. */
          p->tone.frequency *= p->chirp_rate;
        }

        if (p->segment_counter == p->fade_start_index) {
          /* Begin fading out. */
          p->fade_counter = p->segment_counter;
          p->fade_start_index = 0;
        }
        if (p->fade_counter > 0) { /* Fading in or out. */
          p->fade_counter -= 1;
          /* Modulate with a raised cosine. */
          OscillatorNext(&p->fade);
          value *= 0.5f * (1.0f - Phase32Cos(p->fade.phase));
        }
      }

      if (--p->segment_counter == 0) { /* Segment just ended. */
        p->pattern += 1;
        StartSegment(p, 0);
        active_channel = p->active_channel;
      }
    }

    int c;
    if (active_channel < 0) { /* Play on all channels. */
      for (c = 0; c < num_channels; ++c) {
        output[c] = value;
      }
    } else { /* For calibration tones, play only on `active_channel`. */
      for (c = 0; c < num_channels; ++c) {
        output[c] = 0.0f;
      }
      output[active_channel] = value;
    }
  }

  return *p->pattern != '\0';
}
