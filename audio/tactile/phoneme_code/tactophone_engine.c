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
 */

#include "audio/tactile/phoneme_code/tactophone_engine.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "audio/dsp/portable/logging.h"
#include "audio/tactile/phoneme_code/phoneme_code.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* Sample rate for tactile signals. */
const int kSampleRateHz = 44100;
/* Silence spacing in seconds between phoneme codes. */
const float kPhonemeSpacingInSeconds = 0.05f;
/* Duration of a game clock tick in milliseconds. */
const int kMillisecondsPerClockTick = 50;
/* Buttons to use for selecting menu choices. */
const char* kMenuButtons = "123456789ABCDEFGHIJKLMNOP";

void TactophoneEngineInit(TactophoneEngine* engine) {
  memset(engine, 0, sizeof(TactophoneEngine));
  engine->keep_running = 1;
  engine->tactile_player =
      CHECK_NOTNULL(TactilePlayerMake(kNumChannels, kSampleRateHz));
}

void TactophoneEngineFree(TactophoneEngine* engine) {
  TactophoneFreeLessonSet(engine->lesson_set);
  TactilePlayerFree(engine->tactile_player);
  if (engine->log_file != NULL) {
    fclose(engine->log_file);
  }
}

int TactophoneEngineRun(TactophoneEngine* engine, int key_press) {
  if (key_press != -1) {  /* Handle keyboard button press. */
    if (0 <= key_press && key_press < 255) { key_press = toupper(key_press); }
    engine->state->on_key_press(engine, key_press);
  }

  ++engine->clock_ticks;
  engine->state->on_tick(engine);
  return engine->keep_running;
}

void TactophoneSetState(TactophoneEngine* engine,
                        const TactophoneState* new_state) {
  engine->state = new_state;
  engine->state->on_enter_state(engine);
}

void TactophoneLog(TactophoneEngine* engine, const char* format, ...) {
  if (engine->log_file == NULL) {
    return;
  }

  time_t raw_time;
  struct tm* t;
  time(&raw_time);
  t = localtime(&raw_time);
  /* Write log prefix with the syntax "<clock_ticks> HH:MM:SS] ". */
  fprintf(engine->log_file, "%8u %02d:%02d:%02d] ",
          engine->clock_ticks, t->tm_hour, t->tm_min, t->tm_sec);

  va_list args;
  va_start(args, format);
  vfprintf(engine->log_file, format, args);
  va_end(args);

  fputc('\n', engine->log_file);
  fflush(engine->log_file);
}

void TactophoneSetLesson(TactophoneEngine* engine, int lesson) {
  engine->current_lesson = lesson;
  engine->current_question = -1;
  engine->correct_choice = -1;
  engine->trials_correct = 0;
  engine->trials_total = 0;
  engine->correct_in_a_row = 0;
  engine->max_correct_in_a_row = 0;

  TactophoneLog(engine, "SetLesson lesson=%d (%s)",
                lesson, CurrentLesson(engine)->name);
}

const TactophoneLesson* CurrentLesson(TactophoneEngine* engine) {
  return &engine->lesson_set->lessons[engine->current_lesson];
}

const TactophoneQuestion* CurrentQuestion(TactophoneEngine* engine) {
  return &CurrentLesson(engine)->questions[engine->current_question];
}

void TactophoneNextTrial(TactophoneEngine* engine) {
  engine->prev_question = engine->current_question;
  engine->prev_correct_choice = engine->correct_choice;

  const int exclude_previous = (engine->prev_question >= 0);

  int num = CurrentLesson(engine)->num_questions - exclude_previous;
  engine->current_question = RandomInt(num - 1);
  engine->current_question +=
      (exclude_previous && engine->current_question >= engine->prev_question);

  num = CurrentQuestion(engine)->num_choices - exclude_previous;
  engine->correct_choice = RandomInt(num - 1);
  engine->correct_choice +=
      (exclude_previous &&
       engine->correct_choice >= engine->prev_correct_choice);

  TactophoneLog(engine, "NextTrial lesson=%d, question=%d, num_choices=%d",
                engine->current_lesson, engine->current_question,
                CurrentQuestion(engine)->num_choices);
}

int TactophoneSelectChoice(TactophoneEngine* engine, int choice) {
  ++engine->trials_total;
  const int correct_choice = engine->correct_choice;
  TactophoneLog(engine, "SelectChoice selected=%d, correct=%d (%s vs. %s)",
                choice, correct_choice,
                CurrentQuestion(engine)->choices[choice].label,
                CurrentQuestion(engine)->choices[correct_choice].label);

  if (choice == correct_choice) {
    ++engine->trials_correct;
    ++engine->correct_in_a_row;
    if (engine->correct_in_a_row > engine->max_correct_in_a_row) {
      engine->max_correct_in_a_row = engine->correct_in_a_row;
    }
    return 1;
  } else {
    engine->correct_in_a_row = 0;
    return 0;
  }
}

static float SineWave(float frequency_hz, float t) {
  return sin(2 * M_PI * frequency_hz * t);
}

/* Generates a test buzz on one tactor, where `channel` is a base-1 index.
 * Waveform samples are allocated and returned, and should be freed by the
 * caller. The number of frames is written to `*num_frames`.
 */
static float* GenerateTestBuzz(int channel, float sample_rate_hz,
                               int* num_frames) {
  CHECK(1 <= channel && channel <= kNumChannels);
  const float kAmplitude = 0.15f;
  const float kDuration = 0.6f;
  const float kTransition = 0.1f;
  const float kFrequencyHz = 200.0f;
  *num_frames = (int)(kDuration * sample_rate_hz + 0.5f);
  float* waveform =
      (float*)CHECK_NOTNULL(malloc(kNumChannels * *num_frames * sizeof(float)));

  float* dest = waveform;
  int i;
  for (i = 0; i < *num_frames; ++i) {
    int c;
    for (c = 0; c < kNumChannels; ++c) {
      dest[c] = 0.0f;
    }
    const float t = i / sample_rate_hz;
    dest[channel - 1] = kAmplitude * TukeyWindow(kDuration, kTransition, t) *
                        SineWave(kFrequencyHz, t);
    dest += kNumChannels;
  }
  return waveform;
}

/* Generates a tactile pattern for winning a trial. Waveform samples are
 * allocated and returned, and should be freed by the caller. The number of
 * frames is written to `*num_frames`.
 */
static float* GenerateWinBuzz(float sample_rate_hz, int* num_frames) {
  const float kAmplitude = 0.1f;
  const float kDuration = 0.6f;
  const float kFrequencyHz = 120.0f;
  *num_frames = (int)(kDuration * sample_rate_hz + 0.5f);
  float* waveform =
      (float*)CHECK_NOTNULL(malloc(kNumChannels * *num_frames * sizeof(float)));

  float* dest = waveform;
  int i;
  for (i = 0; i < *num_frames; ++i) {
    const float t = i / sample_rate_hz;
    int c;
    for (c = 0; c < kNumChannels; ++c) {
      /* The sleeve has six "rings" of tactors. Each ring has four tactors that
       * happen to have consecutive channel indices, such that `ring = c / 4`
       * determines which ring channel `c` belongs to:
       *
       *                   Top view
       *           -------------------------
       *          |  0   1   2   3   4   5  |
       *   Elbow  |       Dorsal side       |  Wrist   (Back of hand)
       *          |  0   1   2   3   4   5  |
       *           -------------------------
       *
       *                   Top view
       *           -------------------------
       *          |  0   1   2   3   4   5  |
       *   Elbow  |       Volar side        |  Wrist   (Palm of hand)
       *          |  0   1   2   3   4   5  |
       *           -------------------------
       */
      const int ring = c / 4;
      /* Activate all the tactors in a pattern moving up to the wrist and then
       * back to the elbow.
       */
      dest[c] =
          kAmplitude * SineWave(kFrequencyHz, t) *
          (TukeyWindow(0.2f, 0.05f, t - 0.04f * ring) +
           TukeyWindow(0.2f, 0.05f, t - 0.4f + 0.04f * ring) * (ring != 5));
    }

    dest += kNumChannels;
  }
  return waveform;
}

static void PlayTactileSignal(TactophoneEngine* engine, float* samples,
                              int num_frames) {
  PermuteWaveformChannels(engine->channel_permutation, samples, num_frames,
                          kNumChannels);
  TactilePlayerPlay(engine->tactile_player, samples, num_frames);
}

void TactophonePlayPhonemes(TactophoneEngine* engine, const char* phonemes) {
  int num_frames;
  float* samples =
      GeneratePhonemeSignal(phonemes, kPhonemeSpacingInSeconds, NULL, 1.0f,
                            kSampleRateHz, &num_frames);
  PlayTactileSignal(engine, samples, num_frames);
  TactophoneLog(engine, "PlayPhonemes %s", phonemes);
}

void TactophonePlayTestBuzz(TactophoneEngine* engine, int channel) {
  int num_frames;
  float* samples = GenerateTestBuzz(channel, kSampleRateHz, &num_frames);
  PlayTactileSignal(engine, samples, num_frames);
}

void TactophonePlayWinBuzz(TactophoneEngine* engine) {
  int num_frames;
  float* samples = GenerateWinBuzz(kSampleRateHz, &num_frames);
  PlayTactileSignal(engine, samples, num_frames);
}

void TactophoneVisualizeTactors(TactophoneEngine* engine, int y, int x) {
  float rms[kNumChannels];
  TactilePlayerGetRms(engine->tactile_player,
                      0.001f * kMillisecondsPerClockTick, rms);

  /* Consider a tactor "active" if its current RMS value is above 1e-3. */
  int active[kNumChannels];
  int c;
  for (c = 0; c < kNumChannels; c++) {
    active[engine->channel_permutation[c]] = (rms[c] > 1e-3f);
  }

  /* To avoid ugly line wrapping, we constrain the X dimension so that the
   * visualization is on the screen. We don't constrain Y---the visualization
   * just clips if it goes off screen in the Y dimension without bad effects.
   */
  const int max_x = getmaxx(stdscr) - 15;
  if (max_x < x) {
    x = max_x;
  }
  if (x < 0) {
    x = 0;
  }

  mvprintw(y, x, " ------------- ");  /* Crude ASCII drawing of the sleeve. */
  mvprintw(y + 1, x, "| - - - - - - |");
  mvprintw(y + 2, x, "| - - - - - - |");
  mvprintw(y + 3, x, " ============= ");
  mvprintw(y + 4, x, "| - - - - - - |");
  mvprintw(y + 5, x, "| - - - - - - |");
  mvprintw(y + 6, x, " ------------- ");

  attron(A_BOLD | COLOR_PAIR(kColorHighlight));
  static const int kRow[4] = {1, 2, 5, 4};
  int j;
  for (j = 0; j < 6; ++j) {
    int i;
    for (i = 0; i < 4; ++i) {
      if (active[i + 4 * j]) {  /* Draw active tactors as yellow stars. */
        mvprintw(y + kRow[i], x + 2 + 2 * j, "*");
      }
    }
  }
  attroff(A_BOLD | COLOR_PAIR(kColorHighlight));
}

void TactophoneFormatTitle() {
  wmove(stdscr, 0, 0);
  chgat(-1, 0, kColorTitle, NULL);
}

void TactophoneRefresh() {
  /* ALSA may print error messages if e.g. underrun or occurs, which interacts
   * poorly with ncurses. To avoid these messages from obscuring game UI or
   * scrolling the screen, place the cursor a few rows above the bottom.
   */
  wmove(stdscr, getmaxy(stdscr) - 4, 0);
  wrefresh(stdscr);
}
