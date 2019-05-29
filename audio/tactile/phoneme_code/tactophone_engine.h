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
 * Tactophone engine for event handling and Tactophone game variables.
 */

#ifndef AUDIO_TACTILE_PHONEME_CODE_TACTOPHONE_ENGINE_H_
#define AUDIO_TACTILE_PHONEME_CODE_TACTOPHONE_ENGINE_H_

#include <stdio.h>
#include <string.h>

#include "ncurses.h"
#include "portaudio.h"

#include "audio/tactile/phoneme_code/tactophone_lesson.h"
#include "audio/tactile/tactile_player.h"
#include "audio/tactile/util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define kNumChannels 24

extern const int kSampleRateHz;
extern const float kPhonemeSpacingInSeconds;
extern const int kMillisecondsPerClockTick;
extern const char* kMenuButtons;

/* Enumeration for ncurses COLOR_PAIR definitions. */
enum ColorPairs {
  kColorTitle = 1,      /* Color for title bars. */
  kColorKey = 2,        /* Keys like "[Q]" and "[SPACE]". */
  kColorHighlight = 3,  /* Highlight for important text. */
  kColorGreen = 4,      /* Green for correct choice. */
  kColorRed = 5,        /* Red for wrong choice. */
};

struct TactophoneEngine;

/* Different UI screens are implemented as different "states", e.g. main menu
 * vs. lesson trials. A state is defined by three event-handler callbacks.
 */
struct TactophoneState {
  /* Called when the state is entered. Usually it draws UI for the state. */
  void (*on_enter_state)(struct TactophoneEngine*);
  /* Called when a key is pressed. */
  void (*on_key_press)(struct TactophoneEngine*, int);
  /* Called once per game tick, useful for animation. */
  void (*on_tick)(struct TactophoneEngine*);
};
typedef struct TactophoneState TactophoneState;

struct TactophoneEngine {
  /* UI state. */
  const TactophoneState* state;
  unsigned clock_ticks;
  int keep_running;

  /* Tactile output. */
  PaStream* portaudio_stream;
  float gain;
  const int* channel_permutation;
  TactilePlayer* tactile_player;

  /* Logging. */
  FILE* log_file;

  /* Lesson variables. */
  TactophoneLessonSet* lesson_set;
  unsigned start_lesson_clock_ticks;
  int current_lesson;
  int current_question;
  int correct_choice;
  int prev_question;
  int prev_correct_choice;
  int trials_correct;
  int trials_total;
  int correct_in_a_row;
  int max_correct_in_a_row;
};
typedef struct TactophoneEngine TactophoneEngine;

/* Initializes all engine variables. `keep_running` is set to 1.
 * `tactile_player` is initialized. All other variables are set to zero.
 */
void TactophoneEngineInit(TactophoneEngine* engine);

/* Frees engine resources. */
void TactophoneEngineFree(TactophoneEngine* engine);

/* Runs the engine for one tick. This function should be called in a loop to
 * implement an event loop. The `c` arg should be the key pressed since the
 * last tick or -1 if no key was pressed. Returns 1 if the event loop should
 * keep running.
 *
 * The engine must be in nonnull state before calling this function. Use
 * TactophoneSetState to set the initial state.
 *
 * Example use with ncurses:
 *   TactophoneEngine engine;
 *   TactophoneEngineInit(&engine);
 *   TactophoneSetState(&engine, &kInitialState);
 *
 *   int key_press;
 *   do {
 *     // Wait up to kMillisecondsPerClockTick for a key press.
 *     timeout(kMillisecondsPerClockTick);
 *     key_press = getch();
 *   } while (TactophoneEngineRun(&engine, key_press));
 */
int TactophoneEngineRun(TactophoneEngine* engine, int key_press);

/* Sets Tactophone state to `new_state` and calls `on_enter_state`. */
void TactophoneSetState(TactophoneEngine* engine,
                        const TactophoneState* new_state);

/* Write formatted output to log file. */
void TactophoneLog(TactophoneEngine* engine, const char* format, ...);


/* Lesson-related functions. */

/* Sets the current lesson to `lesson`. */
void TactophoneSetLesson(TactophoneEngine* engine, int lesson);
/* Returns pointer to current TactophoneLesson. */
const TactophoneLesson* CurrentLesson(TactophoneEngine* engine);
/* Returns pointer to current TactophoneQuestion. */
const TactophoneQuestion* CurrentQuestion(TactophoneEngine* engine);
/* Starts the next lesson trial, selected randomly from CurrentLesson(). */
void TactophoneNextTrial(TactophoneEngine* engine);
/* Makes selection for the current lesson trial and updates lesson stats. */
int TactophoneSelectChoice(TactophoneEngine* engine, int choice);


/* Tactile output-related functions. */

/* Plays tactile signal for a string of phonemes. */
void TactophonePlayPhonemes(TactophoneEngine* engine, const char* phonemes);
/* Plays a test buzz on one tactor (this is used in the "test tactors" screen).
 * `channel` is a base-1 index in Purdue's enumeration, described in
 * phoneme_code.h.
 */
void TactophonePlayTestBuzz(TactophoneEngine* engine, int channel);
/* Plays a tactile pattern for winning a trial. */
void TactophonePlayWinBuzz(TactophoneEngine* engine);


/* Text display-related functions. */

/* Draws a visualization of tactor activity with upper-left corner (y, x). */
void TactophoneVisualizeTactors(TactophoneEngine* engine, int y, int x);
/* Sets color for row 0 to make a title bar. */
void TactophoneFormatTitle();
/* Moves cursor and refreshes ncurses. */
void TactophoneRefresh();

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_PHONEME_CODE_TACTOPHONE_ENGINE_H_ */
