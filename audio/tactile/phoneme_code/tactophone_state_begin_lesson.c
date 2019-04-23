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
 * Begin lesson screen.
 */

#include "audio/tactile/phoneme_code/tactophone_states.h"

static void BeginLessonOnEnterState(TactophoneEngine* engine) {
  clear();
  mvprintw(0, 0, " LESSON %s", CurrentLesson(engine)->name);
  TactophoneFormatTitle();

  mvprintw(2, 2, "Get ready . . . . . . . . . . . . . . .\n\n");
  printw("   - Put the tactile sleeve on now.\n\n");
  printw("   - Tactile patterns will play on the sleeve.\n");
  printw("     Try to identify what the pattern represents.\n\n");
  printw("   - Use keys [1-5] to select answer.\n\n");
  attron(A_BOLD | COLOR_PAIR(kColorHighlight));
  printw("\n  > > > Press ");
  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("[SPACE] ");
  attron(A_BOLD | COLOR_PAIR(kColorHighlight));
  printw("to start < < <");
  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n\n\n  [Q] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Back to main menu");

  TactophoneRefresh();
}

static void BeginLessonOnKeyPress(TactophoneEngine* engine, int c) {
  if (c == 'Q') {  /* Quit to main menu. */
    TactophoneSetState(engine, &kTactophoneStateMainMenu);
  } else if (c == '\n' || c == ' ') {  /* Start the lesson. */
    engine->start_lesson_clock_ticks = engine->clock_ticks;
    TactophoneSetState(engine, &kTactophoneStateLessonTrial);
  }
}

static void BeginLessonOnTick(TactophoneEngine* engine) {
  /* Animate arrows around "Press [SPACE] to start". */
  if (engine->clock_ticks % 8 == 0) {
    attron(A_BOLD | COLOR_PAIR(kColorHighlight));
    move(12, 0);
    if (engine->clock_ticks % 16) {
      printw("   > > >Press ");
      attron(A_BOLD | COLOR_PAIR(kColorKey));
      printw("[SPACE] ");
      attron(A_BOLD | COLOR_PAIR(kColorHighlight));
      printw("to start< < < ");
    } else {
      printw("  > > > Press ");
      attron(A_BOLD | COLOR_PAIR(kColorKey));
      printw("[SPACE] ");
      attron(A_BOLD | COLOR_PAIR(kColorHighlight));
      printw("to start < < <");
    }
    attroff(A_BOLD | COLOR_PAIR(kColorHighlight));
  }

  TactophoneRefresh();
}

const TactophoneState kTactophoneStateBeginLesson = {
    BeginLessonOnEnterState,
    BeginLessonOnKeyPress,
    BeginLessonOnTick,
};
