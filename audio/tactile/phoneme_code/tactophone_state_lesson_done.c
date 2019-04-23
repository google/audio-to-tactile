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
 * Lesson done.
 */

#include "audio/tactile/phoneme_code/tactophone_states.h"

static void LessonDoneOnEnterState(TactophoneEngine* engine) {
  clear();
  mvprintw(0, 0, " LESSON %s", CurrentLesson(engine)->name);
  TactophoneFormatTitle();
  mvprintw(3, 2, "Lesson complete");

  const int trials_correct = engine->trials_correct;
  const int trials_total = engine->trials_total;
  mvprintw(5, 3, "Accuracy");
  attron(A_BOLD | COLOR_PAIR(kColorHighlight));
  mvprintw(5, 26, "%d%% (%d / %d)",
           (100 * trials_correct + trials_total / 2) / trials_total,
           trials_correct, trials_total);
  attroff(A_BOLD | COLOR_PAIR(kColorHighlight));

  mvprintw(7, 3, "Max correct in a row");
  attron(A_BOLD | COLOR_PAIR(kColorHighlight));
  mvprintw(7, 26, "%d", engine->max_correct_in_a_row);
  attroff(A_BOLD | COLOR_PAIR(kColorHighlight));

  const float elapsed_s =  /* Compute time elapsed while in lesson. */
      (0.001f * kMillisecondsPerClockTick) *
      (engine->clock_ticks - engine->start_lesson_clock_ticks);
  const int int_elapsed_s = (int)(elapsed_s + 0.5f);
  const int minutes = int_elapsed_s / 60;
  const int seconds = int_elapsed_s % 60;
  mvprintw(9, 3, "Time");
  attron(A_BOLD | COLOR_PAIR(kColorHighlight));
  mvprintw(9, 26, "%d:%02d (average %.1fs per trial)", minutes, seconds,
           elapsed_s / trials_total);

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  mvprintw(12, 3, "[SPACE] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Back to main menu");

  TactophoneLog(engine, "LessonDone accuracy=%d/%d, "
                "max_correct_in_a_row=%d, elapsed=%.1fs",
                trials_correct, trials_total,
                engine->max_correct_in_a_row, elapsed_s);
  TactophoneRefresh();
}

static void LessonDoneOnKeyPress(TactophoneEngine* engine, int c) {
  if (c == '\n' || c == ' ' || c == 'Q') {  /* Quit to main menu. */
    TactophoneSetState(engine, &kTactophoneStateMainMenu);
  }
}

static void LessonDoneOnTick(TactophoneEngine* engine) { TactophoneRefresh(); }

const TactophoneState kTactophoneStateLessonDone = {
    LessonDoneOnEnterState,
    LessonDoneOnKeyPress,
    LessonDoneOnTick,
};
