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
 * Tactophone main menu.
 */

#include "audio/tactile/phoneme_code/tactophone_states.h"

static void MainMenuOnEnterState(TactophoneEngine* engine) {
  clear();
  attron(A_BOLD);
  mvprintw(0, 0, "   _____          _              _ \n");
  printw("  |_   _|        | |            | |\n");
  printw("    | | __ _  ___| |_ ___  _ __ | |__   ___  _ __   ___ \n");
  printw("    | |/ _` |/ __| __/ _ \\| '_ \\| '_ \\ / _ \\| '_ \\ / _ \\\n");
  printw("    | | (_| | (__| || (_) | |_) | | | | (_) | | | |  __/\n");
  printw("    |_|\\__,_|\\___|\\__\\___/| .__/|_| |_|\\___/|_| |_|\\___|\n");
  printw("                          | |\n");
  printw("                          |_|\n");
  attroff(A_BOLD);

  printw("  Select lesson\n");

  int i;
  for (i = 0; i < engine->lesson_set->num_lessons; ++i) { /* Print lessons. */
    attron(A_BOLD | COLOR_PAIR(kColorKey));
    printw("\n  [%c] ", kMenuButtons[i]);
    attron(A_BOLD | COLOR_PAIR(kColorHighlight));
    printw(engine->lesson_set->lessons[i].name);
    attroff(A_BOLD | COLOR_PAIR(kColorHighlight));
  }

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n\n  [F] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Free play");

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n  [T] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Test tactors");

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n\n  [Q] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Quit program");

  TactophoneRefresh();
}

static void MainMenuOnKeyPress(TactophoneEngine* engine, int c) {
  if (c == 'Q') {  /* Quit program. */
    engine->keep_running = 0;
  } else if (c == 'F') {  /* Free play. */
    TactophoneSetState(engine, &kTactophoneStateFreePlay);
  } else if (c == 'T') {  /* Test tactors. */
    TactophoneSetState(engine, &kTactophoneStateTestTactors);
  } else {
    const char* p = strchr(kMenuButtons, c);
    if (p != NULL) {  /* Select lesson. */
      const int lesson = p - kMenuButtons;
      if (0 <= lesson && lesson < engine->lesson_set->num_lessons) {
        /* Begin playing lesson. */
        TactophoneSetLesson(engine, lesson);
        TactophoneSetState(engine, &kTactophoneStateBeginLesson);
      }
    }
  }
}

static void MainMenuOnTick(TactophoneEngine* engine) { TactophoneRefresh(); }

const TactophoneState kTactophoneStateMainMenu = {
    MainMenuOnEnterState,
    MainMenuOnKeyPress,
    MainMenuOnTick,
};
