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
 * Lesson trial screen.
 */

#include "extras/references/taps/tactophone_states.h"

static void DrawTrialChoiceMenu(TactophoneEngine* engine,
                                int highlight_correct) {
  const TactophoneQuestion* q = CurrentQuestion(engine);
  move(6, 0);
  int i;
  for (i = 0; i < q->num_choices; ++i) {
    attron(A_BOLD | COLOR_PAIR(kColorHighlight));
    printw("\n   (%c) ", kMenuButtons[i]);
    attroff(A_BOLD | COLOR_PAIR(kColorHighlight));

    int attrs;
    if (!highlight_correct) {
      attrs = A_BOLD;
    } else if (i == engine->correct_choice) {
      attrs = A_BOLD | COLOR_PAIR(kColorGreen);
    } else {
      attrs = COLOR_PAIR(kColorRed);
    }
    attron(attrs);
    printw(q->choices[i].label);
    attroff(attrs);

    printw(" [%s]", q->choices[i].phonemes);

    if (highlight_correct && i == engine->correct_choice) {
      printw("  <");
    }
  }
}

static void LessonTrialOnEnterState(TactophoneEngine* engine) {
  TactophoneNextTrial(engine);

  clear();
  mvprintw(0, 0, " LESSON %s", CurrentLesson(engine)->name);
  TactophoneFormatTitle();
  mvprintw(4, 2, "Playing a tactile pattern. What does it represent?");

  DrawTrialChoiceMenu(engine, 0);

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  const TactophoneQuestion* q = CurrentQuestion(engine);
  printw("\n\n\n  [1-%d] ", q->num_choices);
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Select your answer");
  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("   [SPACE] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Replay");
  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n\n  [Q] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Quit lesson");

  TactophonePlayPhonemes(/* Play correct choice. */
                         engine, q->choices[engine->correct_choice].phonemes);

  TactophoneRefresh();
}

static void PrintWinMessage(void) {
  attron(A_BOLD | COLOR_PAIR(kColorGreen));
  switch (RandomInt(2)) {
    case 0:
      mvprintw(1, 2, "   ___                         _      _ ");
      mvprintw(2, 2, "  / __\\___  _ __ _ __ ___  ___| |_   / \\");
      mvprintw(3, 2, " / /  / _ \\| '__| '__/ _ \\/ __| __| /  /");
      mvprintw(4, 2, "/ /__| (_) | |  | | |  __| (__| |_ /\\_/");
      mvprintw(5, 2, "\\____/\\___/|_|  |_|  \\___|\\___|\\__|\\/");
      break;
    case 1:
      mvprintw(1, 2, "    ___ _       _____________ ____  __  _________");
      mvprintw(2, 2, "   /   | |     / / ____/ ___// __ \\/  |/  / ____/");
      mvprintw(3, 2, "  / /| | | /| / / __/  \\__ \\/ / / / /|_/ / __/");
      mvprintw(4, 2, " / ___ | |/ |/ / /___ ___/ / /_/ / /  / / /___");
      mvprintw(5, 2, "/_/  |_|__/|__/_____//____/\\____/_/  /_/_____/");
      break;
    case 2:
      mvprintw(1, 2, "   ____  __  __   __  ___________    __  ____");
      mvprintw(2, 2, "  / __ \\/ / / /   \\ \\/ / ____/   |  / / / / /");
      mvprintw(3, 2, " / / / / /_/ /     \\  / __/ / /| | / /_/ / /");
      mvprintw(4, 2, "/ /_/ / __  /      / / /___/ ___ |/ __  /_/");
      mvprintw(5, 2, "\\____/_/ /_/      /_/_____/_/  |_/_/ /_(_)");
      break;
  }
  attroff(A_BOLD | COLOR_PAIR(kColorGreen));
}

static void LessonTrialOnKeyPress(TactophoneEngine* engine, int c) {
  const TactophoneQuestion* q = CurrentQuestion(engine);
  if (c == 'Q') {  /* Quit to main menu. */
    TactophoneSetState(engine, &kTactophoneStateMainMenu);
  } else if (c == '\n' || c == ' ') {  /* Replay correct choice. */
    TactophonePlayPhonemes(engine, q->choices[engine->correct_choice].phonemes);
  } else {
    const char* p = strchr(kMenuButtons, c);
    if (p != NULL) {  /* Select choice. */
      int choice = p - kMenuButtons;
      if (0 <= choice && choice < q->num_choices) {
        clear();
        mvprintw(0, 0, " LESSON %s", CurrentLesson(engine)->name);
        TactophoneFormatTitle();
        DrawTrialChoiceMenu(engine, true);

        attron(A_BOLD | COLOR_PAIR(kColorKey));
        printw("\n\n\n  [1-%d] ", q->num_choices);
        attroff(A_BOLD | COLOR_PAIR(kColorKey));
        printw("Compare patterns  ");
        attron(A_BOLD | COLOR_PAIR(kColorKey));
        printw("   [SPACE] ");
        attroff(A_BOLD | COLOR_PAIR(kColorKey));
        printw("Continue");
        attron(A_BOLD | COLOR_PAIR(kColorKey));
        printw("\n\n  [Q] ");
        attroff(A_BOLD | COLOR_PAIR(kColorKey));
        printw("Quit lesson");

        if (TactophoneSelectChoice(engine, choice)) {
          PrintWinMessage();
          TactophonePlayWinBuzz(engine);
        } else {
          mvprintw(4, 2, "Incorrect.");
        }

        TactophoneRefresh();
        TactophoneSetState(engine, &kTactophoneStateLessonReview);
      }
    }
  }
}

static void LessonTrialOnTick(TactophoneEngine* engine) {
  TactophoneVisualizeTactors(engine, 23, 45);
  TactophoneRefresh();
}

const TactophoneState kTactophoneStateLessonTrial = {
    LessonTrialOnEnterState,
    LessonTrialOnKeyPress,
    LessonTrialOnTick,
};
