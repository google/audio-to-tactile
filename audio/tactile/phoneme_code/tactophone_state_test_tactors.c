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
 * Test tactors screen.
 */

#include "audio/tactile/phoneme_code/tactophone_states.h"

/* Table mapping keyboard buttons to base-1 tactors. */
static const struct TestTactorsButton {
  char button;
  int channel;
} kTestTactorsTable[] = {
    /* Dorsal side. */
    {'2', 1},
    {'3', 5},
    {'4', 9},
    {'5', 13},
    {'6', 17},
    {'7', 21},

    {'W', 2},
    {'E', 6},
    {'R', 10},
    {'T', 14},
    {'Y', 18},
    {'U', 22},

    /* Volar side. */
    {'S', 4},
    {'D', 8},
    {'F', 12},
    {'G', 16},
    {'H', 20},
    {'J', 24},

    {'X', 3},
    {'C', 7},
    {'V', 11},
    {'B', 15},
    {'N', 19},
    {'M', 23},
};

/* Draw tactors menu with `active_channel` highlighted. */
void DrawTestTactorsMenu(int active_channel) {
  mvprintw(4, 2, "                      Top view\n");
  printw("        -------------------------------------------\n");
  printw("       | ");

  int i;
  for (i = 0; i < kNumChannels; ++i) {
    if (i == active_channel) {
      attron(A_BOLD | COLOR_PAIR(kColorHighlight));
    } else {
      attron(A_BOLD | COLOR_PAIR(kColorKey));
    }
    printw("[%c]%c", kTestTactorsTable[i].button,
           (i == active_channel) ? '*' : ' ');

    if (i != active_channel) {
      attroff(A_BOLD | COLOR_PAIR(kColorKey));
    }
    printw("%-2d ", kTestTactorsTable[i].channel);
    if (i == active_channel) {
      attroff(A_BOLD | COLOR_PAIR(kColorHighlight));
    }

    if (i == 5) {
      printw("|\n");
      printw("       |                                           |\n");
      printw(
          " Elbow |                Dorsal side                | Wrist  (Back "
          "of hand)\n");
      printw("       |                                           |\n");
      printw("       | ");
    } else if (i == 11) {
      printw("|\n");
      printw("        -------------------------------------------\n");

      printw("\n                        Top view\n");
      printw("        -------------------------------------------\n");
      printw("       | ");
    } else if (i == 17) {
      printw("|\n");
      printw("       |                                           |\n");
      printw(
          " Elbow |                Volar side                 | Wrist  (Palm "
          "of hand)\n");
      printw("       |                                           |\n");
      printw("       | ");
    }
  }

  printw("|\n");
  printw("        -------------------------------------------\n");
}

static void TestTactorsOnEnterState(TactophoneEngine* engine) {
  clear();
  mvprintw(0, 0, " TEST TACTORS");
  TactophoneFormatTitle();

  mvprintw(2, 2, "Use the indicated keys to test tactors.");
  DrawTestTactorsMenu(-1);

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n  [Q] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Back to main menu");

  TactophoneRefresh();
}

static void TestTactorsOnKeyPress(TactophoneEngine* engine, int c) {
  if (c == 'Q') {  /* Quit to main menu. */
    TactophoneSetState(engine, &kTactophoneStateMainMenu);
  } else {
    int i;
    for (i = 0; i < kNumChannels; ++i) {
      if (c == kTestTactorsTable[i].button) {  /* Test tactor. */
        TactophonePlayTestBuzz(engine, kTestTactorsTable[i].channel);
        DrawTestTactorsMenu(i);
        TactophoneRefresh();
        break;
      }
    }
  }
}

static void TestTactorsOnTick(TactophoneEngine* engine) {
  TactophoneVisualizeTactors(engine, 23, 45);
  TactophoneRefresh();
}

const TactophoneState kTactophoneStateTestTactors = {
    TestTactorsOnEnterState,
    TestTactorsOnKeyPress,
    TestTactorsOnTick,
};
