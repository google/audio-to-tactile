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
 * Phoneme free play screen.
 */

#include "extras/references/taps/tactophone_states.h"

/* Table mapping keyboard buttons to phonemes. */
static const struct FreePlayPhonemeButton {
  char button;
  const char* phoneme;
} kFreePlayPhonemeTable[] = {
    /* Row 0. */
    {'1', "AE"},
    {'2', "AH"},
    {'3', "AW"},
    {'4', "AY"},

    {'5', "B"},
    {'6', "CH"},
    {'7', "D"},
    {'8', "DH"},
    {'9', "F"},
    {'0', "G"},

    /* Row 1. */
    {'W', "EE"},
    {'E', "EH"},
    {'R', "ER"},

    {'T', "H"},
    {'Y', "J"},
    {'U', "K"},
    {'I', "L"},
    {'O', "M"},
    {'P', "N"},

    /* Row 2. */
    {'A', "I"},
    {'S', "IH"},
    {'D', "OE"},
    {'F', "OO"},

    {'G', "NG"},
    {'H', "P"},
    {'J', "R"},
    {'K', "S"},
    {'L', "SH"},
    {';', "T"},

    /* Row 3. */
    {'Z', "OW"},
    {'X', "OY"},
    {'C', "UH"},
    {'V', "UU"},

    {'B', "TH"},
    {'N', "V"},
    {'M', "S"},
    {',', "SH"},
    {'.', "W"},
    {'/', "ZH"},
};

/* Draw menu of phonemes with `active_phoneme` highlighted. */
static void DrawFreePlayMenu(int active_phoneme) {
  mvprintw(5, 1, "Vowels");
  mvprintw(5, 31, "Consonants\n ");
  int x = 0;
  int i;
  for (i = 0; i < 39; ++i) {
    if (i == active_phoneme) {
      attron(A_BOLD | COLOR_PAIR(kColorHighlight));
    } else {
      attron(A_BOLD | COLOR_PAIR(kColorKey));
    }
    printw("[%c]%c", kFreePlayPhonemeTable[i].button,
           (i == active_phoneme) ? '*' : ' ');
    if (i != active_phoneme) {
      attroff(A_BOLD | COLOR_PAIR(kColorKey));
    }
    printw("%-2s", kFreePlayPhonemeTable[i].phoneme);
    if (i == active_phoneme) {
      attroff(A_BOLD | COLOR_PAIR(kColorHighlight));
    }
    if (++x % 10 == 0) {
      x = 0;
      printw("\n ");
      if (i == 9) {
        printw("       ");
        ++x;
      }
    } else {
      printw(" ");
    }

    if (x == 4) {
      printw("  ");  /* Extra space between vowels and consonants. */
    }
  }
}

static void FreePlayOnEnterState(TactophoneEngine* engine) {
  clear();
  mvprintw(0, 0, " FREE PLAY");
  TactophoneFormatTitle();

  mvprintw(2, 2, "This is free play mode. Use the indicated keys\n");
  printw("  to play tactile pattern for any phoneme.");
  DrawFreePlayMenu(-1);

  attron(A_BOLD | COLOR_PAIR(kColorKey));
  printw("\n\n  [Q] ");
  attroff(A_BOLD | COLOR_PAIR(kColorKey));
  printw("Back to main menu");

  TactophoneRefresh();
}

static void FreePlayOnKeyPress(TactophoneEngine* engine, int c) {
  if (c == 'Q') {  /* Quit to main menu. */
    TactophoneSetState(engine, &kTactophoneStateMainMenu);
  } else {
    int i;
    for (i = 0; i < 39; ++i) {
      if (c == kFreePlayPhonemeTable[i].button) {
        TactophonePlayPhonemes(engine, kFreePlayPhonemeTable[i].phoneme);
        DrawFreePlayMenu(i);
        TactophoneRefresh();
        break;
      }
    }
  }
}

static void FreePlayOnTick(TactophoneEngine* engine) {
  TactophoneVisualizeTactors(engine, 15, 45);
  TactophoneRefresh();
}

const TactophoneState kTactophoneStateFreePlay = {
    FreePlayOnEnterState,
    FreePlayOnKeyPress,
    FreePlayOnTick,
};
