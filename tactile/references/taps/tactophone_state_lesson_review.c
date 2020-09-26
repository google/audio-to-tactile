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
 * Lesson review.
 */

#include "tactile/references/taps/tactophone_states.h"

static void LessonReviewOnEnterState(TactophoneEngine* engine) {}

/* Heuristic for how many trials to make in a lesson. */
static int NumTrials(const TactophoneLesson* lesson) {
  int num = (int)(1.6f * lesson->num_questions);
  if (num < 5) {
    num = 5;
  }
  if (num > 20) {
    num = 20;
  }
  return num;
}

/* The player can play and compare tactile signals for the different choices.
 * After the player goes through enough trials (as determined by the NumTrials
 * heuristic), the lesson ends.
 */
static void LessonReviewOnKeyPress(TactophoneEngine* engine, int c) {
  if (c == 'Q') {  /* Quit lesson. */
    TactophoneSetState(engine, &kTactophoneStateMainMenu);
  } else if (c == '\n' || c == ' ') {  /* Continue to next trial. */
    if (engine->trials_total >= NumTrials(CurrentLesson(engine))) {
      TactophoneSetState(engine, &kTactophoneStateLessonDone);
    } else {
      TactophoneSetState(engine, &kTactophoneStateLessonTrial);
    }
  } else {
    const char* p = strchr(kMenuButtons, c);
    if (p != NULL) {  /* Play choice. */
      const TactophoneQuestion* q = CurrentQuestion(engine);
      int choice = p - kMenuButtons;
      if (0 <= choice && choice < q->num_choices) {
        TactophonePlayPhonemes(engine, q->choices[choice].phonemes);
      }
    }
  }
}

static void LessonReviewOnTick(TactophoneEngine* engine) {
  TactophoneVisualizeTactors(engine, 23, 45);
  TactophoneRefresh();
}

const TactophoneState kTactophoneStateLessonReview = {
    LessonReviewOnEnterState,
    LessonReviewOnKeyPress,
    LessonReviewOnTick,
};
