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
 * Lesson datastructures and parsing for phoneme code training.
 */

#ifndef AUDIO_TACTILE_REFERENCES_TAPS_TACTOPHONE_LESSON_H_
#define AUDIO_TACTILE_REFERENCES_TAPS_TACTOPHONE_LESSON_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Training works in trials of playing a tactile pattern and asking the
 * participant to identify it from one of several choices. Each choice in a
 * trial is a `TactophoneChoice`.
 *
 * The phoneme string is a comma-delimited list of phonemes in Purdue's
 * notation (example: "H,EH,L,OU" for hello).
 */
struct TactophoneChoice {
  char* label;    /* Label for display in the UI. */
  char* phonemes; /* Phoneme string for the tactile pattern. */
};
typedef struct TactophoneChoice TactophoneChoice;

/* A TactophoneQuestion is an array of TactophoneChoices. Trials are made by
 * randomly choosing a choice's phoneme string to play. Questions may have any
 * number of choices and may vary question to question. 4 to 9 choices is
 * suggested.
 * NOTE: The max supported line length is 255 chars, including the newline char.
 */
struct TactophoneQuestion {
  TactophoneChoice* choices;
  int num_choices;
};
typedef struct TactophoneQuestion TactophoneQuestion;

/* A TactophoneLesson is a collection of related TactophoneQuestions. A
 * lesson might for instance focus on a handful of consonants.
 */
struct TactophoneLesson {
  char* name;
  TactophoneQuestion* questions;
  int num_questions;
};
typedef struct TactophoneLesson TactophoneLesson;

/* A TactophoneLessonSet is a collection of TactophoneLessons. This can be
 * used represent a full training curriculum.
 */
struct TactophoneLessonSet {
  TactophoneLesson* lessons;
  int num_lessons;
};
typedef struct TactophoneLessonSet TactophoneLessonSet;

/* Reads a training lessons file in a text format. The caller must call
 * TactophoneFreeLessonSet on the returned pointer when done. On failure, the
 * function prints an error message and returns NULL.
 *
 * An example lesson file, defining two lessons:
 *
 *   # Lines starting with '#' are comments.
 *
 *   # Define a lesson called "Consonants 1: B, D, R, S, T" with 3 questions.
 *   lesson Consonants 1: B, D, R, S, T
 *   question bail;B,EH,L dale;D,EH,L rail;R,EH,L sail;S,EH,L tail;T,EH,L
 *   question bore;B,AW,R door;D,AW,R roar;R,AW,R sore;S,AW,R tore;T,AW,R
 *   question bun;B,ER,N done;D,ER,N run;R,ER,N sun;S,ER,N ton;T,ER,N
 *
 *   # Define a lesson with 2 questions.
 *   lesson Vowels 1: AE, AH, AY, EE, IH
 *   question bat;B,AE,T bot;B,AH,T bait;B,AY,T beet;B,EE,T bit;B,IH,T
 *   question cap;K,AE,P cop;K,AH,P cape;K,AY,P keep;K,EE,P kip;K,IH,P
 *
 * A line `lesson <name>` defines a new lesson and its name. A line starting
 * with `question` defines a question under the current lesson. The syntax of
 * a question is a space-delimited list of choices in the form
 *
 *   question <label1>;<phonemes1> <label2>;<phonemes2> ...
 */
TactophoneLessonSet* TactophoneReadLessonSet(const char* filename);

/* Free TactophoneLessonSet dynamic memory. */
void TactophoneFreeLessonSet(TactophoneLessonSet* lesson_set);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_REFERENCES_TAPS_TACTOPHONE_LESSON_H_ */
