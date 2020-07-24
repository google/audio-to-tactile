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

#include "audio/tactile/references/taps/tactophone_lesson.h"

#include <stdlib.h>
#include <string.h>

#include "audio/dsp/portable/logging.h"

static const char* kWhitespace = " \t\n\r\v";

/* Check whether [start, end) matches null-terminated string rhs. */
static int SubstringEquals(const char* start, const char* end,
                           const char* rhs) {
  const int length = strlen(rhs);
  return (end - start) == length && !memcmp(start, rhs, length);
}

/* Allocate and return a copy of [start, end) as a null-terminated string. */
static char* DuplicateSubstring(const char* start, const char* end) {
  const int length = end - start;
  char* s = (char*)malloc(length + 1);
  if (s != NULL) {
    memcpy(s, start, length);
    s[length] = '\0';
  }
  return s;
}

/* Heuristic for growing the capacity of dynamic arrays. */
static int ComputeNewCapacity(int new_size) {
  int new_capacity = new_size + new_size / 4; /* Grow by 1.25x factor. */
  const int kMinCapacity = 4; /* Start with a nonempty capacity of 4. */
  return (kMinCapacity >= new_capacity) ? kMinCapacity : new_capacity;
}

/* Parse a question definition like "lie;L,I pie;P,I sigh;S,I". */
static void ParseQuestion(const char* s, TactophoneQuestion* question) {
  int capacity = 0;

  while (*s) {
    /* Label starts after any preceding whitespace and ends at ';'. */
    const char* label_start = s + strspn(s, kWhitespace);
    const char* label_end = strchr(label_start, ';');
    if (label_end == NULL) {
      return;
    }
    /* Phonemes start after the ';' and end at any whitespace. */
    const char* phonemes_start = label_end + 1;
    const char* phonemes_end =
        phonemes_start + strcspn(phonemes_start, kWhitespace);

    /* Append a new TactophoneChoice to `questions->choices`. */
    ++question->num_choices;
    if (question->num_choices > capacity) {
      capacity = ComputeNewCapacity(question->num_choices);
      question->choices = (TactophoneChoice*)CHECK_NOTNULL(
          realloc(question->choices, capacity * sizeof(TactophoneChoice)));
    }

    TactophoneChoice* new_choice =
        question->choices + question->num_choices - 1;
    new_choice->label =
        CHECK_NOTNULL(DuplicateSubstring(label_start, label_end));
    new_choice->phonemes =
        CHECK_NOTNULL(DuplicateSubstring(phonemes_start, phonemes_end));

    s = phonemes_end;
  }
}

TactophoneLessonSet* TactophoneReadLessonSet(const char* filename) {
  FILE* file = fopen(filename, "rt");
  if (!file) {
    perror("Unable to open lessons file");
    return NULL;
  }
  TactophoneLessonSet* lesson_set =
      (TactophoneLessonSet*)CHECK_NOTNULL(malloc(sizeof(TactophoneLessonSet)));
  lesson_set->lessons = NULL;
  lesson_set->num_lessons = 0;

  int lessons_capacity = 0;
  int questions_capacity = 0;
  char line[256];  /* Max supported line length is sizeof(line) - 1. */

  while (fgets(line, sizeof(line), file) != NULL) {
    /* Parse line with the format "<item> <value>". */
    const char* item_start = line + strspn(line, kWhitespace);
    if (*item_start == '\0' || *item_start == '#') {
      continue; /* Skip blank lines and lines beginning with '#'. */
    }
    const char* item_end = item_start + strcspn(item_start, kWhitespace);
    const char* value = item_end + strspn(item_end, kWhitespace);

    if (SubstringEquals(item_start, item_end, "lesson")) {
      /* Append a new TactophoneLesson to `lesson_set->lessons`. */
      ++lesson_set->num_lessons;
      if (lesson_set->num_lessons > lessons_capacity) {
        lessons_capacity = ComputeNewCapacity(lesson_set->num_lessons);
        lesson_set->lessons = (TactophoneLesson*)CHECK_NOTNULL(realloc(
            lesson_set->lessons, lessons_capacity * sizeof(TactophoneLesson)));
      }

      TactophoneLesson* new_lesson =
          lesson_set->lessons + lesson_set->num_lessons - 1;

      /* Don't include newline in lesson name. */
      const char* value_end = value + strcspn(value, "\r\n");
      new_lesson->name = DuplicateSubstring(value, value_end);
      new_lesson->questions = NULL;
      new_lesson->num_questions = 0;
      questions_capacity = 0;
    } else if (SubstringEquals(item_start, item_end, "question")) {
      if (lesson_set->lessons) {
        TactophoneLesson* current_lesson =
            lesson_set->lessons + lesson_set->num_lessons - 1;
        /* Append a new TactophoneQuestion to `current_lesson->questions`. */
        ++current_lesson->num_questions;
        if (current_lesson->num_questions > questions_capacity) {
          questions_capacity =
              ComputeNewCapacity(current_lesson->num_questions);
          current_lesson->questions = (TactophoneQuestion*)CHECK_NOTNULL(
              realloc(current_lesson->questions,
                      questions_capacity * sizeof(TactophoneQuestion)));
        }

        TactophoneQuestion* new_question =
            current_lesson->questions + current_lesson->num_questions - 1;
        new_question->choices = NULL;
        new_question->num_choices = 0;
        ParseQuestion(value, new_question);
      }
    }
  }

  return lesson_set;
}

/* Free the `choices` array inside a TactophoneQuestion. */
static void TactophoneFreeQuestion(TactophoneQuestion* question) {
  int i;
  for (i = 0; i < question->num_choices; ++i) {
    free(question->choices[i].phonemes);
    free(question->choices[i].label);
  }
  free(question->choices);
}

/* Free `name` string and `questions` array inside a TactophoneLesson. */
static void TactophoneFreeLesson(TactophoneLesson* lesson) {
  int i;
  for (i = 0; i < lesson->num_questions; ++i) {
    TactophoneFreeQuestion(&lesson->questions[i]);
  }
  free(lesson->questions);
  free(lesson->name);
}

void TactophoneFreeLessonSet(TactophoneLessonSet* lesson_set) {
  if (lesson_set) {
    int i;
    for (i = 0; i < lesson_set->num_lessons; ++i) {
      TactophoneFreeLesson(&lesson_set->lessons[i]);
    }
    free(lesson_set->lessons);
    free(lesson_set);
  }
}
