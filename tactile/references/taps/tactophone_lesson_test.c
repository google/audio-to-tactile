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

#include "tactile/references/taps/tactophone_lesson.h"

#include <string.h>

#include "dsp/logging.h"

/* Test reading a short lessons file. */
void TestBasic() {
  char filename[L_tmpnam];
  CHECK_NOTNULL(tmpnam(filename));
  { /* Write a short lessons file. */
    FILE* f = CHECK_NOTNULL(fopen(filename, "wt"));
    fputs(
        "# Test lesson file.\n\n"
        "lesson Alpha\n"
        "question chat;CH,AE,T hat;H,AE,T cat;K,AE,T\n"
        "question chop;CH,AH,P hop;H,AH,P cop;K,AH,P\n"
        "lesson Beta\n"
        "question bun;B,ER,N done;D,ER,N run;R,ER,N sun;S,ER,N\n"
        "question beam;B,EE,M deem;D,EE,M ream;R,EE,M\n",
        f);
    CHECK(fclose(f) == 0);
  }

  TactophoneLessonSet* lesson_set =
      CHECK_NOTNULL(TactophoneReadLessonSet(filename));

  CHECK(lesson_set->num_lessons == 2);
  CHECK(lesson_set->lessons != NULL);

  const TactophoneLesson* lesson = &lesson_set->lessons[0];
  CHECK(!strcmp(lesson->name, "Alpha"));
  CHECK(lesson->num_questions == 2);
  CHECK(lesson->questions != NULL);

  CHECK(lesson->questions[0].num_choices == 3);
  const TactophoneQuestion* q = &lesson->questions[1];
  CHECK(q->num_choices == 3);
  CHECK(!strcmp(q->choices[0].label, "chop"));
  CHECK(!strcmp(q->choices[0].phonemes, "CH,AH,P"));
  CHECK(!strcmp(q->choices[1].label, "hop"));
  CHECK(!strcmp(q->choices[1].phonemes, "H,AH,P"));
  CHECK(!strcmp(q->choices[2].label, "cop"));
  CHECK(!strcmp(q->choices[2].phonemes, "K,AH,P"));

  lesson = &lesson_set->lessons[1];
  CHECK(!strcmp(lesson->name, "Beta"));
  CHECK(lesson->num_questions == 2);
  CHECK(lesson->questions != NULL);
  CHECK(lesson->questions[0].num_choices == 4);
  CHECK(lesson->questions[1].num_choices == 3);

  TactophoneFreeLessonSet(lesson_set);
  remove(filename);
}

int main(int argc, char** argv) {
  TestBasic();

  puts("PASS");
  return EXIT_SUCCESS;
}
