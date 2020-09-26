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

#include "tactile/references/taps/tactophone_engine.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/logging.h"

/* Set up an array to record a history of events. */
char* g_events[16];
int g_num_events = 0;

void AddEvent(const char* event, ...) {
  CHECK(g_num_events < sizeof(g_events) / sizeof(*g_events));
  g_events[g_num_events] = (char*)malloc(64);
  va_list args;
  va_start(args, event);
  vsprintf(g_events[g_num_events], event, args);
  va_end(args);
  ++g_num_events;
}

void FreeEvents() {
  int i;
  for (i = 0; i < g_num_events; ++i) {
    free(g_events[i]);
  }
  g_num_events = 0;
}

extern const TactophoneState kBar;

/* "Foo" state. */

static void FooOnEnterState(TactophoneEngine* engine) {
  AddEvent("Foo entered");
}

static void FooOnKeyPress(TactophoneEngine* engine, int c) {
  AddEvent("Foo key press %c", c);
  if (c == 'Q') {
    engine->keep_running = 0;
  } else if (c == 'B') {
    TactophoneSetState(engine, &kBar);
  }
}

static void FooOnTick(TactophoneEngine* engine) { AddEvent("Foo tick"); }

const TactophoneState kFoo = {
    FooOnEnterState,
    FooOnKeyPress,
    FooOnTick,
};

/* "Bar" state. */

static void BarOnEnterState(TactophoneEngine* engine) {
  AddEvent("Bar entered");
}

static void BarOnKeyPress(TactophoneEngine* engine, int c) {
  AddEvent("Bar key press %c", c);
  if (c == 'Q') {
    engine->keep_running = 0;
  }
}

static void BarOnTick(TactophoneEngine* engine) { AddEvent("Bar tick"); }

const TactophoneState kBar = {
    BarOnEnterState,
    BarOnKeyPress,
    BarOnTick,
};

/* Test event handling in running with a couple states. */
void TestEventHandling() {
  TactophoneEngine engine;
  TactophoneEngineInit(&engine);

  /* Start in Foo. */
  TactophoneSetState(&engine, &kFoo);
  /* Run one tick without key press. */
  CHECK(TactophoneEngineRun(&engine, -1));
  CHECK(engine.clock_ticks == 1);
  /* Press B key, causing state to change to Bar. */
  CHECK(TactophoneEngineRun(&engine, 'b'));
  CHECK(engine.clock_ticks == 2);
  CHECK(engine.state == &kBar);
  /* Run a couple ticks without key press. */
  CHECK(TactophoneEngineRun(&engine, -1));
  CHECK(TactophoneEngineRun(&engine, -1));
  CHECK(engine.clock_ticks == 4);
  /* Press Q key, causing engine to signal exit. */
  CHECK(!TactophoneEngineRun(&engine, 'q'));
  CHECK(engine.clock_ticks == 5);

  /* Check event history. */
  CHECK(g_num_events == 9);
  CHECK(!strcmp(g_events[0], "Foo entered"));
  CHECK(!strcmp(g_events[1], "Foo tick"));
  CHECK(!strcmp(g_events[2], "Foo key press B"));
  CHECK(!strcmp(g_events[3], "Bar entered"));
  CHECK(!strcmp(g_events[4], "Bar tick"));
  CHECK(!strcmp(g_events[5], "Bar tick"));
  CHECK(!strcmp(g_events[6], "Bar tick"));
  CHECK(!strcmp(g_events[7], "Bar key press Q"));
  CHECK(!strcmp(g_events[8], "Bar tick"));

  TactophoneEngineFree(&engine);
  FreeEvents();
}

/* Test lesson trial logic. */
void TestLessonTrial() {
  char lessons_file[L_tmpnam];
  CHECK_NOTNULL(tmpnam(lessons_file));
  { /* Write a short lessons file. */
    FILE* f = CHECK_NOTNULL(fopen(lessons_file, "wt"));
    fputs(
        "# Test lessons file.\n\n"
        "lesson Alpha\n"
        "question bow;B,OE hoe;H,OE low;L,OE mow;M,OE though;TH,OE\n"
        "question bay;B,AY hay;H,AY lay;L,AY may;M,AY they;TH,AY\n"
        "question bee;B,EE he;H,EE Li;L,EE me;M,EE thee;TH,EE\n",
        f);
    CHECK(fclose(f) == 0);
  }

  TactophoneEngine engine;
  TactophoneEngineInit(&engine);
  engine.lesson_set = CHECK_NOTNULL(TactophoneReadLessonSet(lessons_file));

  /* Set the "Alpha" lesson. */
  TactophoneSetLesson(&engine, 0);
  CHECK(!strcmp(CurrentLesson(&engine)->name, "Alpha"));
  /* Make a trial. */
  TactophoneNextTrial(&engine);
  int question = engine.current_question;
  /* Submit an incorrect answer. */
  int choice =
      (engine.correct_choice + 1) % CurrentQuestion(&engine)->num_choices;
  CHECK(!TactophoneSelectChoice(&engine, choice));
  CHECK(engine.trials_correct == 0);
  CHECK(engine.trials_total == 1);
  CHECK(engine.correct_in_a_row == 0);
  CHECK(engine.max_correct_in_a_row == 0);

  /* Make 5 more trials. Answer them all correctly. */
  int i;
  for (i = 0; i < 5; ++i) {
    TactophoneNextTrial(&engine);
    /* Current question should differ from previous question. */
    CHECK(question != engine.current_question);
    question = engine.current_question;

    CHECK(TactophoneSelectChoice(&engine, engine.correct_choice));
    CHECK(engine.trials_correct == i + 1);
    CHECK(engine.trials_total == i + 2);
    CHECK(engine.correct_in_a_row == i + 1);
    CHECK(engine.max_correct_in_a_row == i + 1);
  }

  /* Make one last trial, answered incorrectly. */
  TactophoneNextTrial(&engine);
  choice = (engine.correct_choice + 1) % CurrentQuestion(&engine)->num_choices;
  CHECK(!TactophoneSelectChoice(&engine, choice));
  CHECK(engine.trials_correct == 5);
  CHECK(engine.trials_total == 7);
  CHECK(engine.correct_in_a_row == 0);
  CHECK(engine.max_correct_in_a_row == 5);

  TactophoneEngineFree(&engine);
  remove(lessons_file);
}

int main(int argc, char** argv) {
  TestEventHandling();
  TestLessonTrial();

  puts("PASS");
  return EXIT_SUCCESS;
}
