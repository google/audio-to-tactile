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

#include "audio/dsp/portable/logging.h"

typedef struct Thing {
  int stuff;
} Thing;

static int things_made;

struct Thing* MakeThing() {
  ++things_made;
  return (Thing*) malloc(sizeof(Thing));
}

void DoesntEvaluateTwice() {
  things_made = 0;
  Thing* my_thing = CHECK_NOTNULL(MakeThing());
  CHECK(things_made == 1);
  free(my_thing);
}

int main(int argc, char** argv) {
  srand(0);
  DoesntEvaluateTwice();

  puts("PASS");
  return EXIT_SUCCESS;
}
