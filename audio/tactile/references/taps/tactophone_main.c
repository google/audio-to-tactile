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
 * Tactophone phoneme training game.
 */

#include <stdio.h>
#include <stdlib.h>

#include "audio/dsp/portable/logging.h"
#include "audio/tactile/references/taps/tactophone.h"
#include "audio/tactile/references/taps/tactophone_states.h"
#include "audio/tactile/util.h"

int main(int argc, char** argv) {
  const char* lessons_file = "lessons.txt";
  const char* log_file = "tactophone.log";
  const char* channel_source_list =
    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24";
  const char* channel_gains_db_list = NULL;
  int output_device = -1;
  int i;

  for (i = 1; i < argc; ++i) { /* Parse flags. */
    if (StartsWith(argv[i], "--output=")) {
      output_device = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--lessons=")) {
      lessons_file = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--log=")) {
      log_file = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--channels=")) {
      channel_source_list = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--channel_gains_db=")) {
      channel_gains_db_list = strchr(argv[i], '=') + 1;
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      goto fail;
    }
  }

  if (lessons_file == NULL) {
    fprintf(stderr, "Error: Must specify --lessons\n");
    goto fail;
  }

  TactophoneParams params;
  params.lessons_file = lessons_file;
  params.log_file = log_file;
  params.output_device = output_device;
  params.channel_source_list = channel_source_list;
  params.channel_gains_db_list = channel_gains_db_list;
  params.initial_state = &kTactophoneStateMainMenu;
  Tactophone(&params);

  return EXIT_SUCCESS;

fail:
  return EXIT_SUCCESS;
}
