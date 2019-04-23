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
#include "audio/tactile/phoneme_code/tactophone.h"
#include "audio/tactile/phoneme_code/tactophone_states.h"
#include "audio/tactile/util.h"

int* DefaultChannelPermutation() {
  int* channel_permutation = (int*)CHECK_NOTNULL(malloc(24 * sizeof(int)));
  int c;
  for (c = 0; c < 24; ++c) {
    channel_permutation[c] = c;
  }
  return channel_permutation;
}

int CheckChannelPermutation(const int* channel_permutation, int length) {
  if (channel_permutation == NULL) {
    fprintf(stderr, "Error: --channels has invalid syntax\n");
    return 0;
  } else if (length != 24) {
    fprintf(stderr, "Error: --channels must have length 24, got %d\n", length);
    return 0;
  }

  int c;
  for (c = 0; c < 24; ++c) {
    if (!(1 <= channel_permutation[c] && channel_permutation[c] <= 24)) {
      fprintf(stderr, "Error: Channel %d is invalid\n", channel_permutation[c]);
      return 0;
    }
  }
  return 1;
}

int main(int argc, char** argv) {
  const char* lessons_file = "lessons.txt";
  const char* log_file = "tactophone.log";
  int* channel_permutation = DefaultChannelPermutation();
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
      free(channel_permutation);
      int length;
      channel_permutation = ParseListOfInts(strchr(argv[i], '=') + 1, &length);
      if (!CheckChannelPermutation(channel_permutation, length)) {
        goto fail;
      }

      int c;
      for (c = 0; c < 24; ++c) {
        --channel_permutation[c]; /* Change from 1-base to 0-base indices. */
      }
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
  params.channel_permutation = channel_permutation;
  params.initial_state = &kTactophoneStateMainMenu;
  Tactophone(&params);

  free(channel_permutation);
  return EXIT_SUCCESS;

fail:
  free(channel_permutation);
  return EXIT_SUCCESS;
}
