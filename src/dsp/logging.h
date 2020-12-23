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
 * Implements macros LOG_ERROR, CHECK, and CHECK_NOTNULL, which are useful for
 * error handling and unit tests written in C.
 *
 * The LOG_ERROR macro accepts a printf-style format string that is written to
 * stderr. Note that this minimalist macro does not write file and line prefix
 * or append a newline. For example
 *   LOG_ERROR("Error!\n");
 *   LOG_ERROR("Mismatch: %d vs. %d\n", x, y);
 *
 * The statement
 *   CHECK(condition);
 * checks that "condition" is true, and if not, prints a failure message
 * including file and line number and exits the program.
 *
 * CHECK_NOTNULL checks that its argument is nonnull and returns it, for example
 *   void* p = CHECK_NOTNULL(malloc(...));
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_LOGGING_H_
#define AUDIO_TO_TACTILE_SRC_DSP_LOGGING_H_

#include <stdlib.h>
#include <stdio.h>

#define LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)

static void _fatal_error(const char* message) {
  LOG_ERROR("%s", message);
  exit(EXIT_FAILURE);
}

#ifdef __cplusplus
/* Define this function outside of the C compilation because it includes
 * templates.
 * C++ implementation: use templating to accept different pointer types.
 */
template <typename T>
T _check_notnull(const char* message, T ptr) {
  if (ptr == nullptr) {
    _fatal_error(message);
  }
  return ptr;
}

extern "C" {

#else
/* C implementation: use implicit casts between pointer types. */
static void* _check_notnull(const char* message, void* ptr) {
  if (ptr == NULL) {
    _fatal_error(message);
  }
  return ptr;
}
#endif /* __cplusplus */

#define AS_STRING(x) AS_STRING_INTERNAL(x)
#define AS_STRING_INTERNAL(x) #x

#ifdef CHECK
#error CHECK is already defined.
#endif

#define CHECK(condition) \
  if (!(condition)) { \
     _fatal_error(__FILE__ ":" AS_STRING(__LINE__) ": Check failed: " \
                       #condition "\n"); \
  }

#define CHECK_NOTNULL(ptr) \
  _check_notnull(__FILE__ ":" AS_STRING(__LINE__) ": Pointer is null\n", ptr)

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_SRC_DSP_LOGGING_H_ */
