/* Copyright 2021-2022 Google LLC
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
 * Get the build datestamp from __DATE__ as a uint32 constant.
 *
 * For instance when built on June 9, 2021, `DATESTAMP_UINT32` is 20210609.
 *
 * WARNING: DATESTAMP_UINT32 is incorrect when building with Bazel, since in
 * order to make builds hermetic, Bazel sets __DATE__ to "redacted".
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_DATESTAMP_H_
#define AUDIO_TO_TACTILE_SRC_DSP_DATESTAMP_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Datestamp for when the program was built as a uint32 constant. */
#define DATESTAMP_UINT32 DATESTR_TO_UINT32(__DATE__)

/* Converts a date string of the form "Jun 09 2021" to a uint32 20210609. This
 * is implemented as a macro so that an optimized build of
 *   DATESTR_TO_UINT32(__DATE__)
 * results in a compile-time constant.
 */
#define DATESTR_TO_UINT32(datestr) (                            \
  (uint32_t)((datestr)[ 7] - '0') * UINT32_C(10000000) +        \
  (uint32_t)((datestr)[ 8] - '0') * UINT32_C( 1000000) +        \
  (uint32_t)((datestr)[ 9] - '0') * UINT32_C(  100000) +        \
  (uint32_t)((datestr)[10] - '0') * UINT32_C(   10000) +        \
  (                                                             \
  /* Jan */ ((datestr)[0] == 'J' && (datestr)[1] == 'a') ?  1 : \
  /* Feb */ ((datestr)[0] == 'F'                       ) ?  2 : \
  /* Mar */ ((datestr)[0] == 'M' && (datestr)[2] == 'r') ?  3 : \
  /* Apr */ ((datestr)[0] == 'A' && (datestr)[1] == 'p') ?  4 : \
  /* May */ ((datestr)[0] == 'M'                       ) ?  5 : \
  /* Jun */ ((datestr)[0] == 'J' && (datestr)[2] == 'n') ?  6 : \
  /* Jul */ ((datestr)[0] == 'J'                       ) ?  7 : \
  /* Aug */ ((datestr)[0] == 'A'                       ) ?  8 : \
  /* Sep */ ((datestr)[0] == 'S'                       ) ?  9 : \
  /* Oct */ ((datestr)[0] == 'O'                       ) ? 10 : \
  /* Nov */ ((datestr)[0] == 'N'                       ) ? 11 : \
  /* Dec */                                                12   \
  ) * UINT32_C(100) +                                           \
  (((datestr)[4] <= '0') ? 0 :                                  \
      (uint32_t)((datestr)[4] - '0') * UINT32_C(10)) +          \
  (uint32_t)((datestr)[5] - '0')                                \
)

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_SRC_DSP_DATESTAMP_H_ */
