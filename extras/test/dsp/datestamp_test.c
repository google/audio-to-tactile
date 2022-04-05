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
 */

#include "src/dsp/datestamp.h"

#include <inttypes.h>

#include "src/dsp/logging.h"

/* Test DATESTR_TO_UINT32 macro. */
static void TestDatestrToUint32(const char* datestr, uint32_t expected) {
  printf("TestDatestrToUint32(\"%s\")\n", datestr);

  const uint32_t actual = DATESTR_TO_UINT32(datestr);
  if (actual != expected) {
    fprintf(stderr, ("Check failed: expected DATESTR_TO_UINT32(%s) == "
                     "%" PRIu32 ", got: %" PRIu32 "\n"),
            datestr, expected, actual);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  TestDatestrToUint32("Jan 23 2021", UINT32_C(20210123));
  TestDatestrToUint32("Feb 01 1982", UINT32_C(19820201));
  TestDatestrToUint32("Mar 01 1982", UINT32_C(19820301));
  TestDatestrToUint32("Apr 01 1982", UINT32_C(19820401));
  TestDatestrToUint32("May 01 1982", UINT32_C(19820501));
  TestDatestrToUint32("Jun 01 1982", UINT32_C(19820601));
  TestDatestrToUint32("Jul 01 1982", UINT32_C(19820701));
  TestDatestrToUint32("Aug 01 1982", UINT32_C(19820801));
  TestDatestrToUint32("Sep 01 1982", UINT32_C(19820901));
  TestDatestrToUint32("Oct 01 1982", UINT32_C(19821001));
  TestDatestrToUint32("Nov 01 1982", UINT32_C(19821101));
  TestDatestrToUint32("Dec 01 1982", UINT32_C(19821201));

  TestDatestrToUint32("Jun  9 2021", UINT32_C(20210609));

  puts("PASS");
  return EXIT_SUCCESS;
}
