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

#include "audio/dsp/portable/math_constants.h"

#include <math.h>

#include "audio/dsp/portable/logging.h"

/* Test that M_E, etc. are compile-time constants. We check this in C by
 * attempting to use them in enum definitions. The M_SQRT2 check for instance
 * would fail with "error: expression is not an integer constant expression" if
 * it were implemented nonstatically as `#define M_SQRT2 sqrt(2)`.
 */
enum {
  CHECK_M_E_IS_STATICALLY_DEFINED = (int)M_E
};
enum {
  CHECK_M_LN2_IS_STATICALLY_DEFINED = (int)M_LN2
};
enum {
  CHECK_M_LN10_IS_STATICALLY_DEFINED = (int)M_LN10
};
enum {
  CHECK_M_PI_IS_STATICALLY_DEFINED = (int)M_PI
};
enum {
  CHECK_M_SQRT2_IS_STATICALLY_DEFINED = (int)M_SQRT2
};

/* Test that constants have the correct values. */
void TestConstants() {
  CHECK(fabs(M_E - exp(1.0)) < 1e-16);
  CHECK(fabs(M_LN2 - log(2.0)) < 1e-16);
  CHECK(fabs(M_LN10 - log(10.0)) < 1e-16);
  CHECK(fabs(M_PI - acos(-1.0)) < 1e-16);
  CHECK(fabs(M_SQRT2  - sqrt(2.0)) < 1e-16);
}

int main(int argc, char** argv) {
  TestConstants();

  puts("PASS");
  return EXIT_SUCCESS;
}
