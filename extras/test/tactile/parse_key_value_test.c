/* Copyright 2021 Google LLC
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

#include "src/tactile/parse_key_value.h"

#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

static void TestKeyValue(const char* line,
                         const char* expected_key,
                         const char* expected_value,
                         int expected_indent) {
  printf("TestKeyValue(\"%s\")\n", line);
  char buffer[64];
  strcpy(buffer, line);

  ParsedKeyValue result;
  CHECK(ParseKeyValue(buffer, &result));

  if (expected_key) {
    CHECK(strcmp(result.key, expected_key) == 0);
    CHECK(strcmp(result.value, expected_value) == 0);
    CHECK(result.indent == expected_indent);
  } else {
    CHECK(result.key == NULL);
    CHECK(result.value == NULL);
    CHECK(result.indent == 0);
  }
}

static void TestInvalidKeyValue(void) {
  puts("TestInvalidKeyValue");
  char buffer[64];
  strcpy(buffer, "line without a colon");

  ParsedKeyValue result;
  CHECK(!ParseKeyValue(buffer, &result));
  CHECK(result.key == NULL);
  CHECK(result.value == NULL);
  CHECK(result.indent == 0);
}

int main(int argc, char** argv) {
  TestKeyValue("device_name: Slim", "device_name", "Slim", 0);
  TestKeyValue("tuning:", "tuning", "", 0);
  TestKeyValue("  gain :  23 ", "gain", "23", 2);
  TestKeyValue("    foo: 42", "foo", "42", 4);
  TestKeyValue("", NULL, NULL, 0);
  TestKeyValue("# This is a comment.", NULL, NULL, 0);

  TestInvalidKeyValue();

  puts("PASS");
  return EXIT_SUCCESS;
}
