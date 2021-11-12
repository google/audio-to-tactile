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

#include "tactile/parse_key_value.h"

#include <ctype.h>
#include <string.h>

static const char* kWhitespaceChars = " \f\n\r\t\v";

/* Trim from the right of [s, s + length), returning the trimmed length. */
static int TrimRight(const char* s, int length) {
  while (length && isspace(s[length - 1])) {
    --length;
  }
  return length;
}

int /*bool*/ ParseKeyValue(char* line, ParsedKeyValue* result) {
  result->key = NULL;
  result->value = NULL;
  result->indent = 0;

  char* s = line;
  /* Skip leading whitespace. */
  s += strspn(s, kWhitespaceChars);

  /* Ignore empty or comment line. */
  if (*s == '\0' || *s == '#') {
    return 1;
  }

  /* Otherwise, `s` is the beginning of the key. */
  result->key = s;
  /* Find the ':', separating the key and value. */
  char* colon = strchr(s, ':');
  if (!colon) { /* Line has invalid syntax: expected a ':' character. */
    result->key = NULL;
    return 0;
  }
  const int key_length = TrimRight(result->key, colon - result->key);

  /* Find the value as the first non-space char after the colon. */
  s = colon + 1;
  s += strspn(s, kWhitespaceChars);

  result->value = s;
  char* s_end = s + strlen(s);
  const int value_length = TrimRight(result->value, s_end - result->value);

  /* Cut key and value substrings. This modifies the original string. */
  result->key[key_length] = '\0';
  result->value[value_length] = '\0';
  result->indent = result->key - line;

  return 1;
}
