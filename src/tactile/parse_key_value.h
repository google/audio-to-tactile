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
 *
 *
 * Function for parsing text data of the form "key: value".
 *
 * This library implements a simple text parsing function to extract the key
 * and value from a line of text of the form "key: value". By invoking this
 * function on each line of text, this is enough to implement a subset of the
 * NestedText format [https://nestedtext.org/en/latest/index.html]. This is
 * intended for parsing human-readable config files. An example file:
 *
 *   device_name: Slim, left wrist
 *   # This is a comment.
 *   tuning:
 *     input_gain: 127
 *     output_gain: 191
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_PARSE_KEY_VALUE_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_PARSE_KEY_VALUE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  /* Extracted key. If no key is present, then `key` and `value` are null. */
  char* key;
  /* Extracted value. */
  char* value;
  /* Number of indenting spaces before the key. */
  int indent;
} ParsedKeyValue;

/* `ParseKeyValue()` takes one text line as input and extracts the key and
 * value, if present. For the first line of above example, it produces key
 * "device_name" and value "Slim, left wrist". Returns 1 on success and 0 if
 * the line has invalid syntax.
 *
 * NOTE: When parsing is successful, the function modifies the string pointed to
 * by `line`, inserting intermediate nulls to cut the key and value substrings.
 * The `key` and `value` fields point into the `line` string.
 *
 * Example:
 *   ParsedKeyValue result;
 *   if (!ParseKeyValue(line, &result)) {
 *     // Error: Invalid syntax.
 *   } else if (result.key == NULL) {
 *     // Blank or comment line.
 *   } else {
 *     // Do something with `result`.
 *   }
 */
int /*bool*/ ParseKeyValue(char* line, ParsedKeyValue* result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_PARSE_KEY_VALUE_H_ */
