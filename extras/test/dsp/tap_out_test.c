/* Copyright 2022 Google LLC
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

#include "src/dsp/tap_out.h"

#include <string.h>

#include "src/dsp/logging.h"

static const TapOutDescriptor kApple = {"Apple", "float", 2, {3, 8}};
static const TapOutDescriptor kBanana = {"Banana", "text", 1, {30}};
static const TapOutDescriptor kCherry = {"Chocolate cherry", "uint32", 1, {9}};

static int /*bool*/ CheckBytes(const uint8_t* actual, const uint8_t* expected,
                               int num_bytes) {
  int i;
  for (i = 0; i < num_bytes; ++i) {
    if (actual[i] != expected[i]) {
      fprintf(stderr, "Bytes differ at index %d: %d vs. %d\n",
              i, (int)actual[i], (int)expected[i]);
      return 0;
    }
  }
  return 1;
}

static void TestDescriptors(void) {
  puts("TestDescriptors()");

  TapOutClearDescriptors();
  CHECK(TapOutAddDescriptor(&kApple) == 1);
  CHECK(TapOutAddDescriptor(&kBanana) == 2);
  CHECK(TapOutAddDescriptor(&kCherry) == 3);

  CHECK(TapOutWriteDescriptors());
  static const uint8_t kExpected[2 + 20 * 3] = {
    0xfe, 3,
    1, /* kDescriptorA. */
    'A', 'p', 'p', 'l', 'e', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    9, 3, 8, 0,
    2, /* kDescriptorB. */
    'B', 'a', 'n', 'a', 'n', 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0,
    11, 30, 0, 0,
    3, /* kDescriptorC. */
    'C', 'h', 'o', 'c', 'o', 'l', 'a', 't', 'e', ' ', 'c', 'h', 'e', 'r', 'r',
    5, 9, 0, 0,
  };
  CHECK(g_tap_out_buffer_size == sizeof(kExpected));
  CHECK(CheckBytes(g_tap_out_buffer, kExpected, sizeof(kExpected)));
}

static void TestSlices(void) {
  puts("TestSlices()");

  TapOutClearDescriptors();
  TapOutToken token_a = TapOutAddDescriptor(&kApple);
  TapOutToken token_b = TapOutAddDescriptor(&kBanana);
  TapOutToken token_c = TapOutAddDescriptor(&kCherry);

  TapOutToken tokens[2];
  tokens[0] = token_c;
  tokens[1] = token_b;
  CHECK(!TapOutIsEnabled());
  CHECK(TapOutEnable(tokens, 2));
  CHECK(TapOutIsEnabled());
  CHECK(g_tap_out_buffer_size == 1 + 9 * 4 + 30);

  const TapOutSlice* slice;
  slice = TapOutGetSlice(token_c);
  CHECK(slice != NULL);
  CHECK(slice->data == g_tap_out_buffer + 1);
  CHECK(slice->size == 9 * 4);

  slice = TapOutGetSlice(token_b);
  CHECK(slice != NULL);
  CHECK(slice->data == g_tap_out_buffer + 1 + 9 * 4);
  CHECK(slice->size == 30);

  int x = 123;
  float y = 0.45f;
  TapOutTextPrint(token_b, "Test: %d, %g", x, y);
  CHECK(!strcmp(slice->data, "Test: 123, 0.45"));

  slice = TapOutGetSlice(token_a);
  CHECK(slice == NULL);
}

static void TestDescriptorNullName(void) {
  puts("TestDescriptorNullName");

  const TapOutDescriptor kDescriptorNullName = {NULL, "float", 1, {5}};
  CHECK(!TapOutAddDescriptor(&kDescriptorNullName));
}

static void TestDescriptorBadDType(void) {
  puts("TestDescriptorBadDType");

  const TapOutDescriptor kDescriptorBadDType = {"D", "?", 1, {5}};
  CHECK(!TapOutAddDescriptor(&kDescriptorBadDType));
}

static void TestDescriptorShapeTooBig(void) {
  puts("TestDescriptorShapeTooBig");

  const TapOutDescriptor kDescriptorShapeTooBig =
      {"E", "float", 1, {1 + kTapOutBufferCapacity / sizeof(float)}};
  CHECK(!TapOutAddDescriptor(&kDescriptorShapeTooBig));
}

static void TestBadToken(void) {
  puts("TestBadToken()");

  TapOutClearDescriptors();
  TapOutAddDescriptor(&kApple);
  TapOutAddDescriptor(&kBanana);

  TapOutToken tokens[1] = {kInvalidTapOutToken};
  CHECK(!TapOutEnable(tokens, 1));
  CHECK(!TapOutIsEnabled());
  CHECK(g_tap_out_buffer_size == 0);
  CHECK(TapOutGetSlice(kInvalidTapOutToken) == NULL);
}

static void PrintToStderr(const char* message) {
  fprintf(stderr, "Error: TapOut: %s\n", message);
}

int main(int argc, char** argv) {
  TapOutSetErrorFun(PrintToStderr);

  TestDescriptors();
  TestSlices();
  TestDescriptorNullName();
  TestDescriptorBadDType();
  TestDescriptorShapeTooBig();
  TestBadToken();

  puts("PASS");
  return EXIT_SUCCESS;
}
