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
  static const uint8_t kExpected[4 + 3 * 20] = {
    0xfe, kTapOutMessageDescriptors, 3 * 20, 3,
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
  CHECK(TapOutEnable(tokens, 2));
  CHECK(g_tap_out_buffer_size == 3 + 9 * 4 + 30);

  const TapOutSlice* slice;
  slice = TapOutGetSlice(token_c);
  CHECK(slice != NULL);
  CHECK(slice->data == g_tap_out_buffer + 3);
  CHECK(slice->size == 9 * 4);

  slice = TapOutGetSlice(token_b);
  CHECK(slice != NULL);
  CHECK(slice->data == g_tap_out_buffer + 3 + 9 * 4);
  CHECK(slice->size == 30);

  int x = 123;
  float y = 0.45f;
  TapOutTextPrint(token_b, "Test: %d, %g", x, y);
  CHECK(!strcmp((const char*)slice->data, "Test: 123, 0.45"));

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
  CHECK(g_tap_out_buffer_size == 0);
  CHECK(TapOutGetSlice(kInvalidTapOutToken) == NULL);
}

static int /*bool*/ g_tx_callback_called = 0;

static void TestCaptureTxExpectDescriptors(const char* data, int size) {
  static const uint8_t kExpected[4 + 2 * 20] = {
    kTapOutMarker, kTapOutMessageDescriptors, 2 * 20, 2,
    1, /* kDescriptorB. */
    'B', 'a', 'n', 'a', 'n', 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0,
    11, 30, 0, 0,
    2, /* kDescriptorC. */
    'C', 'h', 'o', 'c', 'o', 'l', 'a', 't', 'e', ' ', 'c', 'h', 'e', 'r', 'r',
    5, 9, 0, 0,
  };
  CHECK(size == sizeof(kExpected));
  CHECK(CheckBytes((const uint8_t*)data, kExpected, sizeof(kExpected)));
  g_tx_callback_called = 1;
}

static void TestCaptureTxExpectCapture(const char* data, int size) {
  static const uint8_t kExpected[3 + 30 + 9 * 4] = {
    kTapOutMarker, kTapOutMessageCapture, 30 + 9 * 4,
    /* Banana data. */
    'a', ' ', 't', 'e', 's', 't', ' ', 'm', 'e', 's', 's', 'a', 'g', 'e', 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /* Cherry data. */
    10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0, 40, 0, 0, 0, 50, 0, 0, 0,
    60, 0, 0, 0, 70, 0, 0, 0, 80, 0, 0, 0, 0x78, 0x56, 0x34, 0x12,
  };
  CHECK(size == sizeof(kExpected));
  CHECK(CheckBytes((const uint8_t*)data, kExpected, sizeof(kExpected)));
  g_tx_callback_called = 1;
}

static void TestCapture(void) {
  puts("TestCapture()");

  TapOutClearDescriptors();
  TapOutToken token_b = TapOutAddDescriptor(&kBanana);
  TapOutToken token_c = TapOutAddDescriptor(&kCherry);
  CHECK(token_b == 1);
  CHECK(token_c == 2);

  /* Simulate a GetDescriptors message. Descriptors should be sent back. */
  g_tx_callback_called = 0;
  TapOutSetTxFun(TestCaptureTxExpectDescriptors);

  static const uint8_t kMessageGetDescriptors[3] =
      {kTapOutMarker, kTapOutMessageGetDescriptors, 0};
  TapOutReceiveMessage((const char*)kMessageGetDescriptors,
                       sizeof(kMessageGetDescriptors));
  CHECK(g_tx_callback_called);
  CHECK(TapOutIsActive());

  /* Simulate a StartCapture message. */
  g_tx_callback_called = 0;
  static const uint8_t kMessageStartCapture[5] =
      {kTapOutMarker, kTapOutMessageStartCapture, 2, 1, 2};
  TapOutReceiveMessage((const char*)kMessageStartCapture,
                       sizeof(kMessageStartCapture));
  CHECK(!g_tx_callback_called);
  CHECK(TapOutIsActive());

  /* Fill the slices with test data. */
  const TapOutSlice* slice = TapOutGetSlice(token_c);
  CHECK(slice);
  static const uint32_t kCherryTestData[9] =
      {10, 20, 30, 40, 50, 60, 70, 80, UINT32_C(0x12345678)};
  memcpy(slice->data, kCherryTestData, slice->size);

  TapOutTextPrint(token_b, "a test message");
  CHECK(!g_tx_callback_called);

  /* Call TapOutFinishedCaptureBuffer(), which should send the data. */
  TapOutSetTxFun(TestCaptureTxExpectCapture);
  TapOutFinishedCaptureBuffer();
  CHECK(g_tx_callback_called);
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
  TestCapture();

  puts("PASS");
  return EXIT_SUCCESS;
}
