/* Copyright 2020-2021 Google LLC
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

#include "src/dsp/convert_sample.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/dsp/logging.h"

static double RandUniform() { return (double)rand() / RAND_MAX; }

static void TestConvertInt16ToFromFloat() {
  puts("TestConvertInt16ToFromFloat");

  CHECK(ConvertSampleInt16ToFloat(0) == 0);
  CHECK(ConvertSampleInt16ToFloat(-12) == -12.0f / 32768.0f);
  CHECK(ConvertSampleInt16ToFloat(8192) == 0.25f);
  CHECK(ConvertSampleInt16ToFloat(INT16_MIN) == -1.0f);
  CHECK(ConvertSampleInt16ToFloat(INT16_MAX) == 32767.0f / 32768.0f);

  CHECK(ConvertSampleFloatToInt16(0.0f) == 0);
  CHECK(ConvertSampleFloatToInt16(0.25f) == 8192);
  CHECK(ConvertSampleFloatToInt16(-1.0f) == INT16_MIN);
  CHECK(ConvertSampleFloatToInt16(1.0f) == INT16_MAX);
  CHECK(ConvertSampleFloatToInt16(-1000.0f) == INT16_MIN);
  CHECK(ConvertSampleFloatToInt16(1000.0f) == INT16_MAX);

  const int kNum = 1000;
  float* as_float = (float*)CHECK_NOTNULL(malloc(kNum * sizeof(float)));
  int16_t* as_int16 = (int16_t*)CHECK_NOTNULL(malloc(kNum * sizeof(int16_t)));
  int i;

  /* Convert int16 -> float -> int16. */
  for (i = 0; i < kNum; ++i) {
    as_int16[i] = (int16_t)(65534.0 * (RandUniform() - 0.5));
  }
  ConvertSampleArrayInt16ToFloat(as_int16, kNum, as_float);
  for (i = 0; i < kNum; ++i) {
    const int16_t recovered = ConvertSampleFloatToInt16(as_float[i]);
    CHECK(as_int16[i] == recovered); /* int16 value is preseved exactly. */
  }

  /* Convert float -> int16 -> float. */
  for (i = 0; i < kNum; ++i) {
    as_float[i] = 2.0f * RandUniform() - 1.0f;
  }
  ConvertSampleArrayFloatToInt16(as_float, kNum, as_int16);
  for (i = 0; i < kNum; ++i) {
    const float recovered = ConvertSampleInt16ToFloat(as_int16[i]);
    CHECK(fabs(as_float[i] - recovered) < 1.6e-5); /* Max error of ~1/2^16.*/
  }

  free(as_int16);
  free(as_float);
}

static void TestConvertInt32ToFromFloat() {
  puts("TestConvertInt32ToFromFloat");

  CHECK(ConvertSampleInt32ToFloat(0) == 0);
  CHECK(ConvertSampleInt32ToFloat(-25000) == -25000.0f / 2147483648.0f);
  CHECK(ConvertSampleInt32ToFloat(INT32_C(536870912)) == 0.25f);
  CHECK(ConvertSampleInt32ToFloat(INT32_MIN) == -1.0f);

  CHECK(ConvertSampleFloatToInt32(0.0f) == 0);
  CHECK(ConvertSampleFloatToInt32(0.25f) == INT32_C(536870912));
  CHECK(ConvertSampleFloatToInt32(-1.0f) == INT32_MIN);
  CHECK(ConvertSampleFloatToInt32(1.0f) == INT32_MAX);
  CHECK(ConvertSampleFloatToInt32(-1000.0f) == INT32_MIN);
  CHECK(ConvertSampleFloatToInt32(1000.0f) == INT32_MAX);

  const int kNum = 1000;
  float* as_float = (float*)CHECK_NOTNULL(malloc(kNum * sizeof(float)));
  int32_t* as_int32 = (int32_t*)CHECK_NOTNULL(malloc(kNum * sizeof(int32_t)));
  int i;

  /* Convert int32 -> float -> int32. */
  for (i = 0; i < kNum; ++i) {
    as_int32[i] = (int32_t)(4294967294.0 * (RandUniform() - 0.5));
  }
  ConvertSampleArrayInt32ToFloat(as_int32, kNum, as_float);
  for (i = 0; i < kNum; ++i) {
    const int32_t recovered = ConvertSampleFloatToInt32(as_float[i]);
    CHECK(fabs((double)as_int32[i] - (double)recovered) < 256);
  }

  /* Convert float -> int32 -> float. */
  for (i = 0; i < kNum; ++i) {
    as_float[i] = 2.0f * RandUniform() - 1.0f;
  }
  ConvertSampleArrayFloatToInt32(as_float, kNum, as_int32);
  for (i = 0; i < kNum; ++i) {
    const float recovered = ConvertSampleInt32ToFloat(as_int32[i]);
    CHECK(fabs(as_float[i] - recovered) < 1e-7);
  }

  free(as_int32);
  free(as_float);
}

static void TestConvertFloatTo0_MaxValue() {
  puts("TestConvertFloatTo0_MaxValue");

  CHECK(ConvertSampleFloatTo0_MaxValue(0, 512) == 256);
  CHECK(ConvertSampleFloatTo0_MaxValue(0.25f, 512) == 320);
  CHECK(ConvertSampleFloatTo0_MaxValue(-1.0f, 512) == 0);
  CHECK(ConvertSampleFloatTo0_MaxValue(1.0f, 512) == 512);
  CHECK(ConvertSampleFloatTo0_MaxValue(-1000.0f, 512) == 0);
  CHECK(ConvertSampleFloatTo0_MaxValue(1000.0f, 512) == 512);

  CHECK(ConvertSampleFloatTo0_MaxValue(0.25f, 100) == 63);
  CHECK(ConvertSampleFloatTo0_MaxValue(0.25f, 8192) == 5120);
}

int main(int argc, char** argv) {
  srand(0);
  TestConvertInt16ToFromFloat();
  TestConvertInt32ToFromFloat();
  TestConvertFloatTo0_MaxValue();

  puts("PASS");
  return EXIT_SUCCESS;
}
