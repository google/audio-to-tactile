/* Copyright 2020 Google LLC
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

#include "dsp/convert_sample.h"

void ConvertSampleArrayInt16ToFloat(
    const int16_t* in, int num_samples, float* out) {
  int i;
  for (i = 0; i < num_samples; ++i) {
    out[i] = ConvertSampleInt16ToFloat(in[i]);
  }
}

void ConvertSampleArrayInt32ToFloat(
    const int32_t* in, int num_samples, float* out) {
  int i;
  for (i = 0; i < num_samples; ++i) {
    out[i] = ConvertSampleInt32ToFloat(in[i]);
  }
}

void ConvertSampleArrayFloatToInt16(
    const float* in, int num_samples, int16_t* out) {
  int i;
  for (i = 0; i < num_samples; ++i) {
    out[i] = ConvertSampleFloatToInt16(in[i]);
  }
}

void ConvertSampleArrayFloatToInt32(
    const float* in, int num_samples, int32_t* out) {
  int i;
  for (i = 0; i < num_samples; ++i) {
    out[i] = ConvertSampleFloatToInt32(in[i]);
  }
}
