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
 *
 *
 * Functions for converting samples between different data types.
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_CONVERT_SAMPLE_H_
#define AUDIO_TO_TACTILE_SRC_DSP_CONVERT_SAMPLE_H_

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Convert int16_t sample to float value in [-1, 1]. */
static float ConvertSampleInt16ToFloat(int16_t sample) {
  return sample / 32768.0f /* 2^15 */;
}

/* Convert int32_t sample to float value in [-1, 1]. */
static float ConvertSampleInt32ToFloat(int32_t sample) {
  return sample / 2147483648.0f /* 2^31 */;
}

/* Convert float sample in [-1, 1] to int16_t. */
static int16_t ConvertSampleFloatToInt16(float sample) {
  /* Scale by 2^15 and round. */
  float value = floor(32768.0f * sample + 0.5f);
  /* Clamp to int16_t range. Clang on x86-64 or ARM64 optimizes this into
   * efficient float min and max instructions. But ARM32 doesn't have these
   * instructions, and on any architecture, GCC prefers to branch instead.
   */
  if (INT16_MIN >= value) { value = INT16_MIN; }
  if (INT16_MAX <= value) { value = INT16_MAX; }
  return (int16_t)value;
}

/* Convert float sample in [-1, 1] to int32_t. */
static int32_t ConvertSampleFloatToInt32(float sample) {
  /* Scale by 2^31. It's not worth rounding in this case. Float has only 23
   * fractional bits, so round-off error the float value has previously
   * accumulated is likely much larger than the truncation error made here.
   */
  float value = 2147483648.0f * sample;
  /* We can't repeat the strategy as for int16 conversion since INT32_MAX is
   * not exactly representable as a float: `(float)INT32_MAX = 2147483648.0f`,
   * which is off by one and would overflow if casted to int32_t.
   */
  if (value <= INT32_MIN) {
    return INT32_MIN;
  } else if (value >= INT32_MAX) {
    return INT32_MAX;
  } else {
    return (int32_t)value;
  }
}

/* Convert float sample in [-1, 1] to integer value in [0, max_value]. This is
 * useful for instance for PWM output. Note that the output range is up to and
 * including `max_value`.
 */
static int ConvertSampleFloatTo0_MaxValue(float sample, int max_value) {
  if (-1.0f > sample) { sample = -1.0f; }
  if (1.0f < sample) { sample = 1.0f; }
  /* I verified that when called with a constant `max_value`, both GCC and Clang
   * will inline-expand this function and evaluate `scale` at compile time.
   */
  const float scale = 0.5f * max_value;
  float value = scale * sample + (scale + 0.5f /* for rounding */);
  return (int)value;
}

/* Same as above, but converting an array of samples at once. */
void ConvertSampleArrayInt16ToFloat(
    const int16_t* in, int num_samples, float* out);
void ConvertSampleArrayInt32ToFloat(
    const int32_t* in, int num_samples, float* out);
void ConvertSampleArrayFloatToInt16(
    const float* in, int num_samples, int16_t* out);
void ConvertSampleArrayFloatToInt32(
    const float* in, int num_samples, int32_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_DSP_CONVERT_SAMPLE_H_ */
