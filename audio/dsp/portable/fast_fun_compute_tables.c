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

#include "audio/dsp/portable/fast_fun_compute_tables.h"

#include <math.h>

#include "audio/dsp/portable/math_constants.h"

static const int kTableSize = 256;

/* Compute log2(1 + x). */
static double Log2Plus1(double x) { return log(1 + x) / M_LN2; }

/* Compute log2(1 + x) over 0 <= x < 1 for kFastFunLog2Table. */
void ComputeFastFunLog2Table(float* table) {
  int i;
  for (i = 0; i < kTableSize; ++i) {
    /* In FastLog2, the argument for this table is quantized by truncating. So
     * the ith entry is used for x between i/kTableSize and (i + 1)/kTableSize.
     * We compute the function value at the two endpoints and set the entry to
     * the average. This minimizes the max error to half their difference.
     */
    const double x0 = ((double)i) / kTableSize;
    const double x1 = ((double)(i + 1)) / kTableSize;
    table[i] = (float)((Log2Plus1(x0) + Log2Plus1(x1)) / 2);
  }
}

/* Compute 2^23 * (2^x - 1) for  0 <= x < 1. */
void ComputeFastFunExp2Table(int32_t* table) {
  int i;
  for (i = 0; i < kTableSize; ++i) {
    /* In FastExp2, the table argument gets quantized according to the current
     * floating-point rounding direction, which we don't try to control. So we
     * simply set the ith entry for the function computed at x = i/kTableSize.
     */
    const double x = ((double)i) / kTableSize;
    table[i] = (int32_t)((1 << 23) * (exp(M_LN2 * x) - 1.0) + 0.5);
  }
}
