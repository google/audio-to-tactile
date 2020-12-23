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

#include "dsp/fft.h"

#include <stdio.h>
#include <string.h>
#include "dsp/phase32.h"

/* Size of the sine look up table in phase32. */
#define kTableSize (1 << kPhase32TableBits)
/* A quarter cycle of the table, used to look up cosines. */
#define kTableQuarterCycle (kTableSize / 4)
/* Table indexing mask. */
#define kTableMask (kTableSize - 1)

static void ForwardTransformOneGroup(int twiddle_stride, int quads_in_group,
                                     ComplexFloat* data);

static void ForwardLastStage(int num_groups, ComplexFloat* data);

static void InverseFirstStage(int num_quads, ComplexFloat* data);

static void InverseTransformOneGroup(int twiddle_stride, int quads_in_group,
                                     ComplexFloat* data);

static int CheckSupportedSize(int transform_size);

void FftForwardScrambledTransform(ComplexFloat* data, int transform_size) {
  if (!CheckSupportedSize(transform_size)) { return; }

  int twiddle_stride = (1 << kPhase32TableBits) / transform_size;
  int quads_in_group = transform_size / 4;

  /* We perform FFTs by radix-4 Cooley-Tukey. Generally, a radix-M Cooley-Tukey
   * stage decomposes a size-(M N) FFT into M size-N FFTs followed by N size-M
   * FFTs, combined intermediately with twiddle factors (aka roots of unity)
   *
   *   exp(-i 2 pi m n / (M N)), m = 0, 1, ..., M - 1 and n = 0, 1, ..., N - 1,
   *
   * or with negative sign in the exponent for the inverse transform.
   * Cooley-Tukey is applied recursively to decompose the smaller FFTs further.
   *
   * The jth stage (j = 1, 2, ..) of the forward transform uses M =
   * transform_size / 4^j and N = 4 (radix-4 decimation in frequency). The
   * inverse transform does the same but swapping roles of M and N (radix-4
   * decimation in time).
   *
   * NOTE: For an odd power of two transform size, we would perform one radix-2
   * stage before the loop.
   */
  while (quads_in_group >= 4) { /* Each iteration performs one radix-4 stage. */
    int offset;
    for (offset = 0; offset < transform_size; offset += 4 * quads_in_group) {
      /* The array is partitioned into groups of 4 * quads_in_groups elements.
       * Each iteration performs a radix-4 stage of transformation on one group.
       */
      ForwardTransformOneGroup(twiddle_stride, quads_in_group, data + offset);
    }
    twiddle_stride *= 4;
    quads_in_group /= 4;
  }

  /* Final radix-4 stage with transform_size / 4 groups and 1 quad per group. */
  ForwardLastStage(transform_size / 4, data);
}

/* Scrambles data into bit-reversed order. */
void FftScramble(ComplexFloat* data, int transform_size) {
  const int half_size = transform_size / 2;
  int i;
  int j = 0;
  for (i = 0; i < transform_size; ++i) {
    if (i < j) {
      ComplexFloat temp;
      /* Swap data[i] and data[j]. Each memcpy will compile to 64-bit mov. */
      memcpy(&temp, &data[i], sizeof(ComplexFloat));
      memcpy(&data[i], &data[j], sizeof(ComplexFloat));
      memcpy(&data[j], &temp, sizeof(ComplexFloat));
    }
    /* TODO: Consider optimizing this bit reversal. */
    int bit;
    for (bit = half_size; bit && j >= bit; bit >>= 1) {
      j -= bit;
    }
    j += bit;
  }
}

void FftInverseScrambledTransform(ComplexFloat* data, int transform_size) {
  if (!CheckSupportedSize(transform_size)) { return; }

  /* First radix-4 stage with transform_size / 4 groups and 1 quad per group. */
  InverseFirstStage(transform_size / 4, data);

  int twiddle_stride = 1 << (kPhase32TableBits - 4);
  int quads_in_group = 4;
  while (4 * quads_in_group <= transform_size) {
    int offset;
    for (offset = 0; offset < transform_size; offset += 4 * quads_in_group) {
      InverseTransformOneGroup(twiddle_stride, quads_in_group, data + offset);
    }
    quads_in_group *= 4;
    twiddle_stride /= 4;
  }

  /* NOTE: For an odd power of two transform size, we would perform one radix-2
   * stage after the loop.
   */
}

/* Computes one radix-4 forward FFT stage on one contiguous group of
 * 4 * quads_in_group elements. This function is the bottleneck computation for
 * larger transform sizes.
 */
static void ForwardTransformOneGroup(int twiddle_stride, int quads_in_group,
                                     ComplexFloat* data) {
  const int stride = quads_in_group;
  const int stride2 = stride * 2;
  const int stride3 = stride * 3;
  int twiddle1 = 0;
  int twiddle2 = 0;
  int twiddle3 = 0;

  /* Separately handle the first quad, exploiting that twiddle factors are 1. */
  float a1r = data[0].real - data[stride2].real;
  float a1i = data[0].imag - data[stride2].imag;
  float a2r = data[stride].real + data[stride3].real;
  float a2i = data[stride].imag + data[stride3].imag;
  float a3r = data[stride].real - data[stride3].real;
  float a3i = data[stride].imag - data[stride3].imag;
  data[0].real += data[stride2].real;
  data[0].imag += data[stride2].imag;
  data[stride].real = data[0].real - a2r;
  data[stride].imag = data[0].imag - a2i;
  data[0].real += a2r;
  data[0].imag += a2i;
  data[stride2].real = a1r + a3i;
  data[stride2].imag = a1i - a3r;
  data[stride3].real = a1r - a3i;
  data[stride3].imag = a1i + a3r;

  while (--quads_in_group) {
    twiddle1 += twiddle_stride;
    twiddle2 += 2 * twiddle_stride;
    twiddle3 += 3 * twiddle_stride;
    ++data;

    /* First level of butterflies. */
    a1r = data[0].real - data[stride2].real;
    a1i = data[0].imag - data[stride2].imag;
    a2r = data[stride].real + data[stride3].real;
    a2i = data[stride].imag + data[stride3].imag;
    a3r = data[stride].real - data[stride3].real;
    a3i = data[stride].imag - data[stride3].imag;
    data[0].real += data[stride2].real;
    data[0].imag += data[stride2].imag;

    /* Second level of butterflies. */
    float b1r = data[0].real - a2r;
    float b1i = data[0].imag - a2i;
    data[0].real += a2r;
    data[0].imag += a2i;
    float b2r = a1r + a3i;
    float b2i = a1i - a3r;
    float b3r = a1r - a3i;
    float b3i = a1i + a3r;

    /* Look up twiddle factors from the sine table. Corresponding cosines are
     * found through offsetting by kTableQuarterCycle.
     */
    float t1r = kPhase32SinTable[(twiddle1 + kTableQuarterCycle) & kTableMask];
    float t1i = kPhase32SinTable[twiddle1];
    float t2r = kPhase32SinTable[(twiddle2 + kTableQuarterCycle) & kTableMask];
    float t2i = kPhase32SinTable[twiddle2];
    float t3r = kPhase32SinTable[(twiddle3 + kTableQuarterCycle) & kTableMask];
    float t3i = kPhase32SinTable[twiddle3];

    /* Multiply by the complex conjugate of the twiddle factors. (In the
     * inverse transform, the twiddle factors are not conjugated.)
     */
    data[stride].real = t2r * b1r + t2i * b1i;
    data[stride].imag = t2r * b1i - t2i * b1r;
    data[stride2].real = t1r * b2r + t1i * b2i;
    data[stride2].imag = t1r * b2i - t1i * b2r;
    data[stride3].real = t3r * b3r + t3i * b3i;
    data[stride3].imag = t3r * b3i - t3i * b3r;
  }
}

static void ForwardLastStage(int num_groups, ComplexFloat* data) {
  do {
    /* Compute the size-4 FFT of data[n], n = 0, 1, 2, 3, where the result is in
     * bit-reversed (0, 2, 1, 3) order:
     *   [data[0]]   [ 1   1   1   1 ]   [data[0]]
     *   [data[1]] = [ 1  -1   1  -1 ] * [data[1]]
     *   [data[2]]   [ 1  -i  -1  +i ]   [data[2]]
     *   [data[3]]   [ 1  +i  -1  -i ]   [data[3]]
     */
    float a1r = data[0].real - data[2].real;
    float a1i = data[0].imag - data[2].imag;
    float a2r = data[1].real + data[3].real;
    float a2i = data[1].imag + data[3].imag;
    float a3r = data[1].real - data[3].real;
    float a3i = data[1].imag - data[3].imag;
    data[0].real += data[2].real;
    data[0].imag += data[2].imag;

    data[1].real = data[0].real - a2r;
    data[1].imag = data[0].imag - a2i;
    data[0].real += a2r;
    data[0].imag += a2i;

    data[2].real = a1r + a3i;
    data[2].imag = a1i - a3r;
    data[3].real = a1r - a3i;
    data[3].imag = a1i + a3r;
    data += 4;
  } while (--num_groups);
}

static void InverseFirstStage(int num_quads, ComplexFloat* data) {
  do {
    /* Compute the size-4 IFFT of data[n], n = 0, 1, 2, 3, where the input is in
     * bit-reversed (0, 2, 1, 3) order:
     *   [data[0]]   [ 1   1   1   1 ]   [data[0]]
     *   [data[1]] = [ 1  -1  +i  -i ] * [data[1]]
     *   [data[2]]   [ 1   1  -1  -1 ]   [data[2]]
     *   [data[3]]   [ 1  -1  -i  +i ]   [data[3]]
     */
    float a1r = data[0].real - data[1].real;
    float a1i = data[0].imag - data[1].imag;
    float a2r = data[2].real + data[3].real;
    float a2i = data[2].imag + data[3].imag;
    float a3r = data[2].real - data[3].real;
    float a3i = data[2].imag - data[3].imag;
    data[0].real += data[1].real;
    data[0].imag += data[1].imag;

    data[2].real = data[0].real - a2r;
    data[2].imag = data[0].imag - a2i;
    data[0].real += a2r;
    data[0].imag += a2i;

    data[1].real = a1r - a3i;
    data[1].imag = a1i + a3r;
    data[3].real = a1r + a3i;
    data[3].imag = a1i - a3r;
    data += 4;
  } while (--num_quads);
}

/* Computes one radix-4 inverse FFT stage on one contiguous group of
 * 4 * quads_in_group elements. This function is the bottleneck computation for
 * larger sizes of inverse transforms.
 */
static void InverseTransformOneGroup(int twiddle_stride, int quads_in_group,
                                     ComplexFloat* data) {
  const int stride = quads_in_group;
  const int stride2 = 2 * quads_in_group;
  const int stride3 = 3 * quads_in_group;
  int twiddle1 = 0;
  int twiddle2 = 0;
  int twiddle3 = 0;

  /* Separately handle the first quad, exploiting that twiddle factors are 1. */
  float b1r = data[0].real - data[stride].real;
  float b1i = data[0].imag - data[stride].imag;
  float b2r = data[stride2].real + data[stride3].real;
  float b2i = data[stride2].imag + data[stride3].imag;
  float b3r = data[stride2].real - data[stride3].real;
  float b3i = data[stride2].imag - data[stride3].imag;
  data[0].real += data[stride].real;
  data[0].imag += data[stride].imag;
  data[stride2].real = data[0].real - b2r;
  data[stride2].imag = data[0].imag - b2i;
  data[0].real += b2r;
  data[0].imag += b2i;
  data[stride].real = b1r - b3i;
  data[stride].imag = b1i + b3r;
  data[stride3].real = b1r + b3i;
  data[stride3].imag = b1i - b3r;

  while (--quads_in_group) {
    twiddle1 += twiddle_stride;
    twiddle2 += 2 * twiddle_stride;
    twiddle3 += 3 * twiddle_stride;
    ++data;

    /* Look up twiddle factors from the sine table. */
    float t1r = kPhase32SinTable[(twiddle1 + kTableQuarterCycle) & kTableMask];
    float t1i = kPhase32SinTable[twiddle1];
    float t2r = kPhase32SinTable[(twiddle2 + kTableQuarterCycle) & kTableMask];
    float t2i = kPhase32SinTable[twiddle2];
    float t3r = kPhase32SinTable[(twiddle3 + kTableQuarterCycle) & kTableMask];
    float t3i = kPhase32SinTable[twiddle3];

    /* Multiply by the twiddle factors. */
    float a1r = t2r * data[stride].real - t2i * data[stride].imag;
    float a1i = t2r * data[stride].imag + t2i * data[stride].real;
    float a2r = t1r * data[stride2].real - t1i * data[stride2].imag;
    float a2i = t1r * data[stride2].imag + t1i * data[stride2].real;
    float a3r = t3r * data[stride3].real - t3i * data[stride3].imag;
    float a3i = t3r * data[stride3].imag + t3i * data[stride3].real;

    /* First level of butterflies. */
    b1r = data[0].real - a1r;
    b1i = data[0].imag - a1i;
    data[0].real += a1r;
    data[0].imag += a1i;
    b2r = a2r + a3r;
    b2i = a2i + a3i;
    b3r = a2r - a3r;
    b3i = a2i - a3i;

    /* Second level of butterflies. */
    data[stride2].real = data[0].real - b2r;
    data[stride2].imag = data[0].imag - b2i;
    data[0].real += b2r;
    data[0].imag += b2i;
    /* Compute b1 +/- i b3. (Swap of the forward transform, b1 -/+ i b3.) */
    data[stride].real = b1r - b3i;
    data[stride].imag = b1i + b3r;
    data[stride3].real = b1r + b3i;
    data[stride3].imag = b1i - b3r;
  }
}

/* Returns 1 if transform_size is a supported size and 0 otherwise. The size
 * must be an integer power of 4 between 4 and kTableSize.
 */
static int CheckSupportedSize(int transform_size) {
  if (4 <= transform_size && transform_size <= kTableSize &&
        (transform_size & ((transform_size - 1) | 0x2AAA)) == 0) {
    return 1;
  }
  fprintf(stderr,
          "Error: FFT size must be a power of 4 between 4 and %d, got: %d.\n",
          kTableSize, transform_size);
  return 0;
}
