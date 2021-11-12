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

#include "src/dsp/fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"
#include "src/dsp/complex.h"

static const float kTol = 1e-4f;

/* Fills random complex values. */
static void FillRandomValues(int size, ComplexFloat* output) {
  int i;
  for (i = 0; i < size; ++i) {
    output[i].real = rand() / (0.5f * RAND_MAX) - 1;
    output[i].imag = rand() / (0.5f * RAND_MAX) - 1;
  }
}

/* Check the size-4 transform of {1, 2, 3, 4} against golden data. */
static void TestForwardTransformSize4(void) {
  puts("TestForwardTransformSize4");
  const int kTransformSize = 4;
  ComplexFloat data[4] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
  static const ComplexFloat expected[4] =  /* Computed with numpy. */
      {{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}, {-2.0, -2.0}};

  FftForwardScrambledTransform(data, kTransformSize);

  FftUnscramble(data, kTransformSize);
  int k;
  for (k = 0; k < kTransformSize; ++k) {
    CHECK(fabs(data[k].real - expected[k].real) <= kTol);
    CHECK(fabs(data[k].imag - expected[k].imag) <= kTol);
  }
}

static void TestScrambling(int transform_size) {
  printf("TestScrambling(%d)\n", transform_size);
  ComplexFloat* data =
      CHECK_NOTNULL(malloc(sizeof(ComplexFloat) * transform_size));
  int n;
  for (n = 0; n < transform_size; ++n) {
    data[n] = ComplexFloatMake(n, 0.0f);
  }

  FftScramble(data, transform_size);

  /* Spot check a few entries. */
  CHECK(data[0].real == 0);                      /* 0000 -> 0000. */
  CHECK(data[1].real == transform_size / 2);     /* 0001 -> 1000. */
  CHECK(data[2].real == transform_size / 4);     /* 0010 -> 0100. */
  CHECK(data[3].real == 3 * transform_size / 4); /* 0011 -> 1100. */

  FftUnscramble(data, transform_size);

  for (n = 0; n < transform_size; ++n) { /* Original data is recovered. */
    CHECK(data[n].real == n);
    CHECK(data[n].imag == 0.0f);
  }

  free(data);
}

/* Fill the Dirichlet kernel [http://en.wikipedia.org/wiki/Dirichlet_kernel]. */
static void FillDirichletKernel(int radius, int size, ComplexFloat* output) {
  const float numerator_factor = (2.0 * M_PI * (radius + 0.5)) / size;
  const float denominator_factor = M_PI / size;
  int i;
  output[0].real = (2.0 * radius + 1.0) / size;
  output[0].imag = 0.0f;
  for (i = 1; i < size; ++i) {
    output[i].real = sin(numerator_factor * i) /
        (size * sin(denominator_factor * i));
    output[i].imag = 0.0f;
  }
}

/* Check the forward FFT of a Dirichlet kernel, which has closed-form transform
 * equal to 1 for k <= radius or k >= transform_size - radius and zero othewise.
 */
static void TestForwardTransformOfDirichletKernel(int transform_size) {
  printf("TestForwardTransformOfDirichletKernel(%d)\n", transform_size);
  ComplexFloat* data =
      CHECK_NOTNULL(malloc(sizeof(ComplexFloat) * transform_size));
  const int radius = (int)(0.3f * transform_size + 0.5f);
  FillDirichletKernel(radius, transform_size, data);

  FftForwardScrambledTransform(data, transform_size);

  FftUnscramble(data, transform_size);
  int k;
  for (k = 0; k < transform_size; ++k) {
    ComplexFloat expected;
    expected.real = (k <= radius || k >= transform_size - radius) ? 1.0f : 0.0f;
    expected.imag = 0.0;
    CHECK(fabs(data[k].real - expected.real) <= kTol);
    CHECK(fabs(data[k].imag - expected.imag) <= kTol);
  }

  free(data);
}

/* Similar to the previous test, check the inverse FFT of a Dirichlet kernel.
 * Note that the expected inverse transform is the same as for the forward
 * transform because the input sequence is symmetric about index 0.
 */
static void TestInverseTransformOfDirichletKernel(int transform_size) {
  printf("TestInverseTransformOfDirichletKernel(%d)\n", transform_size);
  ComplexFloat* data =
      CHECK_NOTNULL(malloc(sizeof(ComplexFloat) * transform_size));
  const int radius = (int)(0.3f * transform_size + 0.5f);
  FillDirichletKernel(radius, transform_size, data);
  FftScramble(data, transform_size);

  FftInverseScrambledTransform(data, transform_size);

  int n;
  for (n = 0; n < transform_size; ++n) {
    ComplexFloat expected;
    expected.real = (n <= radius || n >= transform_size - radius) ? 1.0f : 0.0f;
    expected.imag = 0.0f;
    CHECK(fabs(data[n].real - expected.real) <= kTol);
    CHECK(fabs(data[n].imag - expected.imag) <= kTol);
  }

  free(data);
}

/* Perform periodic convolution using scrambled FFTs, and check that it matches
 * direct time-domain convolution.
 */
static void TestFftBasedConvolution(int transform_size) {
  printf("TestFftBasedConvolution(%d)\n", transform_size);
  const int num_bytes = sizeof(ComplexFloat) * transform_size;
  ComplexFloat* signal = CHECK_NOTNULL(malloc(num_bytes));
  ComplexFloat* kernel = CHECK_NOTNULL(malloc(num_bytes));
  ComplexFloat* expected_convolution = CHECK_NOTNULL(malloc(num_bytes));
  ComplexFloat* convolution = CHECK_NOTNULL(malloc(num_bytes));

  FillRandomValues(transform_size, signal);
  FillRandomValues(transform_size, kernel);

  int n;
  for (n = 0; n < transform_size; ++n) {
    ComplexFloat sum = {0.0f, 0.0f};
    int m;
    int n_minus_m;
    for (m = 0, n_minus_m = n; m < transform_size; ++m) {
      sum = ComplexFloatAdd(sum, ComplexFloatMul(kernel[m], signal[n_minus_m]));
      --n_minus_m;
      if (n_minus_m < 0) {
        n_minus_m += transform_size;
      }
    }
    expected_convolution[n] = sum;
  }

  FftForwardScrambledTransform(signal, transform_size);
  FftForwardScrambledTransform(kernel, transform_size);

  int k;
  for (k = 0; k < transform_size; ++k) {
    convolution[k] = ComplexFloatMulReal(ComplexFloatMul(kernel[k], signal[k]),
                                         1.0f / transform_size);
  }
  FftInverseScrambledTransform(convolution, transform_size);

  for (n = 0; n < transform_size; ++n) {
    CHECK(fabs(convolution[n].real - expected_convolution[n].real) <= kTol);
    CHECK(fabs(convolution[n].imag - expected_convolution[n].imag) <= kTol);
  }

  free(convolution);
  free(expected_convolution);
  free(kernel);
  free(signal);
}

/* Check that MicroFftForwardScrambledTransform followed by
 * MicroFftInverseScrambledTransform and normalization recovers the original.
 */
static void TestRoundTrips(int transform_size) {
  printf("TestRoundTrips(%d)\n", transform_size);
  const int kNumTrials = 5;
  const int num_bytes = sizeof(ComplexFloat) * transform_size;
  ComplexFloat* data = CHECK_NOTNULL(malloc(num_bytes));
  ComplexFloat* original = CHECK_NOTNULL(malloc(num_bytes));

  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    FillRandomValues(transform_size, data);
    memcpy(original, data, num_bytes);

    FftForwardScrambledTransform(data, transform_size);
    FftInverseScrambledTransform(data, transform_size);

    int n;
    for (n = 0; n < transform_size; ++n) {
      data[n].real /= transform_size;
      data[n].imag /= transform_size;
    }

    for (n = 0; n < transform_size; ++n) {
      CHECK(fabs(data[n].real - original[n].real) <= kTol);
      CHECK(fabs(data[n].imag - original[n].imag) <= kTol);
    }
  }

  free(original);
  free(data);
}

/* Checks that attempting an unsupport transform size has no effect. */
static void TestUnsupportedSize(int transform_size) {
  printf("TestUnsupportedSize(%d)\n", transform_size);
  const int num_bytes = sizeof(ComplexFloat) * transform_size;
  ComplexFloat* data = CHECK_NOTNULL(malloc(num_bytes));
  ComplexFloat* original = CHECK_NOTNULL(malloc(num_bytes));
  FillRandomValues(transform_size, data);
  memcpy(original, data, sizeof(ComplexFloat) * transform_size);

  FftForwardScrambledTransform(data, transform_size);

  int n;
  for (n = 0; n < transform_size; ++n) { /* data is unchanged. */
    CHECK(data[n].real == original[n].real);
    CHECK(data[n].imag == original[n].imag);
  }

  FftInverseScrambledTransform(data, transform_size);

  for (n = 0; n < transform_size; ++n) { /* data is unchanged. */
    CHECK(data[n].real == original[n].real);
    CHECK(data[n].imag == original[n].imag);
  }

  free(original);
  free(data);
}

int main(int argc, char** argv) {
  srand(0);

  TestForwardTransformSize4();

  int transform_size;
  for (transform_size = 4; transform_size <= 1024; transform_size *= 4) {
    TestScrambling(transform_size);
    TestForwardTransformOfDirichletKernel(transform_size);
    TestInverseTransformOfDirichletKernel(transform_size);
    TestFftBasedConvolution(transform_size);
    TestRoundTrips(transform_size);
  }

  TestUnsupportedSize(3);
  TestUnsupportedSize(25);
  TestUnsupportedSize(32);

  puts("PASS");
  return EXIT_SUCCESS;
}
