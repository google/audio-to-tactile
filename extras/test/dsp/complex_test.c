/* Copyright 2019, 2021 Google LLC
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

#include "src/dsp/complex.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

const int kNumTrials = 25;

static double RandUnif(void) { return (double) rand() / RAND_MAX; }

static void TestComplexDoubleBasic(void) {
  puts("TestComplexDoubleBasic");
  ComplexDouble z = ComplexDoubleMake(4.0, -3.0);
  CHECK(z.real == 4.0);
  CHECK(z.imag == -3.0);

  CHECK(ComplexDoubleConj(z).real == z.real);
  CHECK(ComplexDoubleConj(z).imag == -z.imag);

  CHECK(ComplexDoubleNeg(z).real == -z.real);
  CHECK(ComplexDoubleNeg(z).imag == -z.imag);

  CHECK(fabs(ComplexDoubleAbs(z) - 5.0) <= 1e-15);
  CHECK(fabs(ComplexDoubleAbs2(z) - 25.0) <= 1e-15);
}

static void TestComplexDoubleArithmetic(void) {
  puts("TestComplexDoubleArithmetic");
  int trial;
  for (trial = 0; trial < 10; ++trial) {
    ComplexDouble w = ComplexDoubleMake(RandUnif() - 0.5, RandUnif() - 0.5);
    ComplexDouble z = ComplexDoubleMake(RandUnif() - 0.5, RandUnif() - 0.5);

    ComplexDouble result = ComplexDoubleAdd(w, z);
    ComplexDouble expected;
    expected.real = w.real + z.real;
    expected.imag = w.imag + z.imag;
    CHECK(fabs(result.real - expected.real) <= 1e-15);
    CHECK(fabs(result.imag - expected.imag) <= 1e-15);

    result = ComplexDoubleMul(w, z);
    expected.real = w.real * z.real - w.imag * z.imag;
    expected.imag = w.real * z.imag + w.imag * z.real;
    CHECK(fabs(result.real - expected.real) <= 1e-15);
    CHECK(fabs(result.imag - expected.imag) <= 1e-15);

    result = ComplexDoubleMulReal(w, z.real);
    expected.real = w.real * z.real;
    expected.imag = w.imag * z.real;
    CHECK(fabs(result.real - expected.real) <= 1e-15);
    CHECK(fabs(result.imag - expected.imag) <= 1e-15);

    /* Check that subtraction inverts addition. */
    result = ComplexDoubleSub(ComplexDoubleAdd(w, z), z);
    CHECK(fabs(result.real - w.real) <= 1e-15);
    CHECK(fabs(result.imag - w.imag) <= 1e-15);

    /* Check that division inverts multiplication. */
    result = ComplexDoubleDiv(ComplexDoubleMul(w, z), z);
    CHECK(fabs(result.real - w.real) <= 1e-15);
    CHECK(fabs(result.imag - w.imag) <= 1e-15);

    result = ComplexDoubleMul(ComplexDoubleDiv(w, z), z);
    CHECK(fabs(result.real - w.real) <= 1e-15);
    CHECK(fabs(result.imag - w.imag) <= 1e-15);
  }
}

static void TestComplexDoublePolar(void) {
  puts("TestComplexDoublePolar");
  CHECK(fabs(ComplexDoubleAbs(ComplexDoubleMake(0.0, 0.0))) <= 1e-15);
  CHECK(fabs(ComplexDoubleAbs(ComplexDoubleMake(1.0, 1e-7))
             - (1.0 + 5e-15)) <= 1e-15);
  CHECK(fabs(ComplexDoubleAbs(ComplexDoubleMake(1e-7, 1.0))
             - (1.0 + 5e-15)) <= 1e-15);

  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(RandUnif() - 0.5, RandUnif() - 0.5);
    /* Convert cartesian to polar, then back to cartesian. */
    double abs_z = ComplexDoubleAbs(z);
    double arg_z = ComplexDoubleArg(z);
    CHECK(fabs(abs_z * cos(arg_z) - z.real) <= 1e-15);
    CHECK(fabs(abs_z * sin(arg_z) - z.imag) <= 1e-15);
  }
}

static void TestComplexDoubleSqrt(void) {
  puts("TestComplexDoubleSqrt");
  /* Check sqrt(z) at a few points. */
  ComplexDouble z = ComplexDoubleSqrt(ComplexDoubleMake(0.0, 0.0));
  CHECK(z.real == 0.0);
  CHECK(z.imag == 0.0);
  z = ComplexDoubleSqrt(ComplexDoubleMake(2.0, 0.0));
  CHECK(fabs(z.real - M_SQRT2) <= 1e-15);
  CHECK(z.imag == 0.0);
  z = ComplexDoubleSqrt(ComplexDoubleMake(-2.0, 0.0));
  CHECK(z.real == 0.0);
  CHECK(fabs(z.imag - M_SQRT2) <= 1e-15);
  z = ComplexDoubleSqrt(ComplexDoubleMake(1.0, 1e-8));
  CHECK(fabs(z.real - 1.0) <= 1e-15);
  CHECK(fabs(z.imag - 5e-9) <= 1e-15);
  z = ComplexDoubleSqrt(ComplexDoubleMake(1.0, -1e-8));
  CHECK(fabs(z.real - 1.0) <= 1e-15);
  CHECK(fabs(z.imag + 5e-9) <= 1e-15);

  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(RandUnif() - 0.5, RandUnif() - 0.5);
    ComplexDouble z2 = ComplexDoubleSquare(z);
    ComplexDouble result = ComplexDoubleSqrt(z2);

    /* Complex square root has two signs or "branches". ComplexDoubleSqrt
     * returns the principal branch, which recovers z for z.real >= 0. Otherwise
     * for z.real < 0, we need to flip the sign of the result.
     */
    if (z.real < 0) {
      result.real = -result.real;
      result.imag = -result.imag;
    }
    CHECK(fabs(result.real - z.real) <= 1e-15);
    CHECK(fabs(result.imag - z.imag) <= 1e-15);
  }
}

static void TestComplexDoubleLogExp(void) {
  puts("TestComplexDoubleLogExp");
  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(RandUnif() - 0.5, RandUnif() - 0.5);
    ComplexDouble result = ComplexDoubleExp(ComplexDoubleLog(z));

    CHECK(fabs(result.real - z.real) <= 1e-15);
    CHECK(fabs(result.imag - z.imag) <= 1e-15);
  }
}

static void TestComplexDoubleACoshCosh(void) {
  puts("TestComplexDoubleACoshCosh");
  ComplexDouble z = ComplexDoubleCosh(ComplexDoubleMake(0.0, 0.0));
  CHECK(fabs(z.real - 1.0) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);

  /* Check acosh on a pair of points straddling the branch cut. */
  z = ComplexDoubleACosh(ComplexDoubleMake(-2.0, 1e-15));
  CHECK(fabs(z.real - log(2 + sqrt(3))) <= 1e-14);
  CHECK(fabs(z.imag - M_PI) <= 1e-14);
  z = ComplexDoubleACosh(ComplexDoubleMake(-2.0, -1e-15));
  CHECK(fabs(z.real - log(2 + sqrt(3))) <= 1e-14);
  CHECK(fabs(z.imag + M_PI) <= 1e-14);

  /* Check that cosh(z)^2 - sinh(z)^2 == 1. */
  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(4 * RandUnif() - 2,
                                        4 * RandUnif() - 2);
    ComplexDouble result = ComplexDoubleSub(
        ComplexDoubleSquare(ComplexDoubleCosh(z)),
        ComplexDoubleSquare(ComplexDoubleSinh(z)));

    CHECK(fabs(result.real - 1.0) <= 1e-13);
    CHECK(fabs(result.imag) <= 1e-13);
  }

  /* Check that cosh(acosh(z)) == z. */
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(8 * RandUnif() - 4,
                                        8 * RandUnif() - 4);
    ComplexDouble result = ComplexDoubleCosh(ComplexDoubleACosh(z));

    CHECK(fabs(result.real - z.real) <= 1e-13);
    CHECK(fabs(result.imag - z.imag) <= 1e-13);
  }
}

static void TestComplexDoubleASinhSinh(void) {
  puts("TestComplexDoubleASinhSinh");
  ComplexDouble z = ComplexDoubleSinh(ComplexDoubleMake(0.0, 0.0));
  CHECK(fabs(z.real) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);

  /* Check asinh on a pair of points straddling the branch cut. */
  z = ComplexDoubleASinh(ComplexDoubleMake(1e-15, 2.0));
  CHECK(fabs(z.real - log(2 + sqrt(3))) <= 1e-14);
  CHECK(fabs(z.imag - M_PI / 2) <= 1e-14);
  z = ComplexDoubleASinh(ComplexDoubleMake(-1e-15, 2.0));
  CHECK(fabs(z.real + log(2 + sqrt(3))) <= 1e-14);
  CHECK(fabs(z.imag - M_PI / 2) <= 1e-14);

  /* Check that sinh(asinh(z)) == z. */
  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(8 * RandUnif() - 4,
                                        8 * RandUnif() - 4);
    ComplexDouble result = ComplexDoubleSinh(ComplexDoubleASinh(z));

    CHECK(fabs(result.real - z.real) <= 5e-13);
    CHECK(fabs(result.imag - z.imag) <= 5e-13);
  }
}

static void TestComplexDoubleACosCos(void) {
  puts("TestComplexDoubleACosCos");
  /* Check cos(z) at a few points. */
  ComplexDouble z = ComplexDoubleCos(ComplexDoubleMake(0.0, 0.0));
  CHECK(fabs(z.real - 1.0) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleCos(ComplexDoubleMake(M_PI / 2, 0.0));
  CHECK(fabs(z.real) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleCos(ComplexDoubleMake(M_PI, 0.0));
  CHECK(fabs(z.real + 1.0) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleCos(ComplexDoubleMake(-M_PI / 2, 0.0));
  CHECK(fabs(z.real) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleCos(ComplexDoubleMake(-M_PI, 0.0));
  CHECK(fabs(z.real + 1.0) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);

  /* Check acos on a pair of points straddling the branch cut. */
  z = ComplexDoubleACos(ComplexDoubleMake(2.0, 1e-15));
  CHECK(fabs(z.real) <= 1e-14);
  CHECK(fabs(z.imag + log(2 + sqrt(3))) <= 1e-14);
  z = ComplexDoubleACos(ComplexDoubleMake(2.0, -1e-15));
  CHECK(fabs(z.real) <= 1e-14);
  CHECK(fabs(z.imag - log(2 + sqrt(3))) <= 1e-14);

  /* Check that cos(z)^2 + sin(z)^2 == 1. */
  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(4 * RandUnif() - 2,
                                        4 * RandUnif() - 2);
    ComplexDouble result = ComplexDoubleAdd(
        ComplexDoubleSquare(ComplexDoubleCos(z)),
        ComplexDoubleSquare(ComplexDoubleSin(z)));

    CHECK(fabs(result.real - 1.0) <= 1e-13);
    CHECK(fabs(result.imag) <= 1e-13);
  }

  /* Check that cos(acos(z)) == z. */
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(8 * RandUnif() - 4,
                                        8 * RandUnif() - 4);
    ComplexDouble result = ComplexDoubleCos(ComplexDoubleACos(z));

    CHECK(fabs(result.real - z.real) <= 1e-13);
    CHECK(fabs(result.imag - z.imag) <= 1e-13);
  }

  /* Check that cos(acos(z)) == z on the real line. */
  int i;
  for (i = 0; i < 9; ++i) {
    double x = -1 + 0.25 * i;
    ComplexDouble acos_x = ComplexDoubleACos(ComplexDoubleMake(x, 0.0));

    CHECK(fabs(acos_x.real - acos(x)) <= 1e-15);
    CHECK(fabs(acos_x.imag) <= 1e-15);

    ComplexDouble recovered = ComplexDoubleCos(acos_x);
    CHECK(fabs(recovered.real - x) <= 1e-15);
    CHECK(fabs(recovered.imag) <= 1e-15);
  }
}

static void TestComplexDoubleASinSin(void) {
  puts("TestComplexDoubleASinSin");
  /* Check sin(z) at a few points. */
  ComplexDouble z = ComplexDoubleSin(ComplexDoubleMake(0.0, 0.0));
  CHECK(fabs(z.real) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleSin(ComplexDoubleMake(M_PI / 2, 0.0));
  CHECK(fabs(z.real - 1.0) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleSin(ComplexDoubleMake(M_PI, 0.0));
  CHECK(fabs(z.real) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleSin(ComplexDoubleMake(-M_PI / 2, 0.0));
  CHECK(fabs(z.real + 1.0) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);
  z = ComplexDoubleSin(ComplexDoubleMake(-M_PI, 0.0));
  CHECK(fabs(z.real) <= 1e-15);
  CHECK(fabs(z.imag) <= 1e-15);

  /* Check asin on a pair of points straddling the branch cut. */
  z = ComplexDoubleASin(ComplexDoubleMake(2.0, 1e-15));
  CHECK(fabs(z.real - M_PI / 2) <= 1e-14);
  CHECK(fabs(z.imag - log(2 + sqrt(3))) <= 1e-14);
  z = ComplexDoubleASin(ComplexDoubleMake(2.0, -1e-15));
  CHECK(fabs(z.real - M_PI / 2) <= 1e-14);
  CHECK(fabs(z.imag + log(2 + sqrt(3))) <= 1e-14);

  /* Check that sin(asin(z)) == z. */
  int trial;
  for (trial = 0; trial < kNumTrials; ++trial) {
    ComplexDouble z = ComplexDoubleMake(8 * RandUnif() - 4,
                                        8 * RandUnif() - 4);
    ComplexDouble result = ComplexDoubleSin(ComplexDoubleASin(z));

    CHECK(fabs(result.real - z.real) <= 1e-13);
    CHECK(fabs(result.imag - z.imag) <= 1e-13);
  }

  /* Check that sin(asin(z)) == z on the real line. */
  int i;
  for (i = 0; i < 9; ++i) {
    double x = -1 + 0.25 * i;
    ComplexDouble asin_x = ComplexDoubleASin(ComplexDoubleMake(x, 0.0));

    CHECK(fabs(asin_x.real - asin(x)) <= 1e-15);
    CHECK(fabs(asin_x.imag) <= 1e-15);

    ComplexDouble recovered = ComplexDoubleSin(asin_x);
    CHECK(fabs(recovered.real - x) <= 1e-15);
    CHECK(fabs(recovered.imag) <= 1e-15);
  }
}

static void TestComplexFloatBasic(void) {
  puts("TestComplexFloatBasic");
  ComplexFloat z = ComplexFloatMake(4.0f, -3.0f);
  CHECK(z.real == 4.0f);
  CHECK(z.imag == -3.0f);

  CHECK(ComplexFloatConj(z).real == z.real);
  CHECK(ComplexFloatConj(z).imag == -z.imag);

  CHECK(ComplexFloatNeg(z).real == -z.real);
  CHECK(ComplexFloatNeg(z).imag == -z.imag);

  CHECK(fabs(ComplexFloatAbs2(z) - 25.0f) <= 1e-15);
}

static void TestComplexFloatArithmetic(void) {
  puts("TestComplexFloatArithmetic");
  int trial;
  for (trial = 0; trial < 10; ++trial) {
    ComplexFloat w = ComplexFloatMake(RandUnif() - 0.5, RandUnif() - 0.5);
    ComplexFloat z = ComplexFloatMake(RandUnif() - 0.5, RandUnif() - 0.5);

    ComplexFloat result = ComplexFloatAdd(w, z);
    ComplexFloat expected;
    expected.real = w.real + z.real;
    expected.imag = w.imag + z.imag;
    CHECK(fabs(result.real - expected.real) <= 5e-7f);
    CHECK(fabs(result.imag - expected.imag) <= 5e-7f);

    result = ComplexFloatMul(w, z);
    expected.real = w.real * z.real - w.imag * z.imag;
    expected.imag = w.real * z.imag + w.imag * z.real;
    CHECK(fabs(result.real - expected.real) <= 5e-7f);
    CHECK(fabs(result.imag - expected.imag) <= 5e-7f);

    result = ComplexFloatMulReal(w, z.real);
    expected.real = w.real * z.real;
    expected.imag = w.imag * z.real;
    CHECK(fabs(result.real - expected.real) <= 5e-7f);
    CHECK(fabs(result.imag - expected.imag) <= 5e-7f);

    result = ComplexFloatSquare(w);
    expected = ComplexFloatMul(w, w);
    CHECK(fabs(result.real - expected.real) <= 5e-7f);
    CHECK(fabs(result.imag - expected.imag) <= 5e-7f);

    /* Check that subtraction inverts addition. */
    result = ComplexFloatSub(ComplexFloatAdd(w, z), z);
    CHECK(fabs(result.real - w.real) <= 5e-7f);
    CHECK(fabs(result.imag - w.imag) <= 5e-7f);

    /* Check that division inverts multiplication. */
    result = ComplexFloatDiv(ComplexFloatMul(w, z), z);
    CHECK(fabs(result.real - w.real) <= 5e-7f);
    CHECK(fabs(result.imag - w.imag) <= 5e-7f);

    result = ComplexFloatMul(ComplexFloatDiv(w, z), z);
    CHECK(fabs(result.real - w.real) <= 5e-7f);
    CHECK(fabs(result.imag - w.imag) <= 5e-7f);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestComplexDoubleBasic();
  TestComplexDoubleArithmetic();
  TestComplexDoublePolar();
  TestComplexDoubleSqrt();
  TestComplexDoubleLogExp();
  TestComplexDoubleACoshCosh();
  TestComplexDoubleASinhSinh();
  TestComplexDoubleACosCos();
  TestComplexDoubleASinSin();

  TestComplexFloatBasic();
  TestComplexFloatArithmetic();

  puts("PASS");
  return EXIT_SUCCESS;
}
