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

#include "tactile/hexagon_interpolation.h"

#include <math.h>
#include <stdlib.h>

#include "dsp/logging.h"
#include "dsp/math_constants.h"

static float Square(float x) { return x * x; }

/* Random float distributed uniformly in [0, 1]. */
static float RandUniform() { return (float)rand() / RAND_MAX; }

/* Check interpolation weights at the hexagon vertices. */
void TestInterpolationAtVertices() {
  puts("TestInterpolationAtVertices");
  int i;
  for (i = 0; i < 7; ++i) {
    /* Get the ith vertex. */
    float x;
    float y;
    if (i < 6) {
      x = sin(i * M_PI / 3);
      y = -cos(i * M_PI / 3);
    } else {
      x = 0.0f;
      y = 0.0f;
    }

    float weights[7];
    GetHexagonInterpolationWeights(x, y, weights);

    int n;
    for (n = 0; n < 7; ++n) {
      if (n == i) {
        CHECK(fabs(weights[n] - 1.0f) <= 1e-7f);
      } else {
        CHECK(fabs(weights[n]) <= 1e-7f);
      }
    }
  }
}

/* Test whether a point is inside the hexagonal neighborhood defined by the
 * first six sample positions.
 */
static int InsideHexagonNeighborhood(float x, float y) {
  float x0 = 0.0f;
  float y0 = -1.0f;
  int i;
  for (i = 1; i <= 6; ++i) {  /* Loop over edges of the hexagon. */
    float x1 = sin(i * M_PI / 3);
    float y1 = -cos(i * M_PI / 3);
    /* If (x,y) is to the right of the line segment from (x0,y0) to (x1,y1),
     * then the point is outside.
     */
    if ((y - y0) * (x1 - x0) - (x - x0) * (y1 - y0) < 0.0f) {
      return 0;
    }
    x0 = x1;
    y0 = y1;
  }
  return 1;
}

/* Weights are a convex combination if (x,y) is inside the hexagon. */
void TestConvex() {
  puts("TestConvex");
  int i;
  for (i = 0; i < 100; ++i) {
    const float x = 2.0f * RandUniform() - 1.0f;
    const float y = 2.0f * RandUniform() - 1.0f;

    float weights[7];
    GetHexagonInterpolationWeights(x, y, weights);

    int n;
    for (n = 0; n < 7; ++n) {
      CHECK(weights[n] >= 0.0f);  /* Weights should always be nonnegative. */
    }

    if (InsideHexagonNeighborhood(x, y)) {
      /* If inside the hexagon, the weights should also be convex. */
      float sum = 0.0f;
      for (n = 0; n < 7; ++n) {
        CHECK(weights[n] <= 1.0f + 1e-6f);
        sum += weights[n];
      }
      CHECK(fabs(sum - 1.0f) <= 1e-6f);
    }
  }
}

/* Interpolation weights computed for nearby points should be close. */
void TestContinuity() {
  puts("TestContinuity");
  int i;
  for (i = 0; i < 100; ++i) {
    const float x0 = 2.0f * RandUniform() - 1.0f;
    const float y0 = 2.0f * RandUniform() - 1.0f;
    const float x1 = x0 + 0.02f * (RandUniform() - 0.5f);
    const float y1 = y0 + 0.02f * (RandUniform() - 0.5f);

    float weights0[7];
    GetHexagonInterpolationWeights(x0, y0, weights0);
    float weights1[7];
    GetHexagonInterpolationWeights(x1, y1, weights1);

    float norm = 0.0f;
    int n;
    for (n = 0; n < 7; ++n) {
      norm += Square(weights0[n] - weights1[n]);
    }
    norm = sqrt(norm);
    CHECK(norm <= 0.02f);
  }
}

/* Test the HexagonNorm() function. */
void TestHexagonNorm() {
  puts("TestHexagonNorm");
  /* Check at the origin. */
  CHECK(HexagonNorm(0.0f, 0.0f) == 0.0f);

  int i;
  for (i = 0; i < 100; ++i) {
    const int n = rand() / (RAND_MAX / 6 + 1);

    const float x0 = sin(n * M_PI / 3); /* Get nth and (n+1)th vertices. */
    const float y0 = -cos(n * M_PI / 3);
    const float x1 = sin((n + 1) * M_PI / 3);
    const float y1 = -cos((n + 1) * M_PI / 3);
    /* Convex combination to get a random point on the hexagon's nth edge. */
    const float w = RandUniform();
    const float x = (1.0f - w) * x0 + w * x1;
    const float y = (1.0f - w) * y0 + w * y1;

    /* HexagonNorm is equal to 1.0 on the hexagon boundary. */
    CHECK(fabs(HexagonNorm(x, y) - 1.0f) <= 1e-6f);

    /* Absolute homogeneity: scaling by `r` changes the norm by factor `r`. */
    const float r = 2.0f * RandUniform();
    CHECK(fabs(HexagonNorm(r * x, r * y) - r) <= 1e-6f);
  }
}

/* Test that HexagonNorm() differs from Euclidean (L2) norm by less than 20%. */
void TestHexagonNormNearEuclideanNorm() {
  puts("TestHexagonNormNearEuclideanNorm");
  int i;
  for (i = 0; i < 100; ++i) {
    const float x = 2.0f * RandUniform() - 1.0f;
    const float y = 2.0f * RandUniform() - 1.0f;

    const float hexagon_norm = HexagonNorm(x, y);
    const float euclidean_norm = (float)sqrt(x * x + y * y);
    /* Difference is no more than +/-20%. */
    CHECK(fabs(hexagon_norm - euclidean_norm) / euclidean_norm < 0.2f);
  }
}

int main(int argc, char** argv) {
  srand(0);
  TestInterpolationAtVertices();
  TestConvex();
  TestContinuity();
  TestHexagonNorm();
  TestHexagonNormNearEuclideanNorm();

  puts("PASS");
  return EXIT_SUCCESS;
}

