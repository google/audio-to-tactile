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

#include "phonetics/hexagon_interpolation.h"

#include <math.h>

/* The constants sin(pi/6) [= 1/2] and sec(pi/6) [= 2/sqrt(3)]. */
#define kSinPi_6  0.5f
#define kSecPi_6  1.15470054f

static float Relu(float x) { return x > 0.0f ? x : 0.0f; }

void GetHexagonInterpolationWeights(float x, float y, float weights[7]) {
  x *= kSecPi_6;
  const float u = -kSinPi_6 * x + y;
  const float v = -kSinPi_6 * x - y;

  /* Partition the hex neighborhood into 6 triangles based on signs of x, u, v:
   *
   *      3  |  2     Tri 0: x > 0, v > 0    ``````|``````          |
   *   -._   |   _.-  Tri 1: u < 0, v < 0    ` u > 0 ```.-    -._   |    .
   *   4  "-.|.-"  1  Tri 2: x > 0, u > 0    `````.|.-"       ```"-.|. "
   *     _.-"|"-._    Tri 3: x < 0, v < 0    ``_.-"|  .       ```. "|"-._
   *   -"    |    "-  Tri 4: u > 0, v > 0    -"    |    "     ` v > 0 ```"-
   *      5  |  0     Tri 5: x < 0, u < 0          |          ``````|``````
   */
  if (x > 0.0f) {  /* Right half of the neighborhood. */
    if (u > 0.0f) {  /* Triangle 2. */
      weights[0] = 0.0f;
      weights[1] = 0.0f;
      weights[2] = x;
      weights[3] = u;
      weights[6] = Relu(1.0f - x - u);
    } else if (v > 0.0f) {  /* Triangle 0. */
      weights[0] = v;
      weights[1] = x;
      weights[2] = 0.0f;
      weights[3] = 0.0f;
      weights[6] = Relu(1.0f - x - v);
    } else {  /* Triangle 1. */
      weights[0] = 0.0f;
      weights[1] = -u;
      weights[2] = -v;
      weights[3] = 0.0f;
      weights[6] = Relu(1.0f + u + v);
    }
    weights[4] = 0.0f;
    weights[5] = 0.0f;
  } else {  /* Left half of the neighborhood. */
    if (u > 0.0f) {
      if (v > 0.0f) {  /* Triangle 4. */
        weights[0] = 0.0f;
        weights[3] = 0.0f;
        weights[4] = u;
        weights[5] = v;
        weights[6] = Relu(1.0f - u - v);
      } else {  /* Triangle 3. */
        weights[0] = 0.0f;
        weights[3] = -v;
        weights[4] = -x;
        weights[5] = 0.0f;
        weights[6] = Relu(1.0f + x + v);
      }
    } else {  /* Triangle 5. */
      weights[0] = -u;
      weights[3] = 0.0f;
      weights[4] = 0.0f;
      weights[5] = -x;
      weights[6] = Relu(1.0f + x + u);
    }
    weights[1] = 0.0f;
    weights[2] = 0.0f;
  }
}

float HexagonNorm(float x, float y) {
  x = kSecPi_6 * (float)fabs(x);
  y = fabs(y);
  const float v = kSinPi_6 * x + y;
  return (x > v) ? x : v; /* Get the max of x and v. */
}
