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
 *
 *
 * Interpolation on a hexagonal lattice.
 */

#ifndef AUDIO_TACTILE_HEXAGON_INTERPOLATION_H_
#define AUDIO_TACTILE_HEXAGON_INTERPOLATION_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Gets piecewise trilinear interpolation weights on the regular hexagon:
 *        ___
 *       /   \
 *   ,--(  3  )--.
 *  / 4  \___/  2 \
 *  \    /   \    /
 *   )--(  6  )--(     sample n = (sin(n pi/3), -cos(n pi/3)), n = 0, ..., 5,
 *  /    \___/    \    sample 6 = (0, 0).
 *  \ 5  /   \  1 /
 *   `--(  0  )--'
 *       \___/
 *
 * `weights` is filled with the sample weights. Samples are as labeled above.
 *
 * The first six samples define a hexagonal neighborhood. If (x,y) is within
 * this neighborhood, the weights are a convex combination. If (x,y) is outside,
 * some weights may exceed 1, but all weights are still nonnegative.
 */
void GetHexagonInterpolationWeights(float x, float y, float weights[7]);

/* Computes a 2D norm or distance to the origin for the hexagon. The "unit ball"
 * set {(x,y) : HexagonNorm(x,y) == 1.0} is the hexagon with vertices
 *
 *   (sin(n pi/3), -cos(n pi/3)), n = 0, ..., 5.
 *
 * With this function, any point (x, y) in R^2 is mapped inside the hexagon by:
 *
 *   r = HexagonNorm(x, y)
 *   x *= tanh(r) / r
 *   y *= tanh(r) / r
 *
 * This is useful to constrain the vowel space embedding to the hexagon.
 */
float HexagonNorm(float x, float y);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_HEXAGON_INTERPOLATION_H_ */

