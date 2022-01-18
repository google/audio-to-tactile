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

#include "phonetics/embed_vowel.h"

#include <math.h>

#include "dsp/fast_fun.h"
#include "phonetics/hexagon_interpolation.h"
#include "phonetics/nn_ops.h"
#include "phonetics/embed_vowel_params.h"

const EmbedVowelTarget kEmbedVowelTargets[8] = {
    {"aa", {0.00000f, -1.00000f}},
    {"uw", {0.86603f, -0.50000f}},
    {"ih", {0.86603f, 0.50000f}},
    {"iy", {0.00000f, 1.00000f}},
    {"eh", {-0.86603f, 0.50000f}},
    {"ae", {-0.86603f, -0.50000f}},
    {"uh", {0.00000f, 0.00000f}},
    {"er", {1.00000f, 0.00000f}},
};
const int kEmbedVowelNumTargets =
  sizeof(kEmbedVowelTargets) / sizeof(*kEmbedVowelTargets);
const int kEmbedVowelNumChannels = kNumChannels;

static float SquareDistance(const float* coord,
                            const EmbedVowelTarget* target) {
  const float diff_x = coord[0] - target->coord[0];
  const float diff_y = coord[1] - target->coord[1];
  return diff_x * diff_x + diff_y * diff_y;
}

int EmbedVowelClosestTarget(const float* coord) {
  int min_index = 0;
  float min_square_distance = SquareDistance(coord, &kEmbedVowelTargets[0]);
  int i;
  for (i = 1; i < kEmbedVowelNumTargets; ++i) {
    const float square_distance = SquareDistance(coord, &kEmbedVowelTargets[i]);
    if (square_distance < min_square_distance) {
      min_index = i;
      min_square_distance = square_distance;
    }
  }
  return min_index;
}

int EmbedVowelTargetByName(const char* target_name) {
  int i;
  for (i = 0; i < kEmbedVowelNumTargets; ++i) {
    if (!strcmp(kEmbedVowelTargets[i].name, target_name)) {
      return i;
    }
  }
  return -1;  /* `target_name` was not found. */
}

void EmbedVowel(const float* frame, float* coord) {
  float buffer1[kDense1Units];
  float buffer2[kDense2Units];

  /* First dense layer. */
  DenseReluLayer(kNumChannels, kDense1Units, frame,
                 kDense1Weights, kDense1Bias, buffer1);
  /* Second dense layer. */
  DenseReluLayer(kDense1Units, kDense2Units, buffer1,
                 kDense2Weights, kDense2Bias, buffer2);
  /* Third dense layer, bottleneck layer. */
  DenseLinearLayer(kDense2Units, kDense3Units, buffer2,
                   kDense3Weights, kDense3Bias, coord);

  const float radius = 1e-4f + HexagonNorm(coord[0], coord[1]);
  const float scale = FastTanh(radius) / radius;
  coord[0] *= scale;
  coord[1] *= scale;
}
