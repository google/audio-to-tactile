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
 * Small C library that performs vowel embedding inference.
 *
 * `EmbedVowel()` takes a CARL+PCEN frame as input (using CarlFrontend with
 * default parameters) and returns a 2-D vowel space coordinate as output.
 *
 * The embedding attempts to map monophthong vowels to distinct target points
 * in the 2-D space. The targets points have an angular arrangement with the
 * "uh" target at (0, 0):
 *           ____
 *          /    \
 *     ____/  iy  \____
 *    /    \      /    \
 *   / eh   \____/   ih \
 *   \      /    \      /
 *    \____/  uh  \____/ er
 *    /    \      /    \
 *   /      \____/      \
 *   \ ae   /    \   uw /
 *    \____/      \____/
 *         \  aa  /
 *          \____/
 *
 * The name and coordinate of each target is defined in the table
 * `kEmbedVowelTargets`.
 */


#ifndef AUDIO_TO_TACTILE_PHONETICS_EMBED_VOWEL_H_
#define AUDIO_TO_TACTILE_PHONETICS_EMBED_VOWEL_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Structure describing one target point. */
struct EmbedVowelTarget {
  const char* name;  /* Phone name (e.g. "aa"). */
  float coord[2];  /* 2-D coordinate in the range [-1, 1] x [-1, 1]. */
};
typedef struct EmbedVowelTarget EmbedVowelTarget;

/* Array describing the targets. */
extern const EmbedVowelTarget kEmbedVowelTargets[];
/* Number of targets in the kEmbedVowelTargets array. */
extern const int kEmbedVowelNumTargets;
/* Number of channels in a frame. */
extern const int kEmbedVowelNumChannels;

/* Performs vowel embedding inference. `frame` is an array of size
 * kEmbedVowelNumChannels, a frame from the CARL+PCEN frontend. `coord` should
 * be an array of size 2. The predicted 2-D vowel space coordinate is written to
 * (coord[0], coord[1]).
 */
void EmbedVowel(const float* frame, float coord[2]);

/* Returns index in `kEmbedVowelTargets` of the target closest to `coord`. */
int EmbedVowelClosestTarget(const float coord[2]);

/* Finds index in `kEmbedVowelTargets` of the target with `target_name`. Returns
 * -1 if the name is invalid.
 */
int EmbedVowelTargetByName(const char* target_name);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_PHONETICS_EMBED_VOWEL_H_ */

