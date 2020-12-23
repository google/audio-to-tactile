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

#include "phonetics/classify_phoneme.h"

#include <stdlib.h>

#include "phonetics/classify_phoneme_params.h"
#include "phonetics/nn_ops.h"

/* Constant names prefixed with "kClassifyPhoneme" are exposed in the .h file,
 * while the shorter names like `kNumChannels` are internal to this files.
 */
const int kClassifyPhonemeNumFrames = kNumFrames;
const int kClassifyPhonemeNumChannels = kNumCarlChannels;
const char* kClassifyPhonemePhonemeNames[kClassifyPhonemeNumPhonemes] = {
    "sil", "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey",
    "ih",  "iy", "ow", "oy", "uh", "uw", "y",  "r",  "l",  "w",
    "hh",  "m",  "n",  "ng", "v",  "f",  "dh", "th", "b",  "p",
    "g",   "k",  "d",  "t",  "jh", "ch", "z",  "s",  "zh", "sh"};
const char* kClassifyPhonemeMannerNames[kClassifyPhonemeNumManners] = {
    "nasal", "stop", "affricate", "fricative", "approximant"};
const char* kClassifyPhonemePlaceNames[kClassifyPhonemeNumPlaces] = {
    "front", "middle", "back"};

/* Finds the index of the largest score. */
static int ScoreArgMax(const float* scores, int num_scores) {
  float max_value = scores[0];
  int max_index = 0;
  int i;
  for (i = 1; i < num_scores; ++i) {
    if (scores[i] > max_value) {
      max_value = scores[i];
      max_index = i;
    }
  }
  return max_index;
}

/* Produce a binary classification output score by running a 2-unit dense layer
 * on `in` and then applying softmax. The softmax values are simply `(1-score)`
 * and `score` in the binary case, so we return only the latter value.
 */
static float BinaryScore(int in_size,
                         const float* in,
                         const float* weights,
                         const float* bias) {
  float out[2];
  DenseLinearLayer(in_size, 2, in, weights, bias, out);
  Softmax(out, 2);
  return out[1];
}

void ClassifyPhoneme(const float* frames, ClassifyPhonemeLabels* labels,
                     ClassifyPhonemeScores* scores) {
  float buffer1[kDense1Units];
  float buffer2[kDense2Units];

  /* Run the common portion of the network. */
  DenseReluLayer(kInputUnits, kDense1Units, frames,
                 kDense1Weights, kDense1Bias, buffer1);
  DenseReluLayer(kDense1Units, kDense2Units, buffer1,
                 kDense2Weights, kDense2Bias, buffer2);
  /* We can reuse buffer1 for the output, since kDense3Units < kDense1Units. */
  DenseReluLayer(kDense2Units, kDense3Units, buffer2,
                 kDense3Weights, kDense3Bias, buffer1);

  /* If needed, reuse buffer2 for phonemes; kPhonemeUnits < kDense2Units. */
  float* phoneme_scores = (scores != NULL) ? scores->phoneme : buffer2;
  DenseLinearLayer(kDense3Units, kPhonemeUnits, buffer1,
                   kPhonemeWeights, kPhonemeBias, phoneme_scores);

  if (labels != NULL) {  /* Hard classification labels were requested. */
    labels->phoneme = ScoreArgMax(phoneme_scores, kPhonemeUnits);

    /* Category labels for manner, place, etc. are determined from the phoneme
     * label (and not by argmax of category scores) through kCategoryLookUp.
     * This ensures labels are consistent, e.g. phoneme 'n' is always a nasal.
     */
    const uint16_t categories = kCategoryLookUp[labels->phoneme];
    labels->manner = categories & 7;
    labels->place = (categories >> 3) & 3;
    labels->vad = (labels->phoneme != 0);
    labels->vowel = (categories >> 5) & 1;
    labels->diphthong = (categories >> 6) & 1;
    labels->lax_vowel = (categories >> 7) & 1;
    labels->voiced = (categories >> 8) & 1;
  }

  if (scores != NULL) {  /* Soft classification scores were requested. */
    Softmax(scores->phoneme, kPhonemeUnits);

    /* Manner classification output. */
    DenseLinearLayer(kPhonemeUnits, kMannerOutputUnits, scores->phoneme,
                     kMannerOutputWeights, kMannerOutputBias,
                     scores->manner);
    Softmax(scores->manner, kMannerOutputUnits);
    /* Place classification output. */
    DenseLinearLayer(kPhonemeUnits, kPlaceOutputUnits, scores->phoneme,
                     kPlaceOutputWeights, kPlaceOutputBias, scores->place);
    Softmax(scores->place, kPlaceOutputUnits);
    /* Voice activity detection score. */
    scores->vad = BinaryScore(kPhonemeUnits, scores->phoneme,
                              kVadOutputWeights, kVadOutputBias);
    /* Vowel / consonant score. */
    scores->vowel = BinaryScore(kPhonemeUnits, scores->phoneme,
                                kVowelOutputWeights, kVowelOutputBias);
    /* Monophthong / diphthong score. */
    scores->diphthong = BinaryScore(
        kPhonemeUnits, scores->phoneme,
        kDiphthongOutputWeights, kDiphthongOutputBias);
    /* Lax / tense vowel score. */
    scores->lax_vowel = BinaryScore(
        kPhonemeUnits, scores->phoneme,
        kLaxVowelOutputWeights, kLaxVowelOutputBias);
    /* Voiced / unvoiced score. */
    scores->voiced = BinaryScore(kPhonemeUnits, scores->phoneme,
                                 kVoicedOutputWeights, kVoicedOutputBias);
  }
}
