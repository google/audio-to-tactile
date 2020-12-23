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
 *
 *
 * C library that performs phoneme classification.
 *
 * Implements a simple low-latency 40-way phoneme classifier, taking 5
 * consecutive CARL+PCEN frames as input, and returning classification scores
 * for silence, 15 vowels, 24 consonants. It also produces classification scores
 * for several phonemic features.
 *
 * Classifications from this network are not very reliable compared to state of
 * the art speech recoginition. Its advantage is computation cost and latency
 * are low, so that it may be used to give responsive classification in real
 * time mobile applications, potentially useful to drive tactile cues to
 * complement lip reading.
 */

#ifndef AUDIO_TO_TACTILE_SRC_PHONETICS_CLASSIFY_PHONEME_H_
#define AUDIO_TO_TACTILE_SRC_PHONETICS_CLASSIFY_PHONEME_H_

#ifdef __cplusplus
extern "C" {
#endif

#define kClassifyPhonemeNumPhonemes 40
#define kClassifyPhonemeNumManners 5
#define kClassifyPhonemeNumPlaces 3

/* Struct of hard classification labels returned by ClassifyPhoneme(). Labels
 * are represented as integer indices.
 */
typedef struct {
  /* Phoneme classification as an index into kClassifyPhonemePhonemeNames.
   * Corresponds to the index with the highest score.
   */
  int phoneme;
  /* Manner classification as an index into kClassifyPhonemeMannerNames. */
  int manner;
  /* Place classification as an index into kClassifyPhonemePlaceNames. */
  int place;

  /* The below fields are 0/1-valued binary classifications. */
  /* Voice activity detection (VAD).                            */
  /*bool*/ int vad;        /* 0 => no speech, 1 => speech.      */
  /* Consonant/vowel classification, supposing there is speech. */
  /*bool*/ int vowel;      /* 0 => consonant, 1 => vowel.       */
  /* Monophthong/diphthong classification, supposing a vowel.   */
  /*bool*/ int diphthong;  /* 0 => monophthong, 1 => diphthong. */
  /* Tense/lax classification, supposing a vowel.               */
  /*bool*/ int lax_vowel;  /* 0 => tense vowel, 1 => lax vowel. */
  /* Voicing classification, supposing there is speech.         */
  /*bool*/ int voiced;     /* 0 => unvoiced, 1 => voiced.       */
} ClassifyPhonemeLabels;

/* Struct of classification scores returned by ClassifyPhoneme(). Scores are
 * between 0.0 and 1.0, with higher score implying greater confidence.
 */
typedef struct {
  /* Fine-grained classification scores for each phoneme. The name of the
   * ith phoneme is given by kClassifyPhonemePhonemeNames[i]. Includes silence
   * "sil" as a possible class.
   */
  float phoneme[kClassifyPhonemeNumPhonemes];
  /* Manner scores, supposing the phoneme is a consonant. */
  float manner[kClassifyPhonemeNumManners];
  /* Place of articulation scores, supposing the phoneme is a consonant. */
  float place[kClassifyPhonemeNumPlaces];
  float vad;         /* Voice activity detection (VAD) score.           */
  float vowel;       /* Consonant/vowel score.                          */
  float diphthong;   /* Monophthong/diphthong score, supposing a vowel. */
  float lax_vowel;   /* Tense/lax score, supposing a vowel.             */
  float voiced;      /* Unvoice/voiced score.                           */
} ClassifyPhonemeScores;

/* Number of frames that the network takes as input. */
extern const int kClassifyPhonemeNumFrames;
/* Number of CARL channels in a frame. */
extern const int kClassifyPhonemeNumChannels;
/* Names of phonemes as ARPABET codes: "aa", "r", "jh", etc. */
extern const char* kClassifyPhonemePhonemeNames[kClassifyPhonemeNumPhonemes];
/* Names of manner categories. */
extern const char* kClassifyPhonemeMannerNames[kClassifyPhonemeNumManners];
/* Names of place categories. */
extern const char* kClassifyPhonemePlaceNames[kClassifyPhonemeNumPlaces];

/* Performs phoneme classification from consecutive CARL+PCEN frames. The
 * CarlFrontend must run with the parameters:
 *
 *   input_sample_rate_hz = 16000 Hz
 *   block_size = 128 samples (8ms)
 *   pcen_cross_channel_diffusivity = 60.0
 *
 * and otherwise default parameters. The `frames` arg is an array of the
 * kClassifyPhoneNumFrames most recent frames. `labels` is filled with
 * hard-decision classification labels and `scores` of classification scores for
 * the phoneme in the most recent frame. Either of `labels` or `scores` may be
 * NULL if that output isn't needed.
 *
 * For an example use, see the unit test classify_phoneme_test.c.
 */
void ClassifyPhoneme(const float* frames, ClassifyPhonemeLabels* labels,
                     ClassifyPhonemeScores* scores);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_PHONETICS_CLASSIFY_PHONEME_H_ */
