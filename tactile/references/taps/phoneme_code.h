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
 * Code signals for Purdue's phonemic-based 24-channel tactile display.
 *
 * Tactile codes are presented on a sleeve with 24 tactors. Channels are
 * enumerated as Purdue does, ordered in a spiral beginning from the elbow:
 *
 *                   Top view
 *           -------------------------
 *          |  1   5   9  13  17  21  |
 *   Elbow  |       Dorsal side       |  Wrist   (Back of hand)
 *          |  2   6  10  14  18  22  |
 *           -------------------------
 *
 *                   Top view
 *           -------------------------
 *          |  4   8  12  16  20  24  |
 *   Elbow  |       Volar side        |  Wrist   (Palm of hand)
 *          |  3   7  11  15  19  21  |
 *           -------------------------
 *
 * Reference:
 *   Charlotte M. Reed, Hong Z. Tan, Zachary D. Perez, E. Courtenay Wilson,
 *   Frederico M. Severgnini, Jaehong Jung, Juan S. Martinez, Yang Jiao, Ali
 *   Israr, Frances Lau, Keith Klumb, Robert Turcott, Freddy Abnousi, "A
 *   Phonemic-Based Tactile Display for Speech Communication," IEEE
 *   Transactions on Haptics, 2018.
 */

#ifndef AUDIO_TO_TACTILE_TACTILE_REFERENCES_TAPS_PHONEME_CODE_H_
#define AUDIO_TO_TACTILE_TACTILE_REFERENCES_TAPS_PHONEME_CODE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* 24-channel tactile code signal to represent a phoneme. */
struct PhonemeCode {
  /* Name of the phoneme (example: "AE"). */
  const char* phoneme;
  /* Function that generates one 24-channel frame of the code signal at a
   * requested time, having signature
   *
   *   void fun(float t, float* frame);
   *
   * where time t is in units of seconds and should be between 0 and
   * `duration`. Output is written to frame[c], c = 0, ..., 23. The channels
   * are in Purdue's order described above.
   */
  void (* const fun)(float, float*);
  /* Duration of the code in seconds. */
  float duration;
};
typedef struct PhonemeCode PhonemeCode;

/* Table of all the phoneme codes. */
extern const PhonemeCode kPhonemeCodebook[];
/* Size of the table. */
extern const int kPhonemeCodebookSize;

/* Finds a PhonemeCode by name. To facilitate parsing, `name` is interpreted
 * case-insensitively up to the first non-alphanumeric char. For instance the
 * input "b,ih,r,d" results in the code for B. Returns null if not found.
 */
const PhonemeCode* PhonemeCodeByName(const char* name);

/* Generates phoneme tactile pattern, where `phonemes` is a comma-delimited
 * string of phonemes in Purdue's notation. Codes signals for multiple phonemes
 * are concatenated with a silence gap of `spacing` seconds between them. Zero
 * or even negative spacing is allowed, in which case overlaps are added.
 *
 * If `emphasized_phoneme` is nonnull, output corresponding to occurrences of
 * emphasized_phoneme have amplitude scaled by emphasis_gain.
 *
 * Waveform samples are allocated and returned, and should be free'd by the
 * caller. The number of frames is written to `*num_frames`.
 */
float* GeneratePhonemeSignal(const char* phonemes,
                             float spacing,
                             const char* emphasized_phoneme,
                             float emphasis_gain,
                             float sample_rate_hz,
                             int* num_frames);

/* Checks whether phonemes is a valid string of comma-delimited phonemes. */
int PhonemeStringIsValid(const char* phonemes);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_TACTILE_REFERENCES_TAPS_PHONEME_CODE_H_ */
