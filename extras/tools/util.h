/* Copyright 2019, 2022 Google LLC
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
 * Miscellaneous utilities for tactile processing.
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_TOOLS_UTIL_H_
#define AUDIO_TO_TACTILE_EXTRAS_TOOLS_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Case-insensitive string comparison (according to current C locale). */
int StringEqualIgnoreCase(const char* s1, const char* s2);

/* Case-insensitive find substring. */
const char* FindSubstringIgnoreCase(const char* s, const char* substring);

/* Returns 1 if string `s` starts with `prefix`, 0 otherwise. */
int StartsWith(const char* s, const char* prefix);

/* Case-insensitive version of StartsWith. */
int StartsWithIgnoreCase(const char* s, const char* prefix);

/* Returns 1 if string `s` ends with `suffix`, 0 otherwise. */
int EndsWith(const char* s, const char* suffix);

/* Parses a comma-delimited list of ints. An array of the parsed ints is
 * allocated and returned, and should be freed by the caller. The number of ints
 * is written to `*length`. Returns NULL on failure.
 */
int* ParseListOfInts(const char* s, int* length);

/* Same as above, but for a list of doubles. */
double* ParseListOfDoubles(const char* s, int* length);

/* Rounds up to the next power of two. */
int RoundUpToPowerOfTwo(int value);

/* Generates pseudorandom integer uniformly in {0, 1, ..., max_value}. */
int RandomInt(int max_value);

/* Gets `smoother_coeff` such that the lowpass gamma filter implemented as
 * follows has its half-power frequency at `cutoff_frequency_hz`:
 *
 *   float next_stage_input = input_sample;
 *   for (int k = 0; k < order; ++k) {
 *     state[k] += smoother_coeff * (next_stage_input - state[k]);
 *     next_stage_input = state[k];
 *   }
 *   output_sample = next_stage_input;
 */
float GammaFilterSmootherCoeff(
    int order, float cutoff_frequency_hz, float sample_rate_hz);

/* Evaluates a Tukey window, nonzero over 0 < t < `window_duration` and having
 * transitions of length `transition`.
 */
float TukeyWindow(float window_duration, float transition, float t);

/* Permutes in-place the channels of a multichannel waveform with up to 32
 * channels. The waveform data is assumed to have interleaved channels,
 *
 *   waveform[num_channels * i + c] = sample in frame i, channel c.
 *
 * `permutation` is an array of size `num_channels` of 0-based indices. Input
 * channel permutation[c] is assigned to output channel c,
 *
 *   output_frame[c] = input_frame[permutation[c]].
 */
void PermuteWaveformChannels(const int* permutation, float* waveform,
                             int num_frames, int num_channels);

/* Writes a pretty-printed horizontal text bar to `buffer` as a null-terminated
 * UTF-8 string. This is useful for instance for volume meter displays. The bar
 * is `width` characters total with `fraction` (between 0 and 1) of the bar
 * filled. `buffer` must have size (3 * width + 1).
 */
void PrettyTextBar(int width, float fraction, char* buffer);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_EXTRAS_TOOLS_UTIL_H_ */
