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
 * Miscellaneous utilities for tactile processing.
 */

#ifndef AUDIO_TACTILE_UTIL_H_
#define AUDIO_TACTILE_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if string `s` starts with `prefix`, 0 otherwise. */
int StartsWith(const char* s, const char* prefix);

/* Parses a comma-delimited list of ints. An array of the parsed ints is
 * allocated and returned, and should be freed by the caller. The number of ints
 * is written to `*length`. Returns NULL on failure.
 */
int* ParseListOfInts(const char* s, int* length);

/* Same as above, but for a list of doubles. */
double* ParseListOfDoubles(const char* s, int* length);

/* Generate pseudorandom integer uniformly in {0, 1, ..., max_value}. */
int RandomInt(int max_value);

/* Convert Decibels to amplitude ratio. */
float DecibelsToAmplitudeRatio(float decibels);

/* Convert amplitude ratio to Decibels. */
float AmplitudeRatioToDecibels(float amplitude_ratio);

/* Evaluate a Tukey window, nonzero over 0 < t < `window_duration` and having
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

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TACTILE_UTIL_H_ */
