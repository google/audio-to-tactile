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
 * 16, 24, or 32-bit WAV writer. Don't use this file directly unless you are
 * adding support for a different kind of filesystem.
 *
 * For reading local files, see write_wav_file.h.
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_WRITE_WAV_FILE_GENERIC_H_
#define AUDIO_TO_TACTILE_SRC_DSP_WRITE_WAV_FILE_GENERIC_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allows callers to provide custom IO callbacks for writing WAV files. */
struct WavWriter {
  /* Pointer to user-defined struct of state needed for write_fun. */
  void* io_ptr;

  /* A function for writing bytes. Return the number of bytes written. */
  size_t (*write_fun)(const void* bytes, size_t num_bytes, void* io_ptr);

  /* 1 for error, 0 for no error. This is used internally. No need to set it. */
  int has_error;
};
typedef struct WavWriter WavWriter;

/* Write a WAV file header at the beginning of a WAV file.  Returns 1 on
 * success, 0 on failure.
 */
int WriteWavHeaderGeneric(WavWriter* w, size_t num_samples, int sample_rate_hz,
                          int num_channels);

int WriteWavHeaderGeneric24Bit(WavWriter* w, size_t num_samples,
                               int sample_rate_hz, int num_channels);

int WriteWavHeaderGeneric32Bit(WavWriter* w, size_t num_samples,
                               int sample_rate_hz, int num_channels);

/* Write samples into a WAV file.  samples should be interleaved, and
 * num_samples must be an integer multiple of num_channels. Returns 1 on
 * success, 0 on failure.
 */
int WriteWavSamplesGeneric(WavWriter* w, const int16_t* samples,
                           size_t num_samples);

/* Same as above but writing 24-bit samples. The lower 8 bits truncated. */
int WriteWavSamplesGeneric24Bit(WavWriter* w, const int32_t* samples,
                                size_t num_samples);

/* Same as above but writing 32-bit samples. */
int WriteWavSamplesGeneric32Bit(WavWriter* w, const int32_t* samples,
                                size_t num_samples);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_DSP_WRITE_WAV_FILE_GENERIC_H_ */
