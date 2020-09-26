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
 * Runs AutoGainControl on a WAV file.
 *
 * This is a small program that runs AutoGainControl on a mono WAV file.
 *
 * Flags:
 *  --input=<path>             Input WAV file path.
 *  --output=<path>            Output WAV file path.
 *  --output_gain=<float>
 *  --time_constant_s=<float>  AGC time constant in seconds.
 *  --agc_strength=<float>     AGC strength, a value between 0 and 1.
 *  --power_floor=<float>      Offset added to smoothed power estimate.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/auto_gain_control.h"
#include "dsp/read_wav_file.h"
#include "dsp/write_wav_file.h"
#include "tactile/util.h"

#define kChunkSize 256

static const float kInt16Scale = -1.0f * INT16_MIN;

/* Convert array of int16_t to float values in [-1, 1]. */
void ConvertInt16ToFloat(const int16_t* in, int num_samples, float* out) {
  int i;
  for (i = 0; i < num_samples; ++i) {
    out[i] = in[i] / kInt16Scale;
  }
}

/* Convert array of floats in [-1, 1] to int16_t. */
void ConvertFloatToInt16(const float* in, size_t num_samples, int16_t* out) {
  size_t i;
  for (i = 0; i < num_samples; ++i) {
    const float value = floor(kInt16Scale * in[i] + 0.5f);
    if (value <= (float)INT16_MIN) {
      out[i] = INT16_MIN;
    } else if (value >= (float)INT16_MAX) {
      out[i] = INT16_MAX;
    } else {
      out[i] = (int16_t)value;
    }
  }
}

int main(int argc, char** argv) {
  const char* input_wav = NULL;
  const char* output_wav = NULL;
  FILE* f_in = NULL;
  FILE* f_out = NULL;
  float output_gain = 1.0f;
  float time_constant_s = 0.5f;
  float agc_strength = 0.5f;
  float power_floor = 1e-6f;
  int i;

  for (i = 1; i < argc; ++i) { /* Parse flags. */
    if (StartsWith(argv[i], "--input=")) {
      input_wav = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output=")) {
      output_wav = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output_gain=")) {
      output_gain = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--time_constant_s=")) {
      time_constant_s = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--agc_strength=")) {
      agc_strength = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--power_floor=")) {
      power_floor = atof(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      goto fail;
    }
  }

  if (input_wav == NULL) {
    fprintf(stderr, "Must specify --input\n");
    goto fail;
  } else if (output_wav == NULL) {
    fprintf(stderr, "Must specify --output\n");
    goto fail;
  }

  /* Begin reading input WAV file. */
  f_in = fopen(input_wav, "rb");
  if (f_in == NULL) {
    fprintf(stderr, "Failed to open \"%s\"\n", input_wav);
    goto fail;
  }
  ReadWavInfo wav_info;
  if (!ReadWavHeader(f_in, &wav_info)) {
    fprintf(stderr, "Error reading \"%s\"\n", input_wav);
    goto fail;
  } else if (wav_info.num_channels != 1) {
    fprintf(stderr, "Input WAV must have 1 channel\n");
    goto fail;
  }
  const int sample_rate_hz = wav_info.sample_rate_hz;

  /* Initialize AGC. */
  AutoGainControlState state;
  if (!AutoGainControlInit(&state,
        sample_rate_hz, time_constant_s, agc_strength, power_floor)) {
    fprintf(stderr, "Error: AutoGainControlInit failed.\n");
    goto fail;
  }

  /* Begin writing output WAV file. */
  f_out = fopen(output_wav, "wb");
  if (f_out == NULL) {
    fprintf(stderr, "Failed to create output file \"%s\"\n", output_wav);
    goto fail;
  } else if (!WriteWavHeader(f_out, 0, sample_rate_hz, 1)) {
    fprintf(stderr, "Error while writing \"%s\"\n", output_wav);
    goto fail;
  }

  float input_buffer[kChunkSize];
  float output_buffer[kChunkSize];
  int16_t buffer_int16[kChunkSize];
  size_t num_read;
  size_t total_written = 0;

  /* Main loop, each iteration processing kChunkSize input samples. */
  do {
    num_read = Read16BitWavSamples(f_in, &wav_info, buffer_int16, kChunkSize);
    ConvertInt16ToFloat(buffer_int16, num_read, input_buffer);

    int i;
    for (i = 0; i < num_read; ++i) {
      const float power_sample = input_buffer[i] * input_buffer[i];
      AutoGainControlProcessSample(&state, power_sample);
      output_buffer[i] =
          input_buffer[i] * output_gain * AutoGainControlGetGain(&state);
    }

    const size_t num_written = num_read;
    total_written += num_written;

    ConvertFloatToInt16(output_buffer, num_written, buffer_int16);
    if (!WriteWavSamples(f_out, buffer_int16, num_written)) {
      fprintf(stderr, "Error while writing \"%s\"\n", output_wav);
      goto fail;
    }
  } while (num_read == kChunkSize);

  /* Rewind and write the total number of samples. */
  if (fseek(f_out, 0, SEEK_SET) != 0 ||
      !WriteWavHeader(f_out, total_written, sample_rate_hz, 1)) {
    fprintf(stderr, "Error while writing \"%s\"\n", output_wav);
    goto fail;
  }

  fclose(f_out);
  fclose(f_in);
  return EXIT_SUCCESS;

fail:
  if (f_out) { fclose(f_out); }
  if (f_in) { fclose(f_in); }
  return EXIT_FAILURE;
}

