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
 * Runs Demuxer on a WAV file.
 *
 * This is a small program that runs Demuxer on a muxed WAV file, producing a
 * 12-channel output WAV file.
 *
 * Flags:
 *  --input=<path>              Input WAV file path.
 *  --output=<path>             Output WAV file path.
 *  --output_sample_rate=<int>  Output sample rate in Hz.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "extras/tools/util.h"
#include "src/dsp/convert_sample.h"
#include "src/dsp/logging.h"
#include "src/dsp/rational_factor_resampler.h"
#include "src/dsp/read_wav_file.h"
#include "src/dsp/write_wav_file.h"
#include "src/mux/demuxer.h"

/* The demuxer takes its input in multiples of kMuxRateFactor samples, but in
 * this program its input comes from a resampler. So we use this queue data
 * structure to buffer samples between the resampler and demuxer.
 */
typedef struct {
  float* data;
  int capacity;
  int size;
} Queue;

/* Construct queue with specifie capacity. */
static void QueueInit(Queue* queue, int capacity) {
  queue->data = (float*)CHECK_NOTNULL(malloc(capacity * sizeof(float)));
  queue->capacity = capacity;
  queue->size = 0;
}

/* Insert elements to the tail of the queue. */
static void QueueInsert(Queue* queue, const float* data, int size) {
  const int new_size = queue->size + size;
  CHECK(new_size <= queue->capacity);
  memcpy(queue->data + queue->size, data, size * sizeof(float));
  queue->size = new_size;
}

/* Discard elements from the head of the queue. */
static void QueueDiscard(Queue* queue, int size) {
  const int new_size = queue->size - size;
  CHECK(new_size >= 0);
  if (new_size) {
    memmove(queue->data, queue->data + size, new_size * sizeof(float));
  }
  queue->size = new_size;
}

int main(int argc, char** argv) {
  const char* input_wav = NULL;
  const char* output_wav = NULL;
  int output_sample_rate = 2000;
  FILE* f_in = NULL;
  FILE* f_out = NULL;
  RationalFactorResampler* resampler = NULL;
  RationalFactorResampler* output_resampler = NULL;
  Queue queue = {NULL, 0, 0};
  int16_t* buffer_int16 = NULL;
  float* buffer_float = NULL;
  int status = EXIT_FAILURE;
  int i;

  for (i = 1; i < argc; ++i) { /* Parse flags. */
    if (StartsWith(argv[i], "--input=")) {
      input_wav = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output=")) {
      output_wav = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output_sample_rate=")) {
      output_sample_rate = atoi(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      goto done;
    }
  }

  const int kMinOutputSampleRateHz = ceil(2 * kMuxTactileMaxHz);
  if (input_wav == NULL) {
    fprintf(stderr, "Error: Must specify --input\n");
    goto done;
  } else if (output_wav == NULL) {
    fprintf(stderr, "Error: Must specify --output\n");
    goto done;
  } else if (output_sample_rate < kMinOutputSampleRateHz) {
    fprintf(stderr, "Error: output_sample_rate must be at least %d.\n",
        kMinOutputSampleRateHz);
    goto done;
  }

  /* Begin reading input WAV file. */
  f_in = fopen(input_wav, "rb");
  if (f_in == NULL) {
    fprintf(stderr, "Error: Failed to open \"%s\"\n", input_wav);
    goto done;
  }
  ReadWavInfo wav_info;
  if (!ReadWavHeader(f_in, &wav_info)) {
    fprintf(stderr, "Error reading \"%s\"\n", input_wav);
    goto done;
  } else if (wav_info.num_channels != 1) {
    fprintf(stderr, "Error: Expected a single-channel WAV, got: %d channels\n",
            wav_info.num_channels);
    goto done;
  }
  const int input_channels = wav_info.num_channels;
  const int input_sample_rate_hz = wav_info.sample_rate_hz;
  const int max_in_frames = 1024;

  /* Prepare to resample from the WAV file sample rate to kMuxMuxedRate. */
  resampler = RationalFactorResamplerMake(input_sample_rate_hz, kMuxMuxedRate,
                                          1, max_in_frames, NULL);
  if (resampler == NULL) {
    fprintf(stderr, "Error constructing resampler.\n");
    goto done;
  }
  const int max_resampled_frames =
      RationalFactorResamplerMaxOutputFrames(resampler);

  /* Initialize demuxer. */
  QueueInit(&queue, max_resampled_frames + kMuxRateFactor - 1);
  Demuxer demuxer;
  DemuxerInit(&demuxer);
  const int max_demuxed_frames = queue.capacity / kMuxRateFactor;

  /* Prepare to resample to output_sample_rate. */
  output_resampler =
      RationalFactorResamplerMake(kMuxTactileRate, output_sample_rate,
                                  kMuxChannels, max_demuxed_frames, NULL);
  if (output_resampler == NULL) {
    fprintf(stderr, "Error constructing output resampler.\n");
    goto done;
  }
  const int max_output_frames =
      RationalFactorResamplerMaxOutputFrames(output_resampler);

  /* Begin writing output WAV file. */
  f_out = fopen(output_wav, "wb");
  if (f_out == NULL) {
    fprintf(stderr, "Failed to create output file \"%s\"\n", output_wav);
    goto done;
  } else if (!WriteWavHeader(f_out, 0, output_sample_rate, kMuxChannels)) {
    fprintf(stderr, "Error while writing \"%s\"\n", output_wav);
    goto done;
  }

  int buffer_size = max_demuxed_frames * kMuxChannels;
  if (max_in_frames > buffer_size) { buffer_size = max_in_frames; }
  if (max_output_frames * kMuxChannels > buffer_size) {
    buffer_size = max_output_frames * kMuxChannels;
  }
  if (!(buffer_int16 = (int16_t*)malloc(buffer_size * sizeof(int16_t))) ||
      !(buffer_float = (float*)malloc(buffer_size * sizeof(float)))) {
    fprintf(stderr, "Error: Out of memory.\n");
    goto done;
  }

  /* Compute how many frames are needed to flush the resamplers. */
  int factor_numerator;
  int factor_denominator;
  RationalFactorResamplerGetRationalFactor(
      resampler, &factor_numerator, &factor_denominator);
  int num_flush_frames =
      RationalFactorResamplerFlushFrames(resampler) +
      (RationalFactorResamplerFlushFrames(output_resampler) *
       factor_denominator + factor_numerator - 1) / factor_numerator;

  const size_t max_read = max_in_frames * input_channels;
  size_t num_read;
  size_t total_written = 0;

  /* Main loop, each iteration processing kChunkSize input samples. */
  do {
    /* Read from input WAV. */
    num_read = Read16BitWavSamples(f_in, &wav_info, buffer_int16,
                                   max_read);
    if (num_read < max_read && num_flush_frames > 0) {
      /* At the end of the WAV, append some zeros for flushing. */
      int num_append = (max_read - num_read) / input_channels;
      if (num_flush_frames < num_append) { num_append = num_flush_frames; }
      memset(buffer_int16 + num_read, 0,
             num_append * input_channels * sizeof(int16_t));
      num_read += num_append * input_channels;
      num_flush_frames -= num_append;
    }

    /* Convert int16 samples to float. */
    ConvertSampleArrayInt16ToFloat(buffer_int16, num_read, buffer_float);

    /* Resample to kMuxMuxedRate. */
    const int num_resampled_frames = RationalFactorResamplerProcessSamples(
        resampler, buffer_float, num_read);

    /* Insert buffer_float into queue. */
    QueueInsert(&queue, RationalFactorResamplerOutput(resampler),
                num_resampled_frames);

    /* Demux, writing to buffer_float. */
    const int num_demuxed_frames = queue.size / kMuxRateFactor;
    const int num_consume = num_demuxed_frames * kMuxRateFactor;
    DemuxerProcessSamples(&demuxer, queue.data, num_consume, buffer_float);
    QueueDiscard(&queue, num_consume);

    /* Resample to output_sample_rate. */
    const int num_output_frames = RationalFactorResamplerProcessSamples(
        output_resampler, buffer_float, num_demuxed_frames);
    const int num_output_samples = num_output_frames * kMuxChannels;

    /* Convert float -> int16. */
    ConvertSampleArrayFloatToInt16(
        RationalFactorResamplerOutput(output_resampler), num_output_samples,
        buffer_int16);

    if (!WriteWavSamples(f_out, buffer_int16, num_output_samples)) {
      fprintf(stderr, "Error while writing \"%s\"\n", output_wav);
      goto done;
    }
    total_written += num_output_samples;
  } while (num_read == max_read);

  /* Rewind and write the total number of samples. */
  if (fseek(f_out, 0, SEEK_SET) != 0 ||
      !WriteWavHeader(f_out, total_written, output_sample_rate, kMuxChannels)) {
    fprintf(stderr, "Error while writing \"%s\"\n", output_wav);
    goto done;
  }

  status = EXIT_SUCCESS;

  /* We jump to `done` on error. The cleanup code that follows is executed
   * regardless, on either success or failure.
   */
done:
  if (f_out) { fclose(f_out); }
  if (f_in) { fclose(f_in); }
  free(buffer_float);
  free(buffer_int16);
  RationalFactorResamplerFree(output_resampler);
  free(queue.data);
  RationalFactorResamplerFree(resampler);
  return status;
}
