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

#include "extras/python/tactile/tactile_worker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "extras/tools/portaudio_device.h"
#include "extras/tools/util.h"

void TactileWorkerSetDefaultParams(TactileWorkerParams* params) {
  params->input_device = NULL;
  params->output_device = NULL;
  params->chunk_size = 256;
  TactileProcessorSetDefaultParams(&params->tactile_processor_params);
  PostProcessorSetDefaultParams(&params->post_processor_params);
  ChannelMapParse(kTactileProcessorNumTactors, "0", NULL, &params->channel_map);
}

/* Sets TactileWorker variables to empty initial state. */
static void TactileWorkerInit(TactileWorker* worker) {
  worker->tactile_processor = NULL;
  worker->tactile_output = NULL;

  worker->mic_is_input = 0;
  worker->should_reset_tactile_processor = 0;
  worker->active_buffer_index = 0;
  worker->buffers[0] = NULL;
  worker->buffers[1] = NULL;

  worker->queue_samples = NULL;
  worker->queue_capacity = 0;
  worker->queue_size = 0;
  worker->queue_position = 0;

  worker->queue_thread_initialized = 0;
  worker->queue_thread_keep_running = 1;

  worker->pa_initialized = 0;
  worker->pa_stream = NULL;

  int c;
  for (c = 0; c < kNumTactors; ++c) {
    worker->volume_meters[c] = 0.0f;
  }
}

static int PortAudioCallback(const void* input_buffer, void* output_buffer,
                             unsigned long frames_per_buffer,
                             const PaStreamCallbackTimeInfo* time_info,
                             PaStreamCallbackFlags status_flags,
                             void* user_data) {
  /* Check whether PortAudio detected output underflow, meaning the program did
   * not complete this callback in time to provide an output chunk.
   *
   * Underflow may occur spuriously when the stream starts or stops, and may be
   * safely ignored.
   *
   * However, if underflow occurs during playback, this is a problem. When
   * underflow occurs in audio, it usually produces an audible glitch. This is
   * obviously bad also for tactile signals. A typical cause is that the
   * chunk_size is too small. We suggest the chunk size be a power of two number
   * of samples and at least 5 milliseconds,
   *    chunk_size >= 0.005 * sample_rate_hz.
   */
  if (status_flags & paOutputUnderflow) {
    fprintf(stderr, "Error: Underflow in tactile output. "
        "chunk_size (%lu) might be too small.\n", frames_per_buffer);
  }

  TactileWorker* worker = (TactileWorker*)user_data;

  /* Get input from the microphone or from the playback queue. */
  float* input = worker->mic_is_input
                     ? (float*)input_buffer
                     : worker->buffers[worker->active_buffer_index];
  float* output = (float*)output_buffer;

  const int block_size =
      CarlFrontendBlockSize(worker->tactile_processor->frontend);
  const int num_blocks = worker->chunk_size / block_size;
  float energy_accum[kNumTactors] = {0.0f};

  /* Reset tactile processing if requested. */
  if (worker->should_reset_tactile_processor) {
    worker->should_reset_tactile_processor = 0;
    TactileProcessorReset(worker->tactile_processor);
    PostProcessorReset(&worker->post_processor);
  }

  /* Process the chunk. */
  int b;
  for (b = 0; b < num_blocks; ++b) {
    float* tactile_output = worker->tactile_output;
    /* Run audio-to-tactile processing. */
    TactileProcessorProcessSamples(
        worker->tactile_processor, input, tactile_output);

    /* Accumulate signals for visualization. Do this before post processing. */
    int i;
    for (i = 0; i < block_size; ++i) {
      int c;
      for (c = 0; c < kNumTactors; ++c) {
        energy_accum[c] += tactile_output[c] * tactile_output[c];
      }
      tactile_output += kNumTactors;
    }

    /* Apply equalization, clipping, and lowpass filtering. */
    tactile_output = worker->tactile_output;
    PostProcessorProcessSamples(
        &worker->post_processor, tactile_output, block_size);

    /* Map channels and apply channel gains. */
    ChannelMapApply(&worker->channel_map, tactile_output, block_size, output);

    input += block_size;
    output += worker->channel_map.num_output_channels * block_size;
  }

  const float volume_decay_coeff = worker->volume_decay_coeff;
  int c;
  for (c = 0; c < kNumTactors; ++c) {
    // Convert tactile energy to perceived strength with Steven's power law.
    // Perceived strength is roughly proportional to acceleration^0.55, which is
    // proportional to sqrt(energy)^0.55.
    const float perceived = FastPow(1e-12f + energy_accum[c]
        / (num_blocks * block_size), 0.55f * 0.5f);
    float updated_volume = worker->volume_meters[c] * volume_decay_coeff;
    if (perceived > updated_volume) { updated_volume = perceived; }
    worker->volume_meters[c] = updated_volume;
  }

  if (!worker->mic_is_input) {
    /* Ping pong the buffers. */
    worker->active_buffer_index ^= 1;
    /* Signal the queue thread to start filling the next buffer. */
    pthread_cond_signal(&worker->queue_cond);
  }
  return paContinue;
}

static void* QueueThread(void* user_data) {
  TactileWorker* worker = (TactileWorker*)user_data;
  const int chunk_size = worker->chunk_size;
  pthread_mutex_lock(&worker->queue_mutex);

  while (1) {
    /* Wait until the consumer is done reading from the current buffer. */
    int buffer_index = worker->active_buffer_index;
    while (worker->queue_thread_keep_running &&
           worker->active_buffer_index == buffer_index) {
      pthread_cond_wait(&worker->queue_cond, &worker->queue_mutex);
    }
    if (!worker->queue_thread_keep_running) {
      break;
    }

    /* Now refill the buffer from `queued_samples`, zero-filling if we have
     * exhausted the queue.
     */
    float* dest = (float*)worker->buffers[buffer_index];
    int num_copy = worker->queue_size - worker->queue_position;
    if (num_copy > 0) {
      if (chunk_size < num_copy) {
        num_copy = chunk_size;
      }
      memcpy(dest, worker->queue_samples + worker->queue_position,
             num_copy * sizeof(float));
      worker->queue_position += num_copy;
    }
    int i;
    for (i = num_copy; i < chunk_size; ++i) {
      dest[i] = 0.0f;
    }

    /* Occasionally pop consumed samples off the queue. */
    if (worker->queue_position >= worker->queue_size) {
      worker->queue_size = 0;
      worker->queue_position = 0;
    } else if (worker->queue_position >= 16384) {
      const int new_size = worker->queue_size - worker->queue_position;
      if (new_size > 0) {
        memmove(worker->queue_samples,
                worker->queue_samples + worker->queue_position,
                new_size * sizeof(float));
      }
      worker->queue_size = new_size;
      worker->queue_position = 0;
    }
  }

  pthread_mutex_unlock(&worker->queue_mutex);
  return NULL;
}

/* Starts the queue thread. Returns 1 on success, 0 on failure. */
static int StartQueueThread(TactileWorker* worker) {
  if (pthread_mutex_init(&worker->queue_mutex, NULL) != 0) {
    return 0;
  } else if (pthread_cond_init(&worker->queue_cond, NULL) != 0) {
    pthread_mutex_destroy(&worker->queue_mutex);
    return 0;
  } else if (pthread_create(&worker->queue_thread, NULL, QueueThread, worker) !=
             0) {
    pthread_cond_destroy(&worker->queue_cond);
    pthread_mutex_destroy(&worker->queue_mutex);
    return 0;
  }
  worker->queue_thread_initialized = 1;
  return 1;
}

/* Starts PortAudio. Returns 1 on success, 0 on failure. */
static int StartPortAudio(TactileWorker* worker,
                          const TactileWorkerParams* params) {
  PaError pa_error = Pa_Initialize();
  if (pa_error != paNoError) {
    goto fail;
  }
  worker->pa_initialized = 1;

  /* Find PortAudio devices. */
  const int has_input = (params->input_device && *params->input_device);
  const int num_output_channels = worker->channel_map.num_output_channels;
  const int input_device_index =
      FindPortAudioDevice(params->input_device, 1, 0);
  const int output_device_index =
      FindPortAudioDevice(params->output_device, 0, num_output_channels);
  if ((has_input && input_device_index < 0) || output_device_index < 0) {
    if (has_input && input_device_index < 0) {
      fprintf(stderr, "\nError: Invalid input device: \"%s\"\n",
          params->input_device);
    }
    if (output_device_index < 0) {
      fprintf(stderr, "\nError: Invalid output device: \"%s\"\n",
          params->output_device);
    }

    fprintf(stderr, "\nPlease set valid devices:\n");
    PrintPortAudioDevices();
    goto fail;
  }

  /* Log audio parameters. */
  printf(
      "sample rate: %d Hz\n"
      "chunk size: %d frames (%.1f ms)\n",
      worker->sample_rate_hz, worker->chunk_size,
      (1000.0f * worker->chunk_size) / worker->sample_rate_hz);
  if (params->post_processor_params.use_equalizer) {
    printf("equalizer: mid %.1f dB, high %.1f dB\n",
           AmplitudeRatioToDecibels(params->post_processor_params.mid_gain),
           AmplitudeRatioToDecibels(params->post_processor_params.high_gain));
  } else {
    printf("equalizer: off\n");
  }

  if (has_input) {
    printf("input device: #%d %s\n", input_device_index,
           Pa_GetDeviceInfo(input_device_index)->name);
  } else {
    printf("input device: none\n");
  }
  printf("output device: #%d %s\n", output_device_index,
         Pa_GetDeviceInfo(output_device_index)->name);
  printf("output channels:\n");
  ChannelMapPrint(&worker->channel_map);

  PaStreamParameters input_parameters;
  if (has_input) {
    input_parameters.device = input_device_index;
    input_parameters.channelCount = 1;
    input_parameters.sampleFormat = paFloat32;
    input_parameters.suggestedLatency =
        Pa_GetDeviceInfo(input_device_index)->defaultLowInputLatency;
    input_parameters.hostApiSpecificStreamInfo = NULL;
  }

  PaStreamParameters output_parameters;
  output_parameters.device = output_device_index;
  output_parameters.channelCount = num_output_channels;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_device_index)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  /* Open and start PortAudio stream. */
  pa_error = Pa_OpenStream(&worker->pa_stream,
                           (has_input) ? &input_parameters : NULL,
                           &output_parameters, worker->sample_rate_hz,
                           worker->chunk_size, 0, PortAudioCallback, worker);
  if (pa_error != paNoError) {
    goto fail;
  }

  pa_error = Pa_StartStream(worker->pa_stream);
  if (pa_error != paNoError) {
    goto fail;
  }

  return 1;

fail:
  if (pa_error != paNoError) {
    fprintf(stderr, "Error: %s\n", Pa_GetErrorText(pa_error));
  }
  return 0;
}

TactileWorker* TactileWorkerMake(TactileWorkerParams* params) {
  if (!params || params->channel_map.num_input_channels != kNumTactors) {
    return NULL;
  }
  TactileWorker* worker = (TactileWorker*)malloc(sizeof(TactileWorker));
  if (!worker) {
    return NULL;
  }
  TactileWorkerInit(worker);

  worker->sample_rate_hz =
      params->tactile_processor_params.frontend_params.input_sample_rate_hz;
  worker->channel_map = params->channel_map;

  /* Make TactileProcessor. */
  worker->tactile_processor =
      TactileProcessorMake(&params->tactile_processor_params);
  if (!worker->tactile_processor) {
    goto fail;
  }
  /* Create PostProcessor. */
  const float output_sample_rate_hz =
      TactileProcessorOutputSampleRateHz(&params->tactile_processor_params);
  if (!PostProcessorInit(&worker->post_processor,
                         &params->post_processor_params,
                         output_sample_rate_hz, kTactileProcessorNumTactors)) {
    return 0;
  }

  worker->chunk_size = params->chunk_size;
  /* Adjust chunk_size to a power of two >= max(64, block_size, 5 ms). */
  if (worker->chunk_size < 64) { worker->chunk_size = 64; }
  int block_size = CarlFrontendBlockSize(worker->tactile_processor->frontend);
  if (worker->chunk_size < block_size) { worker->chunk_size = block_size; }
  int size_5ms = (int)ceil(0.005 * worker->sample_rate_hz);
  if (worker->chunk_size < size_5ms) { worker->chunk_size = size_5ms; }
  worker->chunk_size = RoundUpToPowerOfTwo(worker->chunk_size);

  if (worker->chunk_size != params->chunk_size) {
    fprintf(stderr, "Warning: Adjusted chunk_size "
        "from %d (%.1f ms) to %d (%.1f ms).\n",
        params->chunk_size,
        (1000.0f * params->chunk_size) / worker->sample_rate_hz,
        worker->chunk_size,
        (1000.0f * worker->chunk_size) / worker->sample_rate_hz);
  }

  const float kVolumeMeterTimeConstantSeconds = 0.05;
  worker->volume_decay_coeff =
      (float)exp(-worker->chunk_size /
                 (kVolumeMeterTimeConstantSeconds * worker->sample_rate_hz));

  /* Allocate buffers. */
  worker->tactile_output = (float*)malloc(
      sizeof(float) * kNumTactors *
      params->tactile_processor_params.frontend_params.block_size);
  if (!worker->tactile_output) {
    goto fail;
  }
  worker->buffers[0] = (float*)malloc(2 * worker->chunk_size * sizeof(float));
  if (!worker->buffers[0]) {
    goto fail;
  }
  worker->buffers[1] = worker->buffers[0] + worker->chunk_size;

  int i;
  for (i = 0; i < 2 * worker->chunk_size; ++i) {
    worker->buffers[0][i] = 0.0f;
  }

  if (!StartQueueThread(worker) || !StartPortAudio(worker, params)) {
    goto fail;
  }
  return worker;

fail:
  TactileWorkerFree(worker);
  return NULL;
}

void TactileWorkerFree(TactileWorker* worker) {
  if (worker) {
    if (worker->pa_initialized) { /* Clean up PortAudio. */
      if (worker->pa_stream) {
        Pa_CloseStream(worker->pa_stream);
      }
      Pa_Terminate();
    }

    if (worker->queue_thread_initialized) { /* Clean up queue thread. */
      pthread_mutex_lock(&worker->queue_mutex);
      worker->queue_thread_keep_running = 0;
      pthread_mutex_unlock(&worker->queue_mutex);
      pthread_cond_broadcast(&worker->queue_cond);
      pthread_join(worker->queue_thread, NULL);

      pthread_cond_destroy(&worker->queue_cond);
      pthread_mutex_destroy(&worker->queue_mutex);
    }

    free(worker->queue_samples);
    free(worker->buffers[0]);
    free(worker->tactile_output);
    TactileProcessorFree(worker->tactile_processor);
    TactileWorkerInit(worker);
    free(worker);
  }
}

void TactileWorkerReset(TactileWorker* worker) {
  worker->should_reset_tactile_processor = 1;
}

void TactileWorkerSetMicInput(TactileWorker* worker) {
  worker->mic_is_input = 1;
}

void TactileWorkerSetPlaybackInput(TactileWorker* worker) {
  worker->mic_is_input = 0;
}

int TactileWorkerPlay(TactileWorker* worker, float* samples, int num_samples) {
  pthread_mutex_lock(&worker->queue_mutex);

  const int new_size = worker->queue_size + num_samples;
  if (new_size > worker->queue_capacity) { /* Grow queue_samples if needed. */
    const int new_capacity = 2 * new_size;
    float* new_queue_samples =
        (float*)realloc(worker->queue_samples, new_capacity * sizeof(float));
    if (!new_queue_samples) {
      pthread_mutex_unlock(&worker->queue_mutex);
      return 0;
    }
    worker->queue_samples = new_queue_samples;
    worker->queue_capacity = new_capacity;
  }
  /* Append `samples` to queue_samples. */
  memcpy(worker->queue_samples + worker->queue_size, samples,
         num_samples * sizeof(float));
  worker->queue_size = new_size;

  pthread_mutex_unlock(&worker->queue_mutex);
  return 1;
}

int TactileWorkerGetRemainingPlaybackSamples(TactileWorker* worker) {
  int num_remaining;
  pthread_mutex_lock(&worker->queue_mutex);
  num_remaining = worker->queue_size - worker->queue_position;
  pthread_mutex_unlock(&worker->queue_mutex);
  return num_remaining;
}

void TactileWorkerGetVolumeMeters(TactileWorker* worker, float* volume_meters) {
  memcpy(volume_meters, (float*)worker->volume_meters,
         kNumTactors * sizeof(float));
}
