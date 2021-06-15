/* Copyright 2019, 2021 Google LLC
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
 * Run PortAudio and TactileProcessor in a background worker.
 *
 * This C library implements a background worker that runs a PortAudio stream
 * and runs TactileProcessor on it for real-time audio-to-tactile conversion. We
 * wrap this library in tactile_worker_python_bindings.c to run real-time
 * audio-to-tactile processing in Python.
 *
 * Input audio can be taken from either a PortAudio input device (microphone) or
 * from a playback queue of samples.
 *
 * It is a bad idea in an audio thread to use mutexes (which can cause thread
 * priority inversion) or make heap allocation calls (which again involves
 * mutexes). This is explained in detail for instance in
 *
 *   http://atastypixel.com/blog/four-common-mistakes-in-audio-development/
 *
 * So we use a separate thread to manage the playback queue and to mediate
 * between public APIs and the audio thread:
 *
 *   Main/Python thread <---> Queue thread  <---> Audio thread.
 *
 * Synchronization between the main thread and queue thread is done with
 * mutexes. Synchronization between the queue thread and audio thread is done
 * with lockless ping-pong buffering and a few volatile variables.
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_PYTHON_TACTILE_TACTILE_WORKER_H_
#define AUDIO_TO_TACTILE_EXTRAS_PYTHON_TACTILE_TACTILE_WORKER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>

#include "extras/tools/channel_map_tui.h"
#include "src/tactile/post_processor.h"
#include "src/tactile/tactile_processor.h"
#include "portaudio.h"

#define kNumTactors 10

typedef struct {
  /* PortAudio device to take input audio from, or NULL for no input device. */
  char* input_device;
  /* PortAudio device to play tactile output signals to. */
  char* output_device;
  /* Number of frames per audio buffer. */
  int chunk_size;
  /* TactileProcessor parameters, including the sample rate from
   * `tactile_processor_params.frontend_params.input_sample_rate_hz`.
   */
  TactileProcessorParams tactile_processor_params;
  /* Post processing parameters. */
  PostProcessorParams post_processor_params;
  /* Configuration for mapping tactile signals to output channels. */
  ChannelMap channel_map;
} TactileWorkerParams;

/* Set TactileWorkerParams to defaults. */
void TactileWorkerSetDefaultParams(TactileWorkerParams* params);

typedef struct {
  /* The TactileProcessor for audio-to-tactile processing. */
  TactileProcessor* tactile_processor;
  /* Post processing state. */
  PostProcessor post_processor;
  /* Buffer for one CARL block's worth of tactile output. */
  float* tactile_output;
  /* Configuration for mapping tactile signals to output channels. */
  ChannelMap channel_map;
  /* Audio sample rate in Hz. */
  int sample_rate_hz;
  /* Number of frames per audio buffer. */
  int chunk_size;
  /* Coefficient for decaying volume meters, applied once per chunk. */
  float volume_decay_coeff;

  /* Audio thread variables. */

  /* Ping pong buffers for synchronization with the audio thread. */
  float* buffers[2];
  /* The audio thread reads from `buffers[active_buffer_index]`. */
  volatile int active_buffer_index;
  /* 1 => take input from microphone, 0 => input from playback queue. */
  volatile int /* bool */ mic_is_input;
  /* Nonzero if Reset method was called. */
  volatile int /* bool */ should_reset_tactile_processor;
  /* Volume meter for each tactor, used for visualization. */
  volatile float volume_meters[kNumTactors];

  /* Queue thread variables. */

  /* Buffer of audio samples in the queue. */
  float* queue_samples;
  /* Allocated queue capacity. */
  int queue_capacity;
  /* Queue size, always less than `queue_capacity`. */
  int queue_size;
  /* Queue read position, always less than `queue_size`. */
  int queue_position;
  /* Queue thread. */
  pthread_t queue_thread;
  /* Mutex guarding all queue variables. */
  pthread_mutex_t queue_mutex;
  /* Cond used to signal the queue thread when to fill the ring buffer. */
  pthread_cond_t queue_cond;
  /* Nonzero if queue thread initialized successfully. */
  int /* bool */ queue_thread_initialized;
  /* Nonzero if queue thread should keep running. */
  int /* bool */ queue_thread_keep_running;

  /* PortAudio variables. */

  /* Nonzero if PortAudio initialized successfully. */
  int /* bool */ pa_initialized;
  /* The PortAudio stream. */
  PaStream* pa_stream;
} TactileWorker;

/* Makes a `TactileWorker`. The caller should free it when done with
 * `TactileWorkerFree`. Returns NULL on failure.
 */
TactileWorker* TactileWorkerMake(TactileWorkerParams* params);

/* Frees a `TactileWorker`. */
void TactileWorkerFree(TactileWorker* worker);

/* Resets tactile processing to initial state. */
void TactileWorkerReset(TactileWorker* worker);

/* Sets the worker to take input from the microphone. */
void TactileWorkerSetMicInput(TactileWorker* worker);

/* Sets the worker to take input from the playback queue. */
void TactileWorkerSetPlaybackInput(TactileWorker* worker);

/* This function appends `input_samples` audio to the playback queue, which will
 * get converted to tactile and played to the output device when the playback
 * input source is selected (with SetPlaybackInput).
 */
int TactileWorkerPlay(TactileWorker* worker, float* samples, int num_samples);

/* Gets the number of samples remaining before playback completes. */
int TactileWorkerGetRemainingPlaybackSamples(TactileWorker* worker);

/* Gets the current tactor volume levels as size-kNumTactors array. */
void TactileWorkerGetVolumeMeters(TactileWorker* worker, float* volume_meters);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_EXTRAS_PYTHON_TACTILE_TACTILE_WORKER_H_ */
