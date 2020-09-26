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
 * Sound diagnostic program: plays a buzz on one channel.
 *
 * Flags:
 *  --output=<int>          Output device number to play tactor signals to.
 *  --sample_rate_hz=<int>  Sample rate. Note that most devices only support
 *                          a few standard audio sample rates, e.g. 44100.
 *  --num_channels=<int>    Number of output channels. (Default 1)
 *  --channel=<int>         Which channel to play buzz on, counting from 1, or
 *                          `--channel=all` for all channels. (Default 1)
 *  --amplitude=<float>     Buzz amplitude, value in [0.0, 1.0]. (Default 0.2)
 *  --frequency_hz=<float>  Buzz frequency. (Default 250.0)
 */

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "portaudio.h"

#include "dsp/math_constants.h"
#include "tactile/util.h"

/* `Buzzer` is an object that plays a buzz waveform in a loop. */
struct BuzzerState {
  float* waveform;
  int size;
  volatile int position;
};
typedef struct BuzzerState BuzzerState;

/* The main processing loop runs while `is_running` is nonzero. We set a
 * signal handler to change `is_running` to 0 when Ctrl+C is pressed.
 */
static volatile int is_running = 0;
void StopOnCtrlC(int signum /*unused*/) { is_running = 0; }

/* Globals for running the tactors. */
const int kChunkSize = 64;
int g_num_channels = 1;
int g_channel_index = 1;
BuzzerState g_buzzer_state = {NULL, 0, 0};

/* Initializes Buzzer. Generates a 0.8-second long waveform of a sine wave
 * enveloped with a Tukey window. The window is nonzero over 0 <= t <= 0.4s.
 * When looped, it plays, pauses, plays, pauses, and so on with 50% duty cycle.
 */
int BuzzerInit(BuzzerState* state, float amplitude, float frequency_hz,
               float sample_rate_hz) {
  const float kLoopDuration = 0.8f;
  const float kBuzzDuration = 0.4f; /* Play buzz with 50% duty cycle. */
  const float kTransition = 0.1f;

  state->position = 0;
  state->size = (int)(kLoopDuration * sample_rate_hz + 0.5f);
  state->waveform = (float*)malloc(state->size * sizeof(float));
  if (state->waveform == NULL) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return 0;
  }

  const float rad_per_s = 2.0 * M_PI * frequency_hz;
  int i;
  for (i = 0; i < state->size; ++i) {
    const float t = i / sample_rate_hz;
    state->waveform[i] = amplitude * TukeyWindow(kBuzzDuration, kTransition, t)
        * sin(rad_per_s * t);
  }
  return 1;
}

/* Writes the next `num_samples` samples to `output`. */
void BuzzerGenerate(BuzzerState* state, int num_samples,
                    float* output, int output_stride) {
  const float* waveform = state->waveform;
  const int size = state->size;
  int position = state->position;

  while (num_samples > 0) {
    int count = size - position;
    if (num_samples < count) {
      count = num_samples;
    }

    const float* source = waveform + position;
    int i;
    for (i = 0; i < count; ++i) {
      *output = source[i];
      output += output_stride;
    }

    num_samples -= count;
    position += count;
    if (position >= size) {
      position = 0;  /* Loop playback back to beginning of the waveform. */
    }
  }

  state->position = position;
}

/* Checks that selected output device is valid, and if not, print list of
 * available devices along with their max number of output channels.
 */
int CheckOutputDevice(int output_device) {
  const int num_devices = Pa_GetDeviceCount();
  if (!(0 <= output_device && output_device < num_devices)) {
    printf("\nError: Use --output flag to set a valid device:\n");
    int i;
    for (i = 0; i < num_devices; ++i) {
      const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
      printf("#%-2d %-45s channels: %d\n", i, device_info->name,
             device_info->maxOutputChannels);
    }
    return 0;
  }
  printf("Output device: #%d %s\n", output_device,
         Pa_GetDeviceInfo(output_device)->name);
  return 1;
}

/* Stream callback function. In each call, portaudio passes kChunkSize frames
 * of input, and we process it to produce kChunkSize frames of output.
 */
int Callback(const void* input_buffer, void* output_buffer,
             unsigned long frames_per_buffer,
             const PaStreamCallbackTimeInfo* time_info,
             PaStreamCallbackFlags status_flags, void* user_data) {
  float* output = (float*)output_buffer;

  if (g_channel_index == -1) {
    BuzzerGenerate(&g_buzzer_state, frames_per_buffer, output, g_num_channels);
    int i;
    for (i = 0; i < frames_per_buffer; ++i) {
      float value = *output++;
      int c;
      for (c = 1; c < g_num_channels; ++c) {
        *output++ = value;
      }
    }
  } else {
    const int num_samples = frames_per_buffer * g_num_channels;
    int i;
    for (i = 0; i < num_samples; ++i) {
      output[i] = 0.0f;
    }

    BuzzerGenerate(&g_buzzer_state, frames_per_buffer,
                   output + g_channel_index - 1, g_num_channels);
  }
  return paContinue;
}

int main(int argc, char** argv) {
  PaError err = Pa_Initialize();
  if (err != paNoError) { goto fail; }

  int output_device = Pa_GetDefaultOutputDevice();
  int sample_rate_hz = 44100;
  float amplitude = 0.2f;
  float frequency_hz = 250.0f;
  int i;

  for (i = 1; i < argc; ++i) { /* Parse flags. */
    if (StartsWith(argv[i], "--output=")) {
      output_device = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--sample_rate_hz=")) {
      sample_rate_hz = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--num_channels=")) {
      g_num_channels = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--channel=")) {
      if (!strcmp(argv[i], "--channel=all")) {
        g_channel_index = -1;
      } else {
        g_channel_index = atoi(strchr(argv[i], '=') + 1);
      }
    } else if (StartsWith(argv[i], "--amplitude=")) {
      amplitude = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--frequency_hz=")) {
      frequency_hz = atof(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      goto fail;
    }
  }

  if (!(g_channel_index == -1 ||
      (1 <= g_channel_index && g_channel_index <= g_num_channels)) ||
      !(0.0 < frequency_hz && frequency_hz < sample_rate_hz / 2)) {
    fprintf(stderr, "Error: Invalid parameters.\n");
    goto fail;
  } else if (!CheckOutputDevice(output_device) ||
             !BuzzerInit(&g_buzzer_state, amplitude, frequency_hz,
                         sample_rate_hz)) {
    goto fail;
  }

  PaStreamParameters output_parameters;
  output_parameters.device = output_device;
  output_parameters.channelCount = g_num_channels;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  PaStream* stream;
  err = Pa_OpenStream(&stream, NULL, &output_parameters, sample_rate_hz,
                      kChunkSize, 0, Callback, NULL);
  if (err != paNoError) { goto fail; }

  err = Pa_StartStream(stream);
  if (err != paNoError) { goto fail; }

  printf("Buzz: amplitude %.6g, frequency %.6g Hz\n", amplitude, frequency_hz);
  printf("\nPress Ctrl+C to stop program.\n\n");
  if (g_channel_index == -1) {
    printf("  all channels\n");
  } else {
    printf("  channel %d\n", g_channel_index);
  }

  is_running = 1;
  signal(SIGINT, StopOnCtrlC);

  while (1) { /* Loop until Ctrl+C is pressed. */
    Pa_Sleep(100);

    if (is_running) {
      fputc('\r', stderr);
    } else {
      fputc('\n', stderr);
      break;
    }

    /* Display "[BUZZ]" when buzz is active, using escape code for color. */
    if (g_buzzer_state.position < sample_rate_hz / 2) {
      fprintf(stderr, "  [\x1b[1;32mBUZZ\x1b[0m]");
    } else {
      fprintf(stderr, "  [    ]");
    }
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) { goto fail; }

  Pa_Terminate();
  free(g_buzzer_state.waveform);
  printf("\nFinished.\n");
  return EXIT_SUCCESS;

fail:
  Pa_Terminate();
  free(g_buzzer_state.waveform);
  if (err != paNoError) {
    fprintf(stderr, "Error: portaudio: %s\n", Pa_GetErrorText(err));
  }
  return EXIT_FAILURE;
}
