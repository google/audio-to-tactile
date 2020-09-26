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
 * Demo program for Yuan2005.
 *
 * This is a small program that runs Yuan2005 in real time using PortAudio. Mono
 * audio is read from a mic and played as a stereo signal to an output device.
 *
 * Flags:
 *  --input=<int>           Input device number to read source audio from.
 *  --output=<int>          Output device number to play tactor signals to.
 *  --sample_rate_hz=<int>  Sample rate. Note that most devices only support
 *                          a few standard audio sample rates, e.g. 44100.
 *  --chunk_size=<int>      Number of audio frames per buffer (default 256).
 *  --agc_strength=<float>  (Optional) Run auto gain control before tactile
 *                          processing with specified strength in [0, 1].
 *
 *  --low_denoising_threshold=<float>    Denoising threshold for the low channel
 *                                       (default 1e-4).
 *  --high_denoising_threshold=<float>   Threshold for the high channel.
 */

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "portaudio.h"

#include "dsp/auto_gain_control.h"
#include "tactile/references/yuan2005/yuan2005.h"
#include "tactile/util.h"

/* The main processing loop runs while `is_running` is nonzero. We set a
 * signal handler to change `is_running` to 0 when Ctrl+C is pressed.
 */
static volatile int is_running = 0;
void StopOnCtrlC(int signum /*unused*/) { is_running = 0; }

struct {
  AutoGainControlState agc;
  float* buffer;
  Yuan2005State eoa;
  volatile float volume_meters[2];
  float volume_decay_coeff;
} engine;

/* Check that selected input and output devices are valid, and if not, print
 * list of available devices.
 */
int CheckInputOutputDevices(int input_device, int output_device) {
  const int num_devices = Pa_GetDeviceCount();
  if (!(0 <= input_device && input_device < num_devices) ||
      !(0 <= output_device && output_device < num_devices)) {
    printf("\nError: Use --input and --output flags to set valid devices:\n");
    int i;
    for (i = 0; i < num_devices; ++i) {
      printf("#%-2d %s\n", i, Pa_GetDeviceInfo(i)->name);
    }
    return 0;
  }
  printf("Input device: #%d %s\n",
         input_device, Pa_GetDeviceInfo(input_device)->name);
  printf("Output device: #%d %s\n",
         output_device, Pa_GetDeviceInfo(output_device)->name);
  return 1;
}

void UpdateVolumeMeters(const float* output, int frames_per_buffer) {
  float accum[2] = {0.0f, 0.0f};
  int i;
  int c;
  for (i = 0; i < frames_per_buffer; ++i) {
    for (c = 0; c < 2; ++c) {
      accum[c] += output[2 * i + c] * output[2 * i + c];
    }
  }

  for (c = 0; c < 2; ++c) {
    /* Compute RMS value of each output channel in the current chunk. */
    const float rms = sqrt(accum[c] / frames_per_buffer);

    engine.volume_meters[c] *= engine.volume_decay_coeff;
    if (rms > engine.volume_meters[c]) {
      engine.volume_meters[c] = rms;
    }
  }
}

#define kVolumeMeterWidth 9

void PrintVolumeMeters() {
  char bar[3 * kVolumeMeterWidth + 1];
  int c;
  for (c = 0; c < 2; ++c) {
    float x = engine.volume_meters[c];
    /* Convert to log scale to decide how much of the bar to fill. */
    float fraction = -log(x / 0.003) / log(0.003);
    PrettyTextBar(kVolumeMeterWidth, fraction, bar);
    /* Print the bar, using escape codes to color the bar green. */
    fprintf(stderr, "  [\x1b[1;32m%s\x1b[0m] %.3f", bar, x);
  }
}

/* Stream callback function. In each call, portaudio passes chunk_size frames
 * of input, and we process it to produce chunk_size frames of output.
 */
int TactorCallback(const void *input_buffer, void *output_buffer,
                   unsigned long frames_per_buffer,
                   const PaStreamCallbackTimeInfo* time_info,
                   PaStreamCallbackFlags status_flags,
                   void *user_data) {
  const float* input = (const float*) input_buffer;
  float* output = (float*) output_buffer;

  int i;
  for (i = 0; i < frames_per_buffer; ++i) {
    AutoGainControlProcessSample(&engine.agc, input[i] * input[i]);
    engine.buffer[i] = input[i] * AutoGainControlGetGain(&engine.agc);
  }

  Yuan2005ProcessSamples(&engine.eoa, engine.buffer, frames_per_buffer, output);
  UpdateVolumeMeters(output, frames_per_buffer);
  return paContinue;
}

int main(int argc, char** argv) {
  engine.buffer = NULL;
  engine.volume_meters[0] = 0.0f;
  engine.volume_meters[1] = 0.0f;

  PaError err = Pa_Initialize();
  if (err != paNoError) goto fail;

  int input_device = Pa_GetDefaultInputDevice();
  int output_device = Pa_GetDefaultOutputDevice();
  int chunk_size = 256;
  float agc_strength = 0.0f;
  Yuan2005Params params;
  Yuan2005SetDefaultParams(&params);
  params.sample_rate_hz = 44100;
  int i;

  for (i = 1; i < argc; ++i) {  /* Parse flags. */
    if (StartsWith(argv[i], "--input=")) {
      input_device = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--output=")) {
      output_device = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--sample_rate_hz=")) {
      params.sample_rate_hz = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--chunk_size=")) {
      chunk_size = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--agc_strength=")) {
      agc_strength = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--low_denoising_threshold=")) {
      params.low_channel.denoising_threshold = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--high_denoising_threshold=")) {
      params.high_channel.denoising_threshold = atof(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      Pa_Terminate();
      return EXIT_FAILURE;
    }
  }

  if (!CheckInputOutputDevices(input_device, output_device)) {
    Pa_Terminate();
    return EXIT_FAILURE;
  }

  /* Initialize processing. */
  engine.buffer = (float*)malloc(chunk_size * sizeof(float));
  if (engine.buffer == NULL) {
    Pa_Terminate();
    return EXIT_FAILURE;
  }

  if (!AutoGainControlInit(&engine.agc,
                           params.sample_rate_hz,
                           /*time_constant_s=*/3.0f,
                           agc_strength,
                           /*power_floor=*/1e-6f) ||
      !Yuan2005Init(&engine.eoa, &params)) {
    fprintf(stderr, "Error: Initialization failed.\n");
    Pa_Terminate();
    return EXIT_FAILURE;
  }

  const float kVolumeMeterTimeConstantSeconds = 0.05;
  engine.volume_decay_coeff = (float)exp(
      -chunk_size / (kVolumeMeterTimeConstantSeconds * params.sample_rate_hz));

  PaStreamParameters input_parameters;
  input_parameters.device = input_device;
  input_parameters.channelCount = 1;
  input_parameters.sampleFormat = paFloat32;
  input_parameters.suggestedLatency =
      Pa_GetDeviceInfo(input_parameters.device)->defaultLowInputLatency;
  input_parameters.hostApiSpecificStreamInfo = NULL;

  PaStreamParameters output_parameters;
  output_parameters.device = output_device;
  output_parameters.channelCount = 2;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  PaStream *stream;
  err = Pa_OpenStream(
    &stream, &input_parameters, &output_parameters,
    params.sample_rate_hz, chunk_size, 0, TactorCallback, NULL);
  if (err != paNoError) goto fail;

  err = Pa_StartStream(stream);
  if (err != paNoError) goto fail;

  printf("\nPress Ctrl+C to stop program.\n\n");

  is_running = 1;
  signal(SIGINT, StopOnCtrlC);

  while (1) {  /* Loop until Ctrl+C is pressed. */
    Pa_Sleep(50);

    if (is_running) {
      fputc('\r', stderr);
    } else {
      fputc('\n', stderr);
      break;
    }

    PrintVolumeMeters();
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) goto fail;

  free(engine.buffer);
  Pa_Terminate();
  printf("\nFinished.\n");
  return EXIT_SUCCESS;

fail:
  free(engine.buffer);
  Pa_Terminate();
  fprintf(stderr, "Error: portaudio: %s\n", Pa_GetErrorText(err));
  return EXIT_FAILURE;
}

