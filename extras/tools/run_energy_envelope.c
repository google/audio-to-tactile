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
 * Demo program for EnergyEnvelope.
 *
 * This is a small program that runs EnergyEnvelope in real time using
 * portaudio. Mono audio is read from a microphone and played as a stereo
 * signal to an output device. The output is meant to be a circuit that we
 * built, which connects with a 3.5mm stereo audio cable, runs through an
 * amplifier, and plays the signals on two tactors.
 *
 * Since the program output is stereo, but the design has three tactors,
 * there are --left and --right flags to select 2 out of 4 tactors to play.
 *
 * See also run_energy_envelope_on_wav, which runs on an input WAV file.
 *
 * Flags:
 *  --input=<name>          Input device to read source audio from.
 *  --output=<name>         Output device to play tactor signals to.
 *  --sample_rate_hz=<int>  Sample rate. Note that most devices only support
 *                          a few standard audio sample rates, e.g. 44100.
 *  --gain=<float>          Output gain. Use >1.0 to make stronger.
 *  --left=<int>            What to play on the left tactor:
 *                            0 = baseband channel,
 *                            1 = vowel channel,
 *                            2 = sh fricative channel,
 *                            3 = fricative channel.
 *  --right=<int>           What to play on the right tactor.
 */

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "extras/tools/portaudio_device.h"
#include "extras/tools/util.h"
#include "src/tactile/energy_envelope.h"
#include "portaudio.h"

/* The main processing loop runs while `is_running` is nonzero. We set a
 * signal handler to change `is_running` to 0 when Ctrl+C is pressed.
 */
static volatile int is_running = 0;
void StopOnCtrlC(int signum /*unused*/) { is_running = 0; }

/* Globals for running the tactors. */
const int kChunkSize = 64;
EnergyEnvelope g_out_left_state;
EnergyEnvelope g_out_right_state;
static volatile float g_volume_meters[2] = {0.0f, 0.0f};
float g_volume_decay_coeff;

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

    g_volume_meters[c] *= g_volume_decay_coeff;
    if (rms > g_volume_meters[c]) {
      g_volume_meters[c] = rms;
    }
  }
}

#define kVolumeMeterWidth 9

void PrintVolumeMeters() {
  char bar[3 * kVolumeMeterWidth + 1];
  int c;
  for (c = 0; c < 2; ++c) {
    float x = g_volume_meters[c];
    /* Convert to log scale to decide how much of the bar to fill. */
    float fraction = -log(x / 0.003) / log(0.003);
    PrettyTextBar(kVolumeMeterWidth, fraction, bar);
    /* Print the bar, using escape codes to color the bar green. */
    fprintf(stderr, "  [\x1b[1;32m%s\x1b[0m] %.3f", bar, x);
  }
}

/* Stream callback function. In each call, portaudio passes kChunkSize frames
 * of input, and we process it to produce kChunkSize frames of output.
 */
int TactorCallback(const void *input_buffer, void *output_buffer,
                   unsigned long frames_per_buffer,
                   const PaStreamCallbackTimeInfo* time_info,
                   PaStreamCallbackFlags status_flags,
                   void *user_data) {
  const float* input = (const float*) input_buffer;
  float* output = (float*) output_buffer;

  if (input_buffer == NULL) {
    int i;
    for (i = 0; i < frames_per_buffer; ++i) {
      *output++ = 0.0f;
      *output++ = 0.0f;
    }
  } else {
    /* Process the tactors. */
    EnergyEnvelopeProcessSamples(
        &g_out_left_state, input, frames_per_buffer, output, 2);
    EnergyEnvelopeProcessSamples(
        &g_out_right_state, input, frames_per_buffer, output + 1, 2);

    UpdateVolumeMeters(output, frames_per_buffer);
  }

  return paContinue;
}

int main(int argc, char** argv) {
  PaError err = Pa_Initialize();
  if (err != paNoError) goto fail;

  int output_selection[2] = {1, 3};
  const char* input_device = "default";
  const char* output_device = "default";
  int sample_rate_hz = 44100;
  float gain = 1.0f;
  int i;

  for (i = 1; i < argc; ++i) {  /* Parse flags. */
    if (StartsWith(argv[i], "--input=")) {
      input_device = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output=")) {
      output_device = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--sample_rate_hz=")) {
      sample_rate_hz = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--gain=")) {
      gain = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--left=")) {
      output_selection[0] = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--right=")) {
      output_selection[1] = atoi(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      Pa_Terminate();
      return EXIT_FAILURE;
    }
  }
  /* Find PortAudio devices. */
  const int input_device_index = FindPortAudioDevice(input_device, 1, 0);
  const int output_device_index = FindPortAudioDevice(output_device, 0, 2);
  if (input_device_index < 0 || output_device_index < 0) {
    fprintf(stderr, "\nError: "
        "Use --input and --output flags to set valid devices:\n");
    PrintPortAudioDevices();
    Pa_Terminate();
    return EXIT_FAILURE;
  }

  printf("Input device: #%d %s\n",
         input_device_index, Pa_GetDeviceInfo(input_device_index)->name);
  printf("Output device: #%d %s\n",
         output_device_index, Pa_GetDeviceInfo(output_device_index)->name);

  struct {
    const char* name;
    const EnergyEnvelopeParams* params;
  } OutputChannels[4] = {
      {"baseband", &kEnergyEnvelopeBasebandParams},
      {"vowel", &kEnergyEnvelopeVowelParams},
      {"sh fricative", &kEnergyEnvelopeShFricativeParams},
      {"fricative", &kEnergyEnvelopeFricativeParams}};

  /* Initialize tactor processing. */
  if (!EnergyEnvelopeInit(&g_out_left_state,
                          OutputChannels[output_selection[0]].params,
                          sample_rate_hz, 1) ||
      !EnergyEnvelopeInit(&g_out_right_state,
                          OutputChannels[output_selection[1]].params,
                          sample_rate_hz, 1)) {
    fprintf(stderr, "Error: EnergyEnvelopeInit failed.\n");
    Pa_Terminate();
    return EXIT_FAILURE;
  }
  g_out_left_state.output_gain *= gain;
  g_out_right_state.output_gain *= gain;

  const float kVolumeMeterTimeConstantSeconds = 0.05;
  g_volume_decay_coeff = (float) exp(
      -kChunkSize / (kVolumeMeterTimeConstantSeconds * sample_rate_hz));

  PaStreamParameters input_parameters;
  input_parameters.device = input_device_index;
  input_parameters.channelCount = 1;
  input_parameters.sampleFormat = paFloat32;
  input_parameters.suggestedLatency =
      Pa_GetDeviceInfo(input_parameters.device)->defaultLowInputLatency;
  input_parameters.hostApiSpecificStreamInfo = NULL;

  PaStreamParameters output_parameters;
  output_parameters.device = output_device_index;
  output_parameters.channelCount = 2;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  PaStream *stream;
  err = Pa_OpenStream(
    &stream, &input_parameters, &output_parameters,
    sample_rate_hz, kChunkSize, 0, TactorCallback, NULL);
  if (err != paNoError) goto fail;

  err = Pa_StartStream(stream);
  if (err != paNoError) goto fail;

  printf("\nPress Ctrl+C to stop program.\n\n");
  printf("  %-19s%s\n", OutputChannels[output_selection[0]].name,
         OutputChannels[output_selection[1]].name);

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

  Pa_Terminate();
  printf("\nFinished.\n");
  return EXIT_SUCCESS;

fail:
  Pa_Terminate();
  fprintf(stderr, "Error: portaudio: %s\n", Pa_GetErrorText(err));
  return EXIT_FAILURE;
}
