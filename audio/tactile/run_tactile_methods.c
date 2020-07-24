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
 * Demo program for different tactile methods.
 *
 * This is a small program that runs Bratakos2001 and Yuan2005 in real time
 * using PortAudio. Mono audio is read from a mic and played to an output
 * device. Note that the number of output channels depends on the method:
 *
 *   Method         Output channels
 *   Bratakos2001                 1
 *   Yuan2005                     2
 *   EnergyEnvelope               4
 *
 * Use --channels to map tactile signals to output channels. For instance,
 * --channels=3,1,2,2 plays signal 3 on channel 1, signal 1 on channel 2, and
 * signal 2 on channels 3 and 4. A "0" in the channels list means that channel
 * is filled with zeros, e.g. --channels=1,0,2 sets channel 2 to zeros.
 *
 * Flags:
 *  --method=<name>            'Bratakos2001', 'Yuan2005', or 'EnergyEnvelope'.
 *  --input=<name>             Input device to read source audio from.
 *  --output=<name>            Output device to play tactor signals to.
 *  --sample_rate_hz=<int>     Sample rate. Note that most devices only support
 *                             a few standard audio sample rates, e.g. 44100.
 *  --channels=<list>          Channel mapping.
 *  --channel_gains_db=<list>  Gains in dB for each channel. Usually negative
 *                             values, to avoid clipping. More negative value
 *                             means more attenuation, for example -13 is lower
 *                             in level than -10.
 *  --gain_db=<float>          Overall output gain in dB.
 *  --chunk_size=<int>         Frames per PortAudio buffer. (Default 256).
 *  --agc_strength=<float>     (Optional) Run auto gain control before tactile
 *                             processing with specified strength in [0, 1].
 *
 *  --low_denoising_threshold=<float>    Denoising threshold for the low channel
 *                                       (default 1e-4).
 *  --high_denoising_threshold=<float>   Threshold for the high channel.
 */

#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "portaudio.h"

#include "audio/dsp/portable/auto_gain_control.h"
#include "audio/tactile/energy_envelope/energy_envelope.h"
#include "audio/tactile/references/bratakos2001/bratakos2001.h"
#include "audio/tactile/references/yuan2005/yuan2005.h"
#include "audio/tactile/channel_map.h"
#include "audio/tactile/portaudio_device.h"
#include "audio/tactile/util.h"

#define kMaxTactors 10

/* The main processing loop runs while `is_running` is nonzero. We set a
 * signal handler to change `is_running` to 0 when Ctrl+C is pressed.
 */
static volatile int is_running = 0;
void StopOnCtrlC(int signum /*unused*/) { is_running = 0; }

struct {
  int chunk_size;
  ChannelMap channel_map;
  AutoGainControlState agc;
  float* agc_output;
  float* tactile_output;

  void (*method_fun)(const float*, int, float*);
  EnergyEnvelope channel_states[4];
  Bratakos2001State bratakos2001;
  Yuan2005State yuan2005;

  volatile float volume_meters[kMaxTactors];
  float volume_decay_coeff;
} engine;

void UpdateVolumeMeters(const float* output, int frames_per_buffer) {
  const int num_tactors = engine.channel_map.num_input_channels;
  float accum[kMaxTactors];
  int c;
  for (c = 0; c < num_tactors; ++c) {
    accum[c] = 0.0f;
  }

  int i;
  for (i = 0; i < frames_per_buffer; ++i) {
    for (c = 0; c < num_tactors; ++c) {
      accum[c] += output[num_tactors * i + c] * output[num_tactors * i + c];
    }
  }

  for (c = 0; c < num_tactors; ++c) {
    /* Compute RMS value of each output channel in the current chunk. */
    const float rms = sqrt(accum[c] / frames_per_buffer);

    engine.volume_meters[c] *= engine.volume_decay_coeff;
    if (rms > engine.volume_meters[c]) {
      engine.volume_meters[c] = rms;
    }
  }
}

#define kVolumeMeterWidth 4

void PrintVolumeMeters() {
  const int num_tactors = engine.channel_map.num_input_channels;
  char bar[3 * kVolumeMeterWidth + 1];
  int c;
  for (c = 0; c < num_tactors; ++c) {
    float x = engine.volume_meters[c];
    /* Convert to log scale to decide how much of the bar to fill. */
    float fraction = -log(x / 0.003) / log(0.003);
    PrettyTextBar(kVolumeMeterWidth, fraction, bar);
    /* Print the bar, using escape codes to color the bar green. */
    fprintf(stderr, "[\x1b[1;32m%s\x1b[0m] ", bar);
  }
}

/*void RunTactileProcessor(const float* input, int num_samples, float* output) {
  const int block_size = CarlFrontendBlockSize(
      engine.tactile_processor->frontend);
  const int num_blocks = num_samples / block_size;
  int b;
  for (b = 0; b < num_blocks; ++b) {
    TactileProcessorProcessSamples(
        engine.tactile_processor, input, output);
    input += block_size;
    output += kTactileProcessorNumtactors * block_size;
  }
}*/

void RunEnergyEnvelope(const float* input, int num_samples, float* output) {
  int c;
  for (c = 0; c < 4; ++c) {
    EnergyEnvelopeProcessSamples(&engine.channel_states[c], input, num_samples,
                                 output + c, 4);
  }
}

void RunBratakos2001(const float* input, int num_samples, float* output) {
  Bratakos2001ProcessSamples(&engine.bratakos2001, input, num_samples,
                             output);
}

void RunYuan2005(const float* input, int num_samples, float* output) {
  Yuan2005ProcessSamples(&engine.yuan2005, input, num_samples, output);
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
    engine.agc_output[i] = input[i] * AutoGainControlGetGain(&engine.agc);
  }

  engine.method_fun(engine.agc_output, frames_per_buffer,
                    engine.tactile_output);

  UpdateVolumeMeters(engine.tactile_output, frames_per_buffer);
  ChannelMapApply(&engine.channel_map, engine.tactile_output, frames_per_buffer,
                  output);

  return paContinue;
}

int main(int argc, char** argv) {
  engine.agc_output = NULL;
  engine.tactile_output = NULL;
  int i;
  for (i = 0; i < kMaxTactors; ++i) {
    engine.volume_meters[i] = 0.0f;
  }

  PaError err = Pa_Initialize();
  if (err != paNoError) goto fail;

  const char* method = NULL;
  const char* input_device = NULL;
  const char* output_device = NULL;
  const char* source_list = NULL;
  const char* gains_db_list = NULL;
  int sample_rate_hz = 44100;
  int chunk_size = 256;
  float global_gain_db = 0.0f;
  float agc_strength = 0.0f;
  int num_tactors = 0;

  Yuan2005Params yuan2005_params;
  Yuan2005SetDefaultParams(&yuan2005_params);

  for (i = 1; i < argc; ++i) {  /* Parse flags. */
    if (StartsWith(argv[i], "--method=")) {
      method = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--input=")) {
      input_device = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output=")) {
      output_device = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--sample_rate_hz=")) {
      sample_rate_hz = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--channels=")) {
      source_list = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--channel_gains_db=")) {
      gains_db_list = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--gain_db=")) {
      global_gain_db = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--chunk_size=")) {
      chunk_size = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--agc_strength=")) {
      agc_strength = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--low_denoising_threshold=")) {
      yuan2005_params.low_channel.denoising_threshold =
          atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--high_denoising_threshold=")) {
      yuan2005_params.high_channel.denoising_threshold =
          atof(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      goto fail;
    }
  }

  if (!method) {
    fprintf(stderr, "Error: Must specify --method.\n");
    goto fail;
  } else if (StringEqualIgnoreCase(method, "Bratakos2001")) {
    printf("method: Bratakos2001\n");
    engine.method_fun = RunBratakos2001;
    num_tactors = 1;

    if (!Bratakos2001Init(&engine.bratakos2001, sample_rate_hz)) {
      goto fail;
    }
  } else if (StringEqualIgnoreCase(method, "Yuan2005")) {
    printf("method: Yuan2005\n");
    engine.method_fun = RunYuan2005;
    num_tactors = 2;

    if (!Yuan2005Init(&engine.yuan2005, &yuan2005_params)) {
      goto fail;
    }
  } else if (StringEqualIgnoreCase(method, "EnergyEnvelope")) {
    printf("method: EnergyEnvelope\n");
    engine.method_fun = RunEnergyEnvelope;
    num_tactors = 4;

    if (!EnergyEnvelopeInit(&engine.channel_states[0],
                            &kEnergyEnvelopeBasebandParams,
                            sample_rate_hz, 1) ||
        !EnergyEnvelopeInit(&engine.channel_states[1],
                            &kEnergyEnvelopeVowelParams,
                            sample_rate_hz, 1) ||
        !EnergyEnvelopeInit(&engine.channel_states[2],
                            &kEnergyEnvelopeShFricativeParams,
                            sample_rate_hz, 1) ||
        !EnergyEnvelopeInit(&engine.channel_states[3],
                            &kEnergyEnvelopeFricativeParams,
                            sample_rate_hz, 1)) {
      fprintf(stderr, "Error: EnergyEnvelopeInit failed.\n");
      goto fail;
    }
  } else {
    fprintf(stderr, "Error: Invalid method \"%s\".\n", method);
    goto fail;
  }

  if (!source_list) {
    fprintf(stderr, "Error: Must specify --channels.\n");
    goto fail;
  } else if (!ChannelMapParse(num_tactors, source_list, gains_db_list,
                              &engine.channel_map)) {
    goto fail;
  }

  const float global_gain = DecibelsToAmplitudeRatio(global_gain_db);
  for (i = 0; i < engine.channel_map.num_output_channels; ++i) {
    engine.channel_map.channels[i].gain *= global_gain;
  }

  /* Find PortAudio devices. */
  const int input_device_index = FindPortAudioDevice(input_device, 1, 0);
  const int output_device_index = FindPortAudioDevice(
      output_device, 0, engine.channel_map.num_output_channels);
  if (input_device_index < 0 || output_device_index < 0) {
    fprintf(stderr, "\nError: "
        "Use --input and --output flags to set valid devices:\n");
    PrintPortAudioDevices();
    goto fail;
  }

  /* Display audio stream configuration. */
  printf("sample rate: %d Hz\n"
         "chunk size: %d frames (%.1f ms)\n",
         sample_rate_hz, chunk_size, (1000.0f * chunk_size) / sample_rate_hz);
  printf("Input device: #%d %s\n",
         input_device_index, Pa_GetDeviceInfo(input_device_index)->name);
  printf("Output device: #%d %s\n",
         output_device_index, Pa_GetDeviceInfo(output_device_index)->name);
  printf("Output channels:\n");
  ChannelMapPrint(&engine.channel_map);

  /* Initialize processing. */
  engine.agc_output = (float*)malloc(chunk_size * sizeof(float));
  if (engine.agc_output == NULL) { goto fail; }
  engine.tactile_output = (float*)malloc(
      num_tactors * chunk_size * sizeof(float));
  if (engine.tactile_output == NULL) { goto fail; }

  if (!AutoGainControlInit(&engine.agc,
                           sample_rate_hz,
                           /*time_constant_s=*/3.0f,
                           agc_strength,
                           /*power_floor=*/1e-6f)) {
    fprintf(stderr, "Error: AGC initialization failed.\n");
    goto fail;
  }

  const float kVolumeMeterTimeConstantSeconds = 0.05;
  engine.volume_decay_coeff = (float)exp(
      -chunk_size / (kVolumeMeterTimeConstantSeconds * sample_rate_hz));

  PaStreamParameters input_parameters;
  input_parameters.device = input_device_index;
  input_parameters.channelCount = 1;
  input_parameters.sampleFormat = paFloat32;
  input_parameters.suggestedLatency =
      Pa_GetDeviceInfo(input_parameters.device)->defaultLowInputLatency;
  input_parameters.hostApiSpecificStreamInfo = NULL;

  PaStreamParameters output_parameters;
  output_parameters.device = output_device_index;
  output_parameters.channelCount = engine.channel_map.num_output_channels;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  PaStream *stream;
  err = Pa_OpenStream(&stream, &input_parameters, &output_parameters,
                      sample_rate_hz, chunk_size, 0, TactorCallback, NULL);
  if (err != paNoError) { goto fail; }

  err = Pa_StartStream(stream);
  if (err != paNoError) { goto fail; }

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

  free(engine.tactile_output);
  free(engine.agc_output);
  Pa_Terminate();
  printf("\nFinished.\n");
  return EXIT_SUCCESS;

fail:
  if (err != paNoError) {
    fprintf(stderr, "Error: portaudio: %s\n", Pa_GetErrorText(err));
  }
  free(engine.tactile_output);
  free(engine.agc_output);
  Pa_Terminate();
  return EXIT_FAILURE;
}

