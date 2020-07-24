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
 * Play a multichannel WAV file to output device.
 *
 * Use --channels to map tactile signals to output channels. For instance,
 * --channels=3,1,2,2 plays signal 3 on channel 1, signal 1 on channel 2, and
 * signal 2 on channels 3 and 4. A "0" in the channels list means that channel
 * is filled with zeros, e.g. --channels=1,0,2 sets channel 2 to zeros.
 *
 * Flags:
 *  --input=<wavfile>          Input WAV file. The WAV file determines the
 *                             sample rate and number of channels.
 *  --output=<int>             Output device number to play tactor signals to.
 *  --channels=<list>          Channel mapping.
 *  --channel_gains_db=<list>  Gains in dB for each channel. Usually negative
 *                             values, to avoid clipping. More negative value
 *                             means more attenuation, for example -13 is lower
 *                             in level than -10.
 *  --gain_db=<float>          Overall output gain in dB.
 *  --chunk_size=<int>         Frames per PortAudio buffer. (Default 256).
 */

#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "portaudio.h"

#include "audio/dsp/portable/fast_fun.h"
#include "audio/dsp/portable/number_util.h"
#include "audio/dsp/portable/read_wav_file.h"
#include "audio/tactile/channel_map.h"
#include "audio/tactile/portaudio_device.h"
#include "audio/tactile/util.h"

#define kMaxNumTactors 32

/* The main processing loop runs while `is_running` is nonzero. We set a
 * signal handler to change `is_running` to 0 when Ctrl+C is pressed.
 */
static volatile int is_running = 0;
void StopOnCtrlC(int signum /*unused*/) { is_running = 0; }

typedef struct {
  /* PortAudio variables. */
  int pa_initialized;              /* Whether PortAudio was initialized.     */
  PaError pa_error;                /* Last error returned from PortAudio.    */
  PaStream* pa_stream;             /* PortAudio output stream.               */
  ChannelMap channel_map;
  float sample_rate_hz;
  int chunk_size;

  int keep_running;                /* Whether main loop should keep running. */

  float* input_wav_samples;
  int input_wav_size;
  int input_wav_pos;
  int num_input_channels;

  float volume_decay_coeff;
  volatile float volume[kMaxNumTactors];
} Engine;

/* Reads input WAV file. */
int ReadInputWav(Engine* engine, const char* input_wav) {
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;
  int32_t* samples_int32 = ReadWavFile(
      input_wav, &num_samples, &num_channels, &sample_rate_hz);
  if (samples_int32 == NULL) { return 0; }

  const int num_frames = num_samples / num_channels;
  /* Round up to a whole number of chunks to simplify buffering. */
  const int num_frames_padded =
      RoundUpToMultiple(num_frames, engine->chunk_size);
  const int num_samples_padded = num_frames_padded * num_channels;
  engine->input_wav_samples =
      (float*)malloc(sizeof(float) * num_samples_padded);
  if (engine->input_wav_samples == NULL) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return 0;
  }
  engine->input_wav_size = num_samples_padded;
  engine->input_wav_pos = 0;
  engine->num_input_channels = num_channels;
  engine->sample_rate_hz = sample_rate_hz;

  /* Convert to float. */
  int i;
  for (i = 0; i < num_samples; ++i) {
    engine->input_wav_samples[i] = (float)samples_int32[i] / INT32_MAX;
  }
  for (; i < num_samples_padded; ++i) {
    engine->input_wav_samples[i] = 0.0f;
  }

  free(samples_int32);
  return 1;
}

/* Processes one chunk of tactile data. */
void ProcessChunk(Engine* engine, const float* input, float* output) {
  const int num_input_channels = engine->num_input_channels;
  const int chunk_size = engine->chunk_size;
  ChannelMapApply(&engine->channel_map, input, chunk_size, output);

  float volume_accum[kMaxNumTactors] = {0.0f};
  int i;
  int c;
  for (i = 0; i < chunk_size; ++i) {
    /* For visualization, accumulate energy for each tactile signal. */
    for (c = 0; c < num_input_channels; ++c) {
      volume_accum[c] += input[c] * input[c];
    }
    input += num_input_channels;
  }

  for (c = 0; c < num_input_channels; ++c) {
    /* Compute RMS value. */
    const float rms = sqrt(volume_accum[c] / (chunk_size * chunk_size));
    /* Update engine->volume[c] according to
     *   volume = max(rms, volume * volume_decay_coeff).
     * This way the visualization follows the RMS with instantaneous attack but
     * smoothed release, so that onsets are well represented.
     */
    float updated_volume = engine->volume[c] * engine->volume_decay_coeff;
    if (rms > updated_volume) {
      updated_volume = rms;
    }
    engine->volume[c] = updated_volume;
  }
}

/* The audio thread calls this function for every chunk of audio. */
int PortAudioCallback(const void *input_buffer, void *output_buffer,
    unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags,
    void *user_data) {
  if (status_flags & paOutputUnderflow) {
    fprintf(stderr, "Error: Underflow in tactile output. "
        "chunk_size (%lu) might be too small.\n", frames_per_buffer);
  }

  const float* input = (const float*)input_buffer;
  float* output = (float*)output_buffer;
  Engine* engine = (Engine*)user_data;

  /* WAV input. */
  if (engine->input_wav_samples) {
    const int samples_per_buffer =
        engine->num_input_channels * (int)frames_per_buffer;
    if (engine->input_wav_pos + samples_per_buffer > engine->input_wav_size) {
      engine->input_wav_pos = 0;
    }
    input = engine->input_wav_samples + engine->input_wav_pos;
    engine->input_wav_pos += samples_per_buffer;
  }

  ProcessChunk(engine, input, output);

  return paContinue;
}

/* Initializes PortAudio and starts stream. */
int StartPortAudio(Engine* engine, const char* input_wav,
                   const char* output_device) {
  /* Initialize PortAudio. */
  engine->pa_error = Pa_Initialize();
  if (engine->pa_error != paNoError) { return 0; }
  engine->pa_initialized = 1;

  /* Find PortAudio devices. */
  const int output_channels = engine->channel_map.num_output_channels;
  const int output_device_index =
      FindPortAudioDevice(output_device, 0, output_channels);
  if (output_device_index < 0) {
    fprintf(stderr, "\nError: "
        "Use --output flag to set valid devices:\n");
    PrintPortAudioDevices();
    return 0;
  }

  /* Display audio stream configuration. */
  printf("sample rate: %g Hz\n"
         "chunk size: %d frames (%.1f ms)\n",
        engine->sample_rate_hz,
        engine->chunk_size,
        (1000.0f * engine->chunk_size) / engine->sample_rate_hz);

  printf("Input WAV: %s\n", input_wav);
  printf("Output device: #%d %s\n",
         output_device_index, Pa_GetDeviceInfo(output_device_index)->name);
  printf("Output channels:\n");
  ChannelMapPrint(&engine->channel_map);

  /* Open and start PortAudio stream. */
  PaStreamParameters output_parameters;
  output_parameters.device = output_device_index;
  output_parameters.channelCount = output_channels;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  engine->pa_error = Pa_OpenStream(
    &engine->pa_stream, NULL, &output_parameters,
    engine->sample_rate_hz, engine->chunk_size, 0, PortAudioCallback, engine);
  if (engine->pa_error != paNoError) { return 0; }

  engine->pa_error = Pa_StartStream(engine->pa_stream);
  if (engine->pa_error != paNoError) { return 0; }

  return 1;
}

/* Starts up SDL, PortAudio, and tactile processing. */
int EngineInit(Engine* engine, int argc, char** argv) {
  int i;

  engine->pa_initialized = 0;
  engine->pa_error = paNoError;
  engine->pa_stream = NULL;
  engine->input_wav_samples = NULL;
  engine->keep_running = 1;


  const char* input_wav = NULL;
  const char* output_device = NULL;
  const char* source_list = NULL;
  const char* gains_db_list = NULL;
  int chunk_size = 256;
  float global_gain_db = 0.0f;

  for (i = 1; i < argc; ++i) {  /* Parse flags. */
    if (StartsWith(argv[i], "--input=")) {
      input_wav = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--output=")) {
      output_device = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--channels=")) {
      source_list = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--channel_gains_db=")) {
      gains_db_list = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--gain_db=")) {
      global_gain_db = atof(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--chunk_size=")) {
      chunk_size = atoi(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      return 0;
    }
  }

  if (!ReadInputWav(engine, input_wav)) {
    return 0;
  }

  if (!source_list) {
    fprintf(stderr, "Error: Must specify --channels.\n");
    return 0;
  } else if (!ChannelMapParse(engine->num_input_channels,
        source_list, gains_db_list, &engine->channel_map)) {
    return 0;
  }

  engine->chunk_size = chunk_size;

  const float global_gain = DecibelsToAmplitudeRatio(global_gain_db);
  int c;
  for (c = 0; c < engine->channel_map.num_output_channels; ++c) {
    engine->channel_map.channels[c].gain *= global_gain;
  }

  for (c = 0; c < kMaxNumTactors; ++c) {
    engine->volume[c] = 0.0f;
  }
  const float kVolumeMeterTimeConstantSeconds = 0.05;
  engine->volume_decay_coeff = (float)exp(
      -chunk_size / (kVolumeMeterTimeConstantSeconds * engine->sample_rate_hz));

  /* Start PortAudio and audio thread. */
  if (!StartPortAudio(engine, input_wav, output_device)) {
    return 0;
  }

  return 1;
}

/* Cleans everything up. */
void EngineTerminate(Engine* engine) {
  if (engine->pa_stream) {
    engine->pa_error = Pa_CloseStream(engine->pa_stream);
  }

  if (engine->pa_error != paNoError) {
    fprintf(stderr, "Error: PortAudio: %s\n", Pa_GetErrorText(engine->pa_error));
  }
  if (engine->pa_initialized) {
    Pa_Terminate();
  }

  free(engine->input_wav_samples);
}

#define kVolumeMeterWidth 4

void PrintVolumeMeters(const Engine* engine) {
  const int num_tactors = engine->num_input_channels;
  char bar[3 * kVolumeMeterWidth + 1];
  int c;
  fputc('\r', stderr);
  for (c = 0; c < num_tactors; ++c) {
    const float rms = engine->volume[c];
    const float rms_min = 0.001f;
    const float rms_max = 0.04f;
    /* Convert to log scale to decide how much of the bar to fill. */
    float fraction =
        FastLog2(1e-12f + rms / rms_min) / FastLog2(rms_max / rms_min);
    PrettyTextBar(kVolumeMeterWidth, fraction, bar);
    /* Print the bar, using escape codes to color the bar green. */
    fprintf(stderr, "[\x1b[1;32m%s\x1b[0m] ", bar);
  }
}

int main(int argc, char** argv) {
  int exit_status = EXIT_FAILURE;
  Engine engine;
  if (!EngineInit(&engine, argc, argv)) { goto done; }

  printf("\nPress Ctrl+C to stop program.\n\n");

  is_running = 1;
  signal(SIGINT, StopOnCtrlC);

  while (is_running) {  /* Loop until Ctrl+C is pressed. */
    Pa_Sleep(50);
    PrintVolumeMeters(&engine);
  }

  printf("\nFinished.\n");
  exit_status = EXIT_SUCCESS;
done:
  EngineTerminate(&engine);
  return exit_status;
}
