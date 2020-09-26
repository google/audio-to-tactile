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

#include "tactile/references/taps/tactophone.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsp/logging.h"
#include "tactile/references/taps/phoneme_code.h"
#include "tactile/references/taps/tactophone_engine.h"

/* Checks that all phoneme strings in TactophoneLessonSet are valid. */
static void CheckLessonSet(TactophoneLessonSet* lesson_set) {
  int i;
  for (i = 0; i < lesson_set->num_lessons; ++i) {
    const TactophoneLesson* lesson = &lesson_set->lessons[i];
    int j;
    for (j = 0; j < lesson->num_questions; ++j) {
      const TactophoneQuestion* question = &lesson->questions[j];
      int k;
      for (k = 0; k < question->num_choices; ++k) {
        const TactophoneChoice* choice = &question->choices[k];
        if (!PhonemeStringIsValid(choice->phonemes)) {
          fprintf(stderr,
                  "Invalid phonemes \"%s\" in lesson %s, "
                  "question %d, choice %d \"%s\".\n",
                  choice->phonemes, lesson->name, j + 1, k + 1, choice->label);
          exit(1);
        }
      }
    }
  }
}

/* Checks that error is `paNoError`, dies with an error message otherwise.
 * NOTE: To ensure error message is visible, this function must be called
 * outside of when ncurses is active.
 */
static void CheckPortaudioNoError(PaError error) {
  if (error != paNoError) {
    Pa_Terminate();
    fprintf(stderr, "Error: portaudio: %s\n", Pa_GetErrorText(error));
    exit(EXIT_FAILURE);
  }
}

/* Checks that selected output device is valid, and if not, prints list of
 * available devices along with their max number of output channels.
 * NOTE: To ensure printed list is visible, this function must be called
 * outside of when ncurses is active.
 */
static void CheckOutputDevice(int output_device) {
  const int num_devices = Pa_GetDeviceCount();
  if (!(0 <= output_device && output_device < num_devices)) {
    fprintf(stderr, "\nError: Use --output flag to set a valid device:\n");
    int i;
    for (i = 0; i < num_devices; ++i) {
      const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
      fprintf(stderr, "#%-2d %-45s channels: %d\n", i, device_info->name,
              device_info->maxOutputChannels);
    }
    Pa_Terminate();
    exit(EXIT_FAILURE);
  }
}

/* Portaudio stream callback function. */
static int PortaudioStreamCallback(const void* input_buffer,
                                   void* output_buffer,
                                   unsigned long frames_per_buffer,
                                   const PaStreamCallbackTimeInfo* time_info,
                                   PaStreamCallbackFlags status_flags,
                                   void* user_data) {
  float* output = (float*)output_buffer;
  TactophoneEngine* engine = (TactophoneEngine*)user_data;

  TactilePlayerFillBuffer(engine->tactile_player, frames_per_buffer, output);
  return (engine->keep_running) ? paContinue : paComplete;
}

void Tactophone(const TactophoneParams* params) {
  srand(time(NULL));  /* Seed random number generator. */
  TactophoneEngine engine;
  TactophoneEngineInit(&engine);

  CHECK(ChannelMapParse(kNumChannels, params->channel_source_list,
        params->channel_gains_db_list, &engine.channel_map));
  ChannelMapPrint(&engine.channel_map);
  const int num_output_channels = engine.channel_map.num_output_channels;
  if (num_output_channels != kNumChannels) {
    fprintf(stderr, "Error: Must have %d output channels.\n", kNumChannels);
    exit(1);
  }
  engine.lesson_set =
      CHECK_NOTNULL(TactophoneReadLessonSet(params->lessons_file));
  CheckLessonSet(engine.lesson_set);
  engine.log_file = CHECK_NOTNULL(fopen(params->log_file, "a"));


  /* Set up portaudio. */
  CheckPortaudioNoError(Pa_Initialize());
  CheckOutputDevice(params->output_device);

  PaStreamParameters output_parameters;
  output_parameters.device = params->output_device;
  output_parameters.channelCount = num_output_channels;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  const int kFramesPerBuffer = 64;
  CheckPortaudioNoError(Pa_OpenStream(
      &engine.portaudio_stream, NULL, &output_parameters, kSampleRateHz,
      kFramesPerBuffer, 0, PortaudioStreamCallback, (void*)&engine));
  CheckPortaudioNoError(Pa_StartStream(engine.portaudio_stream));

  /* Set up ncurses. */
  initscr();
  raw();                /* Disable line buffering. */
  keypad(stdscr, TRUE); /* Interpret arrow keys, etc. */
  noecho();             /* Don't echo keyboard input. */
  timeout(0);           /* Don't wait for key press. */

  if (has_colors() == TRUE) {
    start_color();
    init_pair(kColorTitle, COLOR_BLUE, COLOR_WHITE);
    init_pair(kColorKey, COLOR_BLUE, COLOR_BLACK);
    init_pair(kColorHighlight, COLOR_YELLOW, COLOR_BLACK);
    init_pair(kColorGreen, COLOR_GREEN, COLOR_BLACK);
    init_pair(kColorRed, COLOR_RED, COLOR_BLACK);
  }

  /* Start the engine. */
  TactophoneLog(&engine, "StartTactophone lessons_file=%s, "
                "sample_rate=%dHz, phoneme_spacing=%.3fs, clock_tick=%dms",
                params->lessons_file, kSampleRateHz,
                kPhonemeSpacingInSeconds, kMillisecondsPerClockTick);
  TactophoneSetState(&engine, params->initial_state);
  do {
    Pa_Sleep(kMillisecondsPerClockTick);
  } while (TactophoneEngineRun(&engine, getch()));

  TactophoneLog(&engine, "QuitTactophone");

  while (Pa_IsStreamActive(engine.portaudio_stream)) {}

  /* Clean up ncurses. */
  endwin();

  /* Clean up portaudio. */
  Pa_CloseStream(engine.portaudio_stream);
  Pa_Terminate();

  TactophoneEngineFree(&engine);
}
