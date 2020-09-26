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
 * Tactometer - program for measuring acuteness of tactile senses.
 *
 * Flags:
 *  --output=<name>         Output device to play tactor signals to.
 *  --sample_rate_hz=<int>  Sample rate. Note that most devices only support
 *                          a few standard audio sample rates, e.g. 44100.
 *  --chunk_size=<int>      Number of frames per buffer in audio callback.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "portaudio.h"

#include "dsp/logging.h"
#include "dsp/math_constants.h"
#include "sdl/basic_sdl_app.h"
#include "sdl/draw_text.h"
#include "sdl/window_icon.h"
#include "tactile/portaudio_device.h"
#include "tactile/util.h"

#define kMaxPlotPoints 128

static const float kFrequenciesHz[] = {
  12.5f, 25.0f, 50.0f, 75.0f, 100.0f, 125.0f, 150.0f, 200.0f, 250.0f, 300.0f,
  350.0f, 400.0f, 500.0f, 600.0f};
#define kNumFrequencies (sizeof(kFrequenciesHz) / sizeof(*kFrequenciesHz))

static const SDL_Scancode kKeytarNoteKeys[kNumFrequencies] = {
    SDL_SCANCODE_A, SDL_SCANCODE_Z, SDL_SCANCODE_S, SDL_SCANCODE_X,
    SDL_SCANCODE_D, SDL_SCANCODE_C, SDL_SCANCODE_F, SDL_SCANCODE_V,
    SDL_SCANCODE_G, SDL_SCANCODE_B, SDL_SCANCODE_H, SDL_SCANCODE_N,
    SDL_SCANCODE_J, SDL_SCANCODE_M,
};

static const int kScreenWidth = 640;
static const int kScreenHeight = 640;
static const float kMaxFrequencyHz = 600.0f;
static const float kMinAmplitude = 0.001f;
static const float kMinAmplitudeDb = -60.0f;

struct Buzzer {
  /* For communication from main thread to audio thread, these variables are
   * atomics. SDL atomics hold an int, so we use fixed point representation
   * to approximate float values.
   */
  /* Target frequency with 4 fractional bits. */
  SDL_atomic_t frequency_hz_atomic;
  /* Target amplitude with 14 fractional bits. */
  SDL_atomic_t target_amplitude_atomic;

  /* Only manipulated by the audio thread. */
  float rotator[2];
  float phasor[2];
  float smoothed_amplitude_stage1;
  float smoothed_amplitude;
};
typedef struct Buzzer Buzzer;

struct Engine {
  /* Portaudio variables. */
  int pa_initialized;              /* Whether portaudio was initialized.     */
  PaError pa_error;                /* Last error returned from portaudio.    */
  PaStream* pa_stream;             /* Portaudio output stream.               */
  char* output_device;             /* Portaudio output device.               */
  int sample_rate_hz;              /* Audio sample rate.                     */
  int chunk_size;                  /* Frames per buffer in audio callback.   */

  BasicSdlApp app;                 /* SDL variables.                         */

  /* Sine wave oscillator. */
  float frequency_hz;              /* Frequency of the sine wave to produce. */
  int selected_frequency_index;    /* Frequency as an index in kFrequencies. */
  float amplitude;                 /* Amplitude for the sine wave.           */
  float amplitude_db;              /* `amplitude`, but in decibels.          */
  float amplitude_smoother;        /* Coefficient for amplitude smoothing.   */
  Buzzer buzzer;                   /* Oscillator state.                      */

  /* Basic UI logic. */
  int keep_running;                /* Whether main loop should keep running. */
  void (*fun)(int);                /* Function for current UI state.         */

  /* Variables for Bekesy's tracking method. */
  float bekesy_prev_reversal_db;   /* Previous reverals amplitude in dB.     */
  float bekesy_midpoint_db_sum;    /* Sum of reversal midpoints in dB.       */
  int bekesy_num_reversals;        /* Number of midpoints in the sum.        */
  int prev_space_pressed;          /* Whether space was previously held.     */
  int plot_points[kMaxPlotPoints]; /* Plot of recent amplitude.              */
  int num_plot_points;             /* Number of points in `plot_points`.     */
  /* Array of amplitude thresholds in dB determined by Bekesy's method. */
  float bekesy_thresholds_db[kNumFrequencies];
} engine;

/* Initializes buzzer to zero state. */
void BuzzerInit(Buzzer* buzzer) {
  SDL_AtomicSet(&buzzer->frequency_hz_atomic, 0);
  SDL_AtomicSet(&buzzer->target_amplitude_atomic, 0);
  buzzer->rotator[0] = 1.0f;
  buzzer->rotator[1] = 0.0f;
  buzzer->phasor[0] = 1.0f;
  buzzer->phasor[1] = 0.0f;
  buzzer->smoothed_amplitude_stage1 = 0.0f;
  buzzer->smoothed_amplitude = 0.0f;
}

/* Sets buzzer frequency safely through the atomic. */
void BuzzerSetFrequency(Buzzer* buzzer, float frequency_hz) {
  /* Avoid overflow. */
  assert(frequency_hz / 16 < INT_MAX);
  /* Convert to fixed point with 4 fractional bits. */
  const int fixed = (int)(frequency_hz * 16 + 0.5f);
  SDL_AtomicSet(&buzzer->frequency_hz_atomic, fixed);
}

/* Sets buzzer amplitude safely through the atomic. */
void BuzzerSetAmplitude(Buzzer* buzzer, float target_amplitude) {
  /* Avoid overflow. */
  assert(target_amplitude / 16384 < INT_MAX);
  /* Convert to fixed point with 14 fractional bits. */
  const int fixed = (int)(target_amplitude * 16384 + 0.5f);
  SDL_AtomicSet(&buzzer->target_amplitude_atomic, fixed);
}

/* The portaudio callback, runs the sine wave oscillator. */
int PortaudioCallback(const void* input_buffer, void* output_buffer,
    unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags, void* user_data) {
  float* output = (float*)output_buffer;

  Buzzer* buzzer = &engine.buzzer;
  /* Get frequency and target amplitude from the atomics. */
  const int frequency_hz_fixed =
    SDL_AtomicGet(&buzzer->frequency_hz_atomic);
  const int target_amplitude_fixed =
    SDL_AtomicGet(&buzzer->target_amplitude_atomic);
  /* Convert to float. */
  const float frequency_hz = frequency_hz_fixed / 16.0f;
  const float target_amplitude = target_amplitude_fixed / 16384.0f;

  /* Determine the rotator for the desired frequency. */
  const float radians_per_sample = (float)(
      2.0 * M_PI * frequency_hz / engine.sample_rate_hz);
  float rotator[2];
  rotator[0] = cos(radians_per_sample);
  rotator[1] = sin(radians_per_sample);

  float phasor[2];
  phasor[0] = buzzer->phasor[0];
  phasor[1] = buzzer->phasor[1];
  float smoothed_amplitude_stage1 = buzzer->smoothed_amplitude_stage1;
  float smoothed_amplitude = buzzer->smoothed_amplitude;

  int i;
  for (i = 0; i < frames_per_buffer; ++i) {
    /* Apply second-order Gamma filter to smooth the target amplitude. */
    smoothed_amplitude_stage1 += engine.amplitude_smoother * (
        target_amplitude - smoothed_amplitude_stage1);
    smoothed_amplitude += engine.amplitude_smoother * (
        smoothed_amplitude_stage1 - smoothed_amplitude);

    /* Use phasor-rotator to efficiently generate a sine wave. */
    output[i] = smoothed_amplitude * phasor[0];
    float tmp = rotator[0] * phasor[0] - rotator[1] * phasor[1];
    phasor[1] = rotator[1] * phasor[0] + rotator[0] * phasor[1];
    phasor[0] = tmp;
  }

  /* Occasionally correct for accumulating round-off error by normalizing phasor
   * back to unit magnitude.
   */
  const float mag = sqrt(phasor[0] * phasor[0] + phasor[1] * phasor[1]);
  buzzer->phasor[0] = phasor[0] / mag;
  buzzer->phasor[1] = phasor[1] / mag;
  buzzer->smoothed_amplitude_stage1 = smoothed_amplitude_stage1;
  buzzer->smoothed_amplitude = smoothed_amplitude;
  return paContinue;
}

/* Starts portaudio. Returns 1 on success, 0 on failure. */
int StartPortaudio() {
  /* Initialize portaudio. */
  engine.pa_error = Pa_Initialize();
  if (engine.pa_error != paNoError) { return 0; }
  engine.pa_initialized = 1;

  /* Find output device. */
  const int output_device_index =
      FindPortAudioDevice(engine.output_device, 0, 1);
  if (output_device_index < 0) {
    fprintf(stderr, "\nError: "
        "Use --output flag to set a valid device:\n");
    PrintPortAudioDevices();
    return 0;
  }

  fprintf(stderr, "Output device: %s\n",
          Pa_GetDeviceInfo(output_device_index)->name);

  /* Set up buzzer. */
  engine.amplitude_smoother = 1.0 - exp(
      -1.0 / (0.002 * engine.sample_rate_hz));
  BuzzerInit(&engine.buzzer);

  /* Start output stream. */
  PaStreamParameters output_parameters;
  output_parameters.device = output_device_index;
  output_parameters.channelCount = 1;
  output_parameters.sampleFormat = paFloat32;
  output_parameters.suggestedLatency =
      Pa_GetDeviceInfo(output_parameters.device)->defaultLowOutputLatency;
  output_parameters.hostApiSpecificStreamInfo = NULL;

  engine.pa_error = Pa_OpenStream(
      &engine.pa_stream, NULL, &output_parameters,
      engine.sample_rate_hz,
      engine.chunk_size, 0, PortaudioCallback, NULL);
  if (engine.pa_error != paNoError) { return 0; }

  engine.pa_error = Pa_StartStream(engine.pa_stream);
  if (engine.pa_error != paNoError) { return 0; }
  return 1;
}

void CleanupPortaudio() {
  if (engine.pa_error != paNoError) {  /* Print PA error if there is one. */
    fprintf(stderr, "Error: portaudio: %s\n", Pa_GetErrorText(engine.pa_error));
  }

  if (engine.pa_stream) { Pa_CloseStream(engine.pa_stream); }
  if (engine.pa_initialized) { Pa_Terminate(); }
}

/* Starts SDL. Returns 1 on success, 0 on failure. */
int StartSdl() {
  /* Initialize SDL. */
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Error: Failed to initialize SDL\n");
    return 0;
  }

  SDL_EventState(SDL_SYSWMEVENT, SDL_IGNORE);
  SDL_EventState(SDL_USEREVENT, SDL_IGNORE);

  if (!BasicSdlAppCreate(&engine.app, "Tactometer",
      kScreenWidth, kScreenHeight, SDL_WINDOW_SHOWN)) {
    return 0;
  }
  SetWindowIcon(engine.app.window);

  /* Initialize font for drawing text. */
  return DrawTextInitFontTexture(engine.app.renderer);
}

void CleanupSdl() {
  if (strlen(SDL_GetError())) {  /* Print SDL error if there is one. */
    fprintf(stderr, "Error: SDL: %s\n", SDL_GetError());
  }

  DrawTextFreeFontTexture();
  BasicSdlAppDestroy(&engine.app);
  SDL_Quit();
}

/* Draws a horizontal meter bar, where value`is in [0, 1]. */
void DrawHBar(int x, int y, float value) {
  const int kWidth = 200;
  const int kHeight = 20;
  const int value_pixels = (int)((kWidth - 2) * value);
  SDL_SetRenderDrawColor(engine.app.renderer, 0x50, 0x50, 0x50, 0xff);
  SDL_Rect rect;
  rect.x = x;
  rect.y = y;
  rect.w = kWidth;
  rect.h = kHeight;
  SDL_RenderFillRect(engine.app.renderer, &rect);  /* Draw border. */

  SDL_SetRenderDrawColor(engine.app.renderer, 0x7e, 0x8a, 0xa2, 0xff);
  rect.x = x + 1;
  rect.y = y + 1;
  rect.w = value_pixels;
  rect.h -= 2;
  SDL_RenderFillRect(engine.app.renderer, &rect);  /* Draw filled part. */

  SDL_SetRenderDrawColor(engine.app.renderer, 0x30, 0x30, 0x30, 0xff);
  rect.x = x + value_pixels + 1;
  rect.w = kWidth - value_pixels - 2;
  SDL_RenderFillRect(engine.app.renderer, &rect);  /* Draw empty part. */
}

/* Draws a keytar key. */
void DrawKey(int x, int y, const char* label, int pressed) {
  const int kWidth = 48;
  const int kHeight = 85;
  SDL_SetRenderDrawColor(engine.app.renderer, 0x50, 0x50, 0x50, 0xff);
  SDL_Rect rect;
  rect.x = x;
  rect.y = y;
  rect.w = kWidth;
  rect.h = kHeight;
  SDL_RenderFillRect(engine.app.renderer, &rect);  /* Draw border. */

  if (pressed) {
    SDL_SetRenderDrawColor(engine.app.renderer, 0x7e, 0x8a, 0xa2, 0xff);
  } else {
    SDL_SetRenderDrawColor(engine.app.renderer, 0x30, 0x30, 0x30, 0xff);
  }
  ++rect.x;
  ++rect.y;
  rect.w -= 2;
  rect.h -= 2;
  SDL_RenderFillRect(engine.app.renderer, &rect);  /* Draw interior. */

  DrawTextSetColor(0xe2, 0xe2, 0xe5);
  DrawText(engine.app.renderer, x + kWidth/2 - 6,  /* Draw key label. */
      y + kHeight/2 - 10, "%s", label);
}

/* Draws frequency and amplitude UI. */
void DrawFrequencyAndAmplitudeMeters() {
  DrawText(engine.app.renderer, 5, 30,
      "frequency: %5.0f Hz", engine.frequency_hz);
  DrawText(engine.app.renderer, 5, 55,
      "amplitude: %5.1f dBFS", engine.amplitude_db);
  DrawHBar(300, 30, engine.frequency_hz / kMaxFrequencyHz);
  DrawHBar(300, 55, (-kMinAmplitudeDb + engine.amplitude_db)
      / -kMinAmplitudeDb);
}

/* Computes decibels from linear amplitude. */
float DbFromAmplitude(float amplitude) {
  return 20 * log(amplitude) / M_LN10;
}

/* Changes the current amplitude by `factor`. */
void ChangeAmplitude(float factor) {
  engine.amplitude *= factor;
  if (engine.amplitude > 1) { engine.amplitude = 1.0f; }
  if (engine.amplitude < kMinAmplitude) { engine.amplitude = kMinAmplitude; }
  engine.amplitude_db = DbFromAmplitude(engine.amplitude);
}

void BekesyFun(int first_call);

/* Keytar mode ****************************************************************/

void KeytarHandleKeyboard() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT ||
        (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
      engine.keep_running = 0;  /* Quit program. */
      break;
    }

    if (event.type == SDL_KEYUP) {
      if (event.key.keysym.sym == SDLK_TAB) {
        engine.fun = BekesyFun;  /* Switch to Bekesy's tracking mode. */
      }
    }
  }

  const uint8_t* key_state = SDL_GetKeyboardState(NULL);
  if (key_state[SDL_SCANCODE_MINUS]) {
    ChangeAmplitude(1.0 / 1.05);  /* Decrease amplitude when holding '-'. */
  } else if (key_state[SDL_SCANCODE_EQUALS]) {
    ChangeAmplitude(1.05f);  /* Increase amplitude when holding '=' or '+'. */
  }
  /* Check keytar note keys to determine what frequency to play. */
  float frequency_hz = 0.0f;
  int i;
  for (i = 0; i < kNumFrequencies; ++i) {
    if (key_state[kKeytarNoteKeys[i]]) {
      frequency_hz = kFrequenciesHz[i];
    }
  }

  if (frequency_hz > 0) {
    /* Update the audio callback on what to play. */
    engine.frequency_hz = frequency_hz;
    BuzzerSetFrequency(&engine.buzzer, frequency_hz);
    BuzzerSetAmplitude(&engine.buzzer, engine.amplitude);
  } else {
    BuzzerSetAmplitude(&engine.buzzer, 0.0f);  /* Set to silence. */
  }
}

void KeytarFun(int unused) {
  KeytarHandleKeyboard();

  /* Draw keytar keys. */
  const uint8_t* key_state = SDL_GetKeyboardState(NULL);
  int i;
  for (i = 0; i < kNumFrequencies; ++i) {
    const int x = 5 + 25 * i;
    const int y = (i % 2) ? 440 : 340;
    const SDL_Scancode scancode = kKeytarNoteKeys[i];
    DrawKey(x, y, SDL_GetScancodeName(scancode), key_state[scancode]);
  }

  DrawTextSetColor(0xe2, 0xe2, 0xe5);
  SDL_Renderer* renderer = engine.app.renderer;
  DrawText(renderer, 5, 5, "Keytar mode");
  DrawFrequencyAndAmplitudeMeters();
  DrawText(renderer, 5, 180, "Controls:");
  DrawText(renderer, 5, 205, "ASD...M   Note keys");
  DrawText(renderer, 5, 230, "-=        Change volume");
  DrawText(renderer, 5, 255, "Tab       Switch to Bekesy's tracking");
  DrawText(renderer, 5, 280, "Esc       Quit program");
}

/* Bekesy threshold mode ******************************************************/

void BekesyReset() {
  engine.frequency_hz = kFrequenciesHz[engine.selected_frequency_index];
  engine.bekesy_prev_reversal_db = 0.0f;
  engine.bekesy_midpoint_db_sum = 0.0f;
  engine.bekesy_num_reversals = 0;
  engine.num_plot_points = 0;
}

/* Prints a table to stderr of frequencies and amplitude thresholds, in CSV
 * format for convenient copying.
 */
void BekesyPrintThresholds() {
  int first = 1;
  int i;
  for (i = 0; i < kNumFrequencies; ++i) {
    if (engine.bekesy_thresholds_db[i] < 0.0f) {
      if (first) {
        printf("\nBekesy sensory threshold amplitudes\n");
        printf("Frequency (Hz),Threshold (dBFS)\n");
        first = 0;
      }
      printf("%g,%g\n", kFrequenciesHz[i], engine.bekesy_thresholds_db[i]);
    }
  }
  if (!first) {
    printf("\n");
  }
}

int BekesyHandleKeyboard() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT ||
        (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
      engine.keep_running = 0;  /* Quit program. */
      break;
    }

    if (event.type == SDL_KEYUP) {
      if (event.key.keysym.sym == SDLK_TAB) {
        engine.fun = KeytarFun;  /* Switch to keytar mode. */
      } else if (event.key.keysym.sym == SDLK_p) {
        BekesyPrintThresholds();  /* Print thresholds. */
      }
    }

    if (event.type == SDL_KEYDOWN) {
      /* Left/right keys to change frequency. */
      if (event.key.keysym.sym == SDLK_LEFT) {
        --engine.selected_frequency_index;
        if (engine.selected_frequency_index < 0) {
          engine.selected_frequency_index = 0;
        }
        BekesyReset();
      } else if (event.key.keysym.sym == SDLK_RIGHT) {
        ++engine.selected_frequency_index;
        if (engine.selected_frequency_index >= kNumFrequencies) {
          engine.selected_frequency_index = kNumFrequencies - 1;
        }
        BekesyReset();
      }
    }
  }
  const uint8_t* key_state = SDL_GetKeyboardState(NULL);
  int space_pressed;

  if (key_state[SDL_SCANCODE_SPACE]) {
    ChangeAmplitude(1.0 / 1.05);  /* Decrease amplitude when space is held. */
    space_pressed = 1;
  } else {
    ChangeAmplitude(1.05f);  /* Increase amplitude when space is not held. */
    space_pressed = 0;
  }
  return space_pressed;
}

void BekesyFun(int first_call) {
  if (first_call) {
    BekesyReset();
  }

  const int space_pressed = BekesyHandleKeyboard();
  /* Update the audio callback on what to play. */
  BuzzerSetFrequency(&engine.buzzer, engine.frequency_hz);
  BuzzerSetAmplitude(&engine.buzzer, engine.amplitude);

  if (space_pressed != engine.prev_space_pressed) {
    /* A "reversal" happened, space was just pressed or released. */
    const float reversal_db = engine.amplitude_db;
    ++engine.bekesy_num_reversals;

    /* Discard first reversal. Get first midpoint on the 3rd reversal. */
    if (engine.bekesy_num_reversals >= 3) {
      const float midpoint_db =
        0.5f * (reversal_db + engine.bekesy_prev_reversal_db);
      engine.bekesy_midpoint_db_sum += midpoint_db;
    }

    engine.bekesy_prev_reversal_db = reversal_db;
    engine.prev_space_pressed = space_pressed;
  }

  /* Update plot of recent amplitude. */
  if (engine.num_plot_points == kMaxPlotPoints) {
    /* Scroll the plot left one point. */
    memmove(engine.plot_points, engine.plot_points + 1,
        sizeof(int) * (kMaxPlotPoints - 1));
    --engine.num_plot_points;
  }
  int amplitude_db_y = (int)(
      engine.amplitude_db * kScreenHeight / kMinAmplitudeDb);
  engine.plot_points[engine.num_plot_points] = amplitude_db_y;
  ++engine.num_plot_points;

  /* Draw the plot. */
  SDL_Point points[kMaxPlotPoints];
  const int x_offset = kMaxPlotPoints - engine.num_plot_points;
  int i;
  for (i = 0; i < engine.num_plot_points; ++i) {
    points[i].x = (i + x_offset) * kScreenWidth / kMaxPlotPoints;
    points[i].y = engine.plot_points[i];
  }
  SDL_SetRenderDrawColor(engine.app.renderer, 0x7e, 0x8a, 0xa2, 0xff);
  SDL_RenderDrawLines(engine.app.renderer, points, engine.num_plot_points);

  DrawTextSetColor(0xe2, 0xe2, 0xe5);
  DrawText(engine.app.renderer, 5, 5, "Bekesy's tracking method");
  DrawFrequencyAndAmplitudeMeters();

  DrawText(engine.app.renderer, 5, 180, "Controls:");
  DrawText(engine.app.renderer, 5, 205, "Space     Decrease amplitude");
  DrawText(engine.app.renderer, 5, 230, "\x7f\x81        Select frequency");
  DrawText(engine.app.renderer, 5, 255, "Tab       Switch to keytar mode");
  DrawText(engine.app.renderer, 5, 280, "Esc       Quit program");

  DrawTextSetColor(0xb1, 0xd6, 0x31);
  DrawText(engine.app.renderer, 5, 130, "Hold SPACE when you perceive the signal.");

  if (engine.bekesy_num_reversals >= 3) {
    /* Estimate the threshold from the average of the reversals. */
    float threshold_db = (engine.bekesy_midpoint_db_sum + engine.amplitude_db/2) /
      (engine.bekesy_num_reversals - 1.5f);

    engine.bekesy_thresholds_db[engine.selected_frequency_index] = threshold_db;

    DrawTextSetColor(0xff, 0x00, 0x66);
    DrawText(engine.app.renderer, 5, 80, "threshold: %5.1f dBFS", threshold_db);

    SDL_SetRenderDrawColor(engine.app.renderer, 0xff, 0x00, 0x66, 0xff);

    /* Draw horizontal threshold line on the plot. */
    const int threshold_db_y = (int)(
        threshold_db * kScreenHeight / kMinAmplitudeDb);
    SDL_RenderDrawLine(engine.app.renderer,
        0, threshold_db_y, kScreenWidth, threshold_db_y);

    /* Draw vertical threshold line on the amplitude meter. */
    const int threshold_db_x = 300 + (int)(
        (-kMinAmplitudeDb + threshold_db) * 200 / -kMinAmplitudeDb);
    SDL_RenderDrawLine(engine.app.renderer,
        threshold_db_x, 56, threshold_db_x, 55 + 18);
  } else {
    DrawTextSetColor(0xff, 0x00, 0x66);
    DrawText(engine.app.renderer, 5, 80, "threshold:  ---- dBFS");
  }
}

int main(int argc, char** argv) {
  engine.pa_initialized = 0;
  engine.pa_error = paNoError;
  engine.pa_stream = NULL;
  engine.output_device = "default";
  /* We test up to only 600Hz, so we don't need as high as 16kHz sample rate for
   * our signal generation. However, many portaudio devices support just a few
   * standard rates, and 16kHz seems more likely to be supported than 8kHz. In
   * any case, the sample rate is configurable with the --sample_rate_hz flag.
   */
  engine.sample_rate_hz = 16000;
  engine.chunk_size = 1024;

  BasicSdlAppInit(&engine.app);

  engine.selected_frequency_index = 0;
  engine.frequency_hz = kFrequenciesHz[engine.selected_frequency_index];
  engine.amplitude = 0.2;
  engine.amplitude_db = DbFromAmplitude(engine.amplitude);

  engine.keep_running = 1;
  engine.fun = KeytarFun;

  engine.num_plot_points = 0;

  int i;
  for (i = 0; i < kNumFrequencies; ++i) {
    engine.bekesy_thresholds_db[i] = 1.0f;
  }

  for (i = 1; i < argc; ++i) { /* Parse flags. */
    if (StartsWith(argv[i], "--output=")) {
      engine.output_device = strchr(argv[i], '=') + 1;
    } else if (StartsWith(argv[i], "--sample_rate_hz=")) {
      engine.sample_rate_hz = atoi(strchr(argv[i], '=') + 1);
    } else if (StartsWith(argv[i], "--chunk_size=")) {
      engine.chunk_size = atoi(strchr(argv[i], '=') + 1);
    } else {
      fprintf(stderr, "Error: Invalid flag \"%s\"\n", argv[i]);
      goto quit;
    }
  }

  if (!StartPortaudio() || !StartSdl()) {
    fprintf(stderr, "Unable to start tactometer\n");
    goto quit;
  }

  int first_call = 1;
  while (engine.keep_running) {  /* Engine main loop. */
    SDL_SetRenderDrawColor(engine.app.renderer, 0x20, 0x20, 0x20, 0xff);
    SDL_RenderClear(engine.app.renderer);

    void (*prev_fun)(int) = engine.fun;
    engine.fun(first_call);
    first_call = (prev_fun != engine.fun);

    SDL_RenderPresent(engine.app.renderer);
    SDL_Delay(25);
  }

  BekesyPrintThresholds();
  /* Set buzzer to silence and wait a moment to avoid click when exiting. */
  BuzzerSetAmplitude(&engine.buzzer, 0.0f);
  SDL_Delay(100);

quit:
  CleanupSdl();
  CleanupPortaudio();
  return 0;
}

