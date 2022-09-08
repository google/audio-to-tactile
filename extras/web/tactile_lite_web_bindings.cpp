// Copyright 2019, 2021-2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if !defined(__EMSCRIPTEN__)
#error This file must be built with emscripten
#endif

#include <emscripten.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/tactile/tactile_lite.h"
#include "extras/tools/sdl/basic_sdl_app.h"

constexpr int kDecimationFactor = 32;
constexpr int kNumPoints = 2048;
constexpr int kMaxPeaks = kNumPoints / 32;
constexpr int kCanvasWidth = 800;
constexpr int kCanvasHeight = 600;

struct {
  BasicSdlApp app;
  SingleBandEnvelope envelope;
  SparsePeakPicker peak_picker;
  float* tactile_output;
  SDL_Point envelope_plot[kNumPoints];
  SDL_Point smoothed_plot[kNumPoints];
  SDL_Point thresh_plot[kNumPoints];
  int peak_i[kMaxPeaks];
  int peak_y[kMaxPeaks];
  int num_peaks;
} engine;

static void MainTick();

// Initializes SDL. This gets called immediately after the emscripten runtime
// has initialized.
extern "C" void EMSCRIPTEN_KEEPALIVE OnLoad() {
  for (int i = 0; i < kNumPoints; ++i) {
    const int x = (kCanvasWidth * i + (kNumPoints - 1) / 2) / (kNumPoints - 1);
    engine.envelope_plot[i].x = x;
    engine.envelope_plot[i].y = kCanvasHeight - 1;
    engine.smoothed_plot[i].x = x;
    engine.smoothed_plot[i].y = kCanvasHeight - 1;
    engine.thresh_plot[i].x = x;
    engine.thresh_plot[i].y = kCanvasHeight - 1;
  }
  engine.num_peaks = 0;

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Error: %s\n", SDL_GetError());
    exit(1);
  }

  SDL_EventState(SDL_MOUSEMOTION, SDL_IGNORE);
  // Disable SDL keyboard events. Otherwise, the tab key (to navigate
  // interactive elements) does not work on the web page since SDL captures it.
  SDL_EventState(SDL_TEXTINPUT, SDL_DISABLE);
  SDL_EventState(SDL_KEYDOWN, SDL_DISABLE);
  SDL_EventState(SDL_KEYUP, SDL_DISABLE);

  // Set the event handling loop. This must be set *before* creating the window,
  // otherwise there is an error "Cannot set timing mode for main loop".
  emscripten_set_main_loop(MainTick, 0, 0);

  if (!BasicSdlAppCreate(&engine.app, "", kCanvasWidth, kCanvasHeight,
                         SDL_WINDOW_SHOWN)) {
    exit(1);
  }
}

static void SetColor(uint32_t color) {
  SDL_SetRenderDrawColor(engine.app.renderer, color >> 16, (color >> 8) & 0xff,
                         color & 0xff, SDL_ALPHA_OPAQUE);
}

// Emscripten will call this function once per frame to do event processing
// (though we ignore all events) and to render the next frame.
static void MainTick() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {}  // Ignore events.

  SetColor(0x000000);
  SDL_RenderClear(engine.app.renderer);

  SetColor(0x67493e);
  SDL_RenderDrawLines(engine.app.renderer, engine.envelope_plot, kNumPoints);

  SetColor(0x5e94b8);
  SDL_RenderDrawLines(engine.app.renderer, engine.thresh_plot, kNumPoints);

  SetColor(0xf1bc1c);
  SDL_RenderDrawLines(engine.app.renderer, engine.smoothed_plot, kNumPoints);

  SetColor(0xffffcc);
  for (int i = 0; i < engine.num_peaks; ++i) {
    SDL_Rect rect;
    rect.x = engine.envelope_plot[engine.peak_i[i]].x;
    rect.y = engine.peak_y[i];
    rect.w = 3;
    rect.h = kCanvasHeight - rect.y;
    SDL_RenderFillRect(engine.app.renderer, &rect);
  }

  SDL_RenderPresent(engine.app.renderer);
}

// Initializes TactileProcessor. This gets called after WebAudio has started.
extern "C" void EMSCRIPTEN_KEEPALIVE TactileInitAudio(
    int sample_rate_hz, int chunk_size) {
  SingleBandEnvelopeInit(&engine.envelope, &kDefaultSingleBandEnvelopeParams,
                         sample_rate_hz, kDecimationFactor);
  SparsePeakPickerInit(&engine.peak_picker, &kDefaultSparsePeakPickerParams,
                       sample_rate_hz / kDecimationFactor);
  engine.tactile_output =
      (float*)malloc((chunk_size / kDecimationFactor) * sizeof(float));
}

// Processes one chunk of audio data. Called from onaudioprocess.
extern "C" void EMSCRIPTEN_KEEPALIVE TactileProcessAudio(
    intptr_t input_ptr, int chunk_size) {
  float* input = reinterpret_cast<float*>(input_ptr);

  SingleBandEnvelopeProcessSamples(&engine.envelope, input, chunk_size,
                                   engine.tactile_output);

  const int points_discarded = chunk_size / kDecimationFactor;

  int peaks_discarded = 0;
  for (; peaks_discarded < engine.num_peaks; ++peaks_discarded) {
    if (engine.peak_i[peaks_discarded] >= points_discarded) { break; }
  }
  for (int i = 0, j = peaks_discarded; j < engine.num_peaks; ++i, ++j) {
    engine.peak_i[i] = engine.peak_i[j] - points_discarded;
    engine.peak_y[i] = engine.peak_y[j];
  }
  engine.num_peaks -= peaks_discarded;

  int i = 0;
  for (int j = points_discarded; j < kNumPoints; ++i, ++j) {
    engine.envelope_plot[i].y = engine.envelope_plot[j].y;
    engine.smoothed_plot[i].y = engine.smoothed_plot[j].y;
    engine.thresh_plot[i].y = engine.thresh_plot[j].y;
  }

  const float scale = kCanvasHeight / 2.5f;
  for (int j = 0; i < kNumPoints; ++i, ++j) {
    const float sample = engine.tactile_output[j];
    const float peak =
        SparsePeakPickerProcessSamples(&engine.peak_picker, &sample, 1);

    engine.envelope_plot[i].y = kCanvasHeight - 1 - scale * sample;
    engine.smoothed_plot[i].y =
        kCanvasHeight - 1 - scale * engine.peak_picker.smoothed[1];
    engine.thresh_plot[i].y =
        kCanvasHeight - 1 - scale * engine.peak_picker.thresh;

    if (peak > 0.0f && engine.num_peaks < kMaxPeaks) {
      engine.peak_i[engine.num_peaks] = i;
      engine.peak_y[engine.num_peaks] = kCanvasHeight - 1 - scale * peak;
      ++engine.num_peaks;
    }
  }
}
