// Copyright 2019 Google LLC
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

#include "dsp/fast_fun.h"
#include "tactile/tactile_processor.h"
#include "extras/tools/sdl/basic_sdl_app.h"
#include "extras/tools/sdl/texture_from_rle_data.h"

constexpr int kNumTactors = 10;
constexpr int kDecimationFactor = 8;
constexpr int kBlockSize = 64;
constexpr int kOutputBlockSize = kBlockSize / kDecimationFactor;

// Defined in run_tactile_processor_assets.c.
constexpr int kNumImageAssets = (kNumTactors + 1);
extern const uint8_t* kImageAssetsRle[kNumImageAssets];

struct {
  BasicSdlApp app;
  SDL_Texture* image_assets[kNumImageAssets];
  SDL_Rect image_asset_rects[kNumImageAssets];
  uint8_t colormap[256 * 3];

  int chunk_size;
  TactileProcessor* tactile_processor;
  float tactile_output[kOutputBlockSize * kNumTactors];
  float volume_decay_coeff;
  float volume[kNumTactors];
} engine;

static void MainTick();
static void GenerateColormap(uint8_t* colormap);

// Initializes SDL. This gets called immediately after the emscripten runtime
// has initialized.
extern "C" void EMSCRIPTEN_KEEPALIVE OnLoad() {
  for (int c = 0; c < kNumTactors; ++c) {
    engine.volume[c] = 0.0f;
  }

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

  if (!BasicSdlAppCreate(&engine.app, "", 326, 512, SDL_WINDOW_SHOWN)) {
    exit(1);
  }

  /* Create SDL_Textures from embedded image assets. */
  for (int i = 0; i < kNumImageAssets; ++i) {
    engine.image_assets[i] = CreateTextureFromRleData(
        kImageAssetsRle[i], engine.app.renderer,
        &engine.image_asset_rects[i]);
    if (!engine.image_assets[i]) { exit(1); }
  }

  GenerateColormap(engine.colormap);
  SDL_SetTextureColorMod(engine.image_assets[kNumTactors], 0x37, 0x71, 0x8d);
}

// Emscripten will call this function once per frame to do event processing
// (though we ignore all events) and to render the next frame.
static void MainTick() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {}  // Ignore events.

  SDL_RenderClear(engine.app.renderer);
  // Render background texture.
  SDL_Texture* background_image = engine.image_assets[kNumTactors];
  SDL_Rect background_rect = engine.image_asset_rects[kNumTactors];
  SDL_RenderCopy(engine.app.renderer, background_image,
                 nullptr, &background_rect);

  for (int c = 0; c < kNumTactors; ++c) {
    // Get the RMS value of the cth tactor.
    const float rms = engine.volume[c];
    // Map the RMS in range [rms_min, rms_max] logarithmically to [0, 1].
    constexpr float kRmsMin = 0.003f;
    constexpr float kRmsMax = 0.05f;
    float activation =
        FastLog2(1e-12f + rms / kRmsMin) / FastLog2(kRmsMax / kRmsMin);
    activation = std::min<float>(std::max<float>(activation, 0.0f), 1.0f);

    // Render the cth texture with color according to `activation`.
    const int index = static_cast<int>(std::round(255 * activation));
    const uint8_t* rgb = &engine.colormap[3 * index];
    SDL_SetTextureColorMod(engine.image_assets[c], rgb[0], rgb[1], rgb[2]);
    SDL_RenderCopy(engine.app.renderer, engine.image_assets[c],
                   nullptr, &engine.image_asset_rects[c]);
  }

  SDL_RenderPresent(engine.app.renderer);
}

// Initializes TactileProcessor. This gets called after WebAudio has started.
extern "C" void EMSCRIPTEN_KEEPALIVE TactileInitAudio(
    int sample_rate_hz, int chunk_size) {
  engine.chunk_size = chunk_size;
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  params.decimation_factor = kDecimationFactor;
  params.frontend_params.block_size = kBlockSize;
  params.frontend_params.input_sample_rate_hz = sample_rate_hz;

  const float master_gain = 0.5f;
  params.baseband_channel_params.output_gain *= master_gain;
  params.vowel_channel_params.output_gain *= master_gain;
  params.fricative_channel_params.output_gain *= master_gain;

  engine.tactile_processor = TactileProcessorMake(&params);
  if (!engine.tactile_processor) {
    fprintf(stderr, "Error: Failed to create TactileProcessor.\n");
    exit(1);
  }

  constexpr float kVolumeMeterTimeConstantSeconds = 0.05f;
  engine.volume_decay_coeff = std::exp(
      -chunk_size / (kVolumeMeterTimeConstantSeconds * sample_rate_hz));
}

// Processes one chunk of audio data. Called from onaudioprocess.
extern "C" void EMSCRIPTEN_KEEPALIVE TactileProcessAudio(
    intptr_t input_ptr, int chunk_size) {
  float* input = reinterpret_cast<float*>(input_ptr);
  const int num_blocks = chunk_size / kBlockSize;
  float volume_accum[kNumTactors] = {0.0f};

  for (int b = 0; b < num_blocks; ++b) {
    float* tactile = engine.tactile_output;
    TactileProcessorProcessSamples(engine.tactile_processor, input, tactile);

    for (int i = 0; i < kOutputBlockSize; ++i) {
      // For visualization, accumulate energy for each tactile signal.
      for (int c = 0; c < kNumTactors; ++c) {
        volume_accum[c] += tactile[c] * tactile[c];
      }
      tactile += kNumTactors;
    }

    input += kBlockSize;
  }

  for (int c = 0; c < kNumTactors; ++c) {
    // Compute RMS value and update volume meters.
    const float rms = std::sqrt(volume_accum[c]
        / (num_blocks * (kOutputBlockSize * kNumTactors)));
    float updated_volume = engine.volume[c] * engine.volume_decay_coeff;
    if (rms > updated_volume) { updated_volume = rms; }
    engine.volume[c] = updated_volume;
  }
}

// Generates a colormap that fades from a dark blue color to white.
static void GenerateColormap(uint8_t* colormap) {
  const uint8_t kStartR = 0x14;
  const uint8_t kStartG = 0x2a;
  const uint8_t kStartB = 0x38;
  for (int i = 0; i < 256; ++i, colormap += 3) {
    const float x = i / 255.0f;
    colormap[0] = static_cast<int>(std::round(kStartR + (255 - kStartR) * x));
    colormap[1] = static_cast<int>(std::round(kStartG + (255 - kStartG) * x));
    colormap[2] = static_cast<int>(std::round(kStartB + (255 - kStartB) * x));
  }
}
