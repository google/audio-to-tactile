// Copyright 2020 Google LLC
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

// TODO: Visualize label output in addition to scores.

#if !defined(__EMSCRIPTEN__)
#error This file must be built with emscripten
#endif

#include <emscripten.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "extras/tools/sdl/basic_sdl_app.h"
#include "src/dsp/fast_fun.h"
#include "src/frontend/carl_frontend.h"
#include "src/phonetics/classify_phoneme.h"

const int kNumFrames = kClassifyPhonemeNumFrames;
const int kNumChannels = kClassifyPhonemeNumChannels;
constexpr int kNumPhonemes = kClassifyPhonemeNumPhonemes;
constexpr int kBlockSize = 128;
constexpr float kPcenDiffusivity = 60.0f;

constexpr int kWidth = 200;
constexpr int kHeight = kNumPhonemes;
constexpr int kScaleX = 4;
constexpr int kScaleY = 16;

struct {
  BasicSdlApp app;
  SDL_Texture* texture;

  CarlFrontend* frontend;
  float* frames;
  ClassifyPhonemeScores scores;
} engine;

static void MainTick();

// RGBA colormap for displaying the scores.
static constexpr uint32_t kColormap[256] = {
    0x372788ff, 0x37298aff, 0x372b8cff, 0x372d8eff, 0x362f91ff, 0x363193ff,
    0x363396ff, 0x363598ff, 0x36369bff, 0x36389eff, 0x363aa2ff, 0x353ca5ff,
    0x353ea8ff, 0x353facff, 0x3441afff, 0x3343b3ff, 0x3345b6ff, 0x3246baff,
    0x3148beff, 0x2f4ac1ff, 0x2e4bc4ff, 0x2c4dc8ff, 0x294fcbff, 0x2750cdff,
    0x2452d0ff, 0x2053d3ff, 0x1d55d5ff, 0x1956d7ff, 0x1558d9ff, 0x1159daff,
    0x0d5bdcff, 0x095cddff, 0x075edeff, 0x045fdeff, 0x0361dfff, 0x0262dfff,
    0x0264e0ff, 0x0265e0ff, 0x0366e0ff, 0x0468e0ff, 0x0569dfff, 0x066adfff,
    0x076cdfff, 0x086ddeff, 0x0a6edeff, 0x0b70ddff, 0x0c71dcff, 0x0d72dcff,
    0x0e74dbff, 0x0f75dbff, 0x0f76daff, 0x1077d9ff, 0x1178d9ff, 0x117ad8ff,
    0x127bd8ff, 0x127cd7ff, 0x127dd7ff, 0x127ed6ff, 0x1380d6ff, 0x1381d5ff,
    0x1382d5ff, 0x1383d4ff, 0x1384d4ff, 0x1385d3ff, 0x1386d3ff, 0x1287d3ff,
    0x1288d2ff, 0x1289d2ff, 0x128bd2ff, 0x118cd2ff, 0x118dd1ff, 0x118ed1ff,
    0x108fd1ff, 0x1090d1ff, 0x0f91d0ff, 0x0f92d0ff, 0x0e93d0ff, 0x0e94d0ff,
    0x0d95cfff, 0x0c95cfff, 0x0c96cfff, 0x0b97cfff, 0x0b98ceff, 0x0a99ceff,
    0x099aceff, 0x099bcdff, 0x089ccdff, 0x089dccff, 0x079eccff, 0x079ecbff,
    0x069fcbff, 0x06a0caff, 0x05a1caff, 0x05a2c9ff, 0x05a3c8ff, 0x05a3c8ff,
    0x05a4c7ff, 0x05a5c6ff, 0x05a6c5ff, 0x05a7c4ff, 0x05a7c3ff, 0x06a8c2ff,
    0x06a9c1ff, 0x07aac0ff, 0x08aabfff, 0x09abbeff, 0x0aacbdff, 0x0bacbbff,
    0x0cadbaff, 0x0daeb9ff, 0x0eaeb7ff, 0x10afb6ff, 0x12b0b5ff, 0x13b0b3ff,
    0x15b1b2ff, 0x17b1b1ff, 0x19b2afff, 0x1bb3aeff, 0x1db3acff, 0x1fb4abff,
    0x21b4a9ff, 0x24b5a8ff, 0x26b5a6ff, 0x28b6a5ff, 0x2bb6a4ff, 0x2db7a2ff,
    0x30b7a1ff, 0x32b89fff, 0x35b89eff, 0x38b99cff, 0x3ab99bff, 0x3dba9aff,
    0x40ba98ff, 0x42ba97ff, 0x45bb95ff, 0x48bb94ff, 0x4abc93ff, 0x4dbc91ff,
    0x50bc90ff, 0x53bd8fff, 0x55bd8dff, 0x58bd8cff, 0x5bbd8bff, 0x5dbe8aff,
    0x60be88ff, 0x63be87ff, 0x66be86ff, 0x68bf85ff, 0x6bbf83ff, 0x6ebf82ff,
    0x70bf81ff, 0x73bf80ff, 0x76bf7fff, 0x78bf7dff, 0x7bc07cff, 0x7ec07bff,
    0x80c07aff, 0x83c079ff, 0x85c078ff, 0x88c077ff, 0x8bc076ff, 0x8dc074ff,
    0x90c073ff, 0x92c072ff, 0x95c071ff, 0x97bf70ff, 0x9abf6fff, 0x9dbf6eff,
    0x9fbf6dff, 0xa2bf6cff, 0xa4bf6bff, 0xa7bf6aff, 0xa9be69ff, 0xacbe68ff,
    0xaebe67ff, 0xb0be66ff, 0xb3be65ff, 0xb5bd64ff, 0xb8bd63ff, 0xbabd62ff,
    0xbcbd61ff, 0xbfbc60ff, 0xc1bc5fff, 0xc3bc5eff, 0xc6bc5dff, 0xc8bb5cff,
    0xcabb5bff, 0xcdbb5aff, 0xcfbb59ff, 0xd1bb58ff, 0xd3ba57ff, 0xd5ba56ff,
    0xd7ba55ff, 0xd9ba54ff, 0xdbba53ff, 0xddba52ff, 0xdfba51ff, 0xe1ba50ff,
    0xe3ba4fff, 0xe5ba4eff, 0xe7ba4dff, 0xe8ba4bff, 0xeaba4aff, 0xecba49ff,
    0xedba48ff, 0xefbb47ff, 0xf0bb46ff, 0xf2bb45ff, 0xf3bc44ff, 0xf4bc43ff,
    0xf5bd42ff, 0xf6bd41ff, 0xf7be40ff, 0xf8bf3fff, 0xf9c03eff, 0xfac03dff,
    0xfbc13cff, 0xfbc23bff, 0xfcc33aff, 0xfcc438ff, 0xfdc537ff, 0xfdc636ff,
    0xfdc735ff, 0xfdc934ff, 0xfdca33ff, 0xfdcb32ff, 0xfdcc31ff, 0xfdce2fff,
    0xfdcf2eff, 0xfcd02dff, 0xfcd22cff, 0xfbd32bff, 0xfbd529ff, 0xfad628ff,
    0xfad827ff, 0xf9da26ff, 0xf8db25ff, 0xf7dd23ff, 0xf7de22ff, 0xf6e021ff,
    0xf5e11fff, 0xf5e31eff, 0xf4e51dff, 0xf4e61cff, 0xf3e81aff, 0xf3ea19ff,
    0xf3eb18ff, 0xf3ed16ff, 0xf3ee15ff, 0xf4f013ff, 0xf5f212ff, 0xf5f311ff,
    0xf7f50fff, 0xf8f60eff, 0xfaf80cff, 0xfcf90bff};

// Initializes SDL. This gets called immediately after the emscripten runtime
// has initialized.
extern "C" void EMSCRIPTEN_KEEPALIVE OnLoad() {
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

  if (!BasicSdlAppCreate(&engine.app, "", kScaleX * kWidth, kScaleY * kHeight,
                         SDL_WINDOW_SHOWN)) {
    exit(1);
  }

  engine.texture =
      SDL_CreateTexture(engine.app.renderer, SDL_PIXELFORMAT_RGBA8888,
                        SDL_TEXTUREACCESS_STREAMING, kWidth, kHeight);
  if (engine.texture == NULL) {
    fprintf(stderr, "Error: %s", SDL_GetError());
    exit(1);
  }

  int pitch;
  uint8_t* pixels;
  SDL_LockTexture(engine.texture, NULL, (void**)&pixels, &pitch);
  for (int y = 0; y < kHeight; y++) {
    uint32_t* row = (uint32_t*)(pixels + pitch * y);
    for (int x = 0; x < kWidth; x++) {
      row[x] = kColormap[0];
    }
  }
  SDL_UnlockTexture(engine.texture);

  SDL_SetRenderDrawColor(engine.app.renderer, 0x37, 0x27, 0x88, 255);
}

// Emscripten will call this function once per frame to do event processing
// (though we ignore all events) and to render the next frame.
static void MainTick() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
  }  // Ignore events.

  Uint32 format;
  int w;
  int h;
  SDL_QueryTexture(engine.texture, &format, NULL, &w, &h);

  int pitch;
  uint8_t* pixels;
  SDL_LockTexture(engine.texture, NULL, (void**)&pixels, &pitch);

  memmove(pixels, pixels + 4, pitch * h - 4);
  pixels += 4 * (w - 1);
  for (int y = 0; y < h; y++, pixels += pitch) {
    const uint8_t i = 255 * engine.scores.phoneme[y];
    *((uint32_t*)pixels) = kColormap[i];
  }

  SDL_UnlockTexture(engine.texture);

  SDL_RenderClear(engine.app.renderer);
  SDL_Rect dest{0, 0, kScaleX * kWidth, kScaleY * kHeight};
  SDL_RenderCopy(engine.app.renderer, engine.texture, NULL, &dest);
  SDL_RenderPresent(engine.app.renderer);
}

// Initializes CarlFrontend. This gets called after WebAudio has started.
extern "C" void EMSCRIPTEN_KEEPALIVE ClassifierInitAudio(int sample_rate_hz,
                                                         int chunk_size) {
  CarlFrontendParams frontend_params = kCarlFrontendDefaultParams;
  frontend_params.input_sample_rate_hz = sample_rate_hz;
  frontend_params.block_size = kBlockSize;
  frontend_params.pcen_cross_channel_diffusivity = kPcenDiffusivity;
  engine.frontend = CarlFrontendMake(&frontend_params);
  if (engine.frontend == NULL) {
    fprintf(stderr, "Error: Failed to create CarlFrontend.\n");
    exit(1);
  }

  engine.frames = (float*)malloc(sizeof(float) * kNumFrames * kNumChannels);
}

// Processes one chunk of audio data. Called from onaudioprocess.
extern "C" void EMSCRIPTEN_KEEPALIVE ClassifierProcessAudio(intptr_t input_ptr,
                                                            int chunk_size) {
  float* input = reinterpret_cast<float*>(input_ptr);
  const int num_blocks = chunk_size / kBlockSize;

  for (int b = 0; b < num_blocks; ++b) {
    memmove(engine.frames, engine.frames + kNumChannels,
            sizeof(float) * kNumChannels * (kNumFrames - 1));
    float* output = engine.frames + kNumChannels * (kNumFrames - 1);
    CarlFrontendProcessSamples(engine.frontend, input, output);
    input += kBlockSize;
  }

  ClassifyPhoneme(engine.frames, nullptr, &engine.scores);
}
