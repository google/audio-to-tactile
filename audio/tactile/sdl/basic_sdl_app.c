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

#include "audio/tactile/sdl/basic_sdl_app.h"

#include <stdio.h>

void BasicSdlAppInit(BasicSdlApp* app) {
  if (app) {
    app->window = NULL;
    app->renderer = NULL;
  }
}

int BasicSdlAppCreate(BasicSdlApp* app,
                      const char* title,
                      int width,
                      int height,
                      Uint32 flags) {
  if (app == NULL) { return 0; }

  /* Create application window. */
  app->window = SDL_CreateWindow(
      title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      width, height, flags);
  if (app->window == NULL) {
    fprintf(stderr, "Error: Failed to create window: %s\n", SDL_GetError());
    goto fail;
  }

  /* Create a hardware-accelerated renderer. */
  app->renderer = SDL_CreateRenderer(
      app->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (app->renderer == NULL) {
    /* Failed to initialize hardware-accelerated renderer.
     * Fall back to software renderer.
     */
    app->renderer = SDL_CreateRenderer(app->window, -1, 0);
  }

  static SDL_RendererInfo renderer_info = {0};
  if (app->renderer) {
    SDL_GetRendererInfo(app->renderer, &renderer_info);
  }

  if (app->renderer == NULL || renderer_info.num_texture_formats == 0) {
    fprintf(stderr, "Error: Failed to create renderer: %s\n", SDL_GetError());
    goto fail;
  }
  return 1;

fail:
  BasicSdlAppDestroy(app);
  return 0;
}

void BasicSdlAppDestroy(BasicSdlApp* app) {
  if (app) {
    if (app->renderer) { SDL_DestroyRenderer(app->renderer); }
    if (app->window) { SDL_DestroyWindow(app->window); }
    BasicSdlAppInit(app);  /* Reset pointers to null. */
  }
}
