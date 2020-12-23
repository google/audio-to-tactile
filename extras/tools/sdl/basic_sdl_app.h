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
 * Make a basic single-window SDL application.
 *
 * This library creates an SDL window and renderer. This is enough to make a
 * basic single-window SDL application.
 *
 * Use:
 *   BasicSdlApp app;
 *   BasicSdlAppInit(&app);  // Initialize pointers to null.
 *
 *   SDL_Init(SDL_INIT_VIDEO);
 *
 *   if (!BasicSdlAppCreate(&app, "Title", 640, 480, SDL_WINDOW_SHOWN)) {
 *     exit(1);
 *   }
 *
 *   int keep_running = 1;
 *   while (keep_running) {
 *     SDL_Event event;
 *     while (SDL_PollEvent(&event)) {
 *       if (event.type == SDL_QUIT) {
 *         keep_running = 0;
 *       }
 *     }
 *
 *     SDL_RenderClear(engine.app.renderer);
 *     // Drawing code here...
 *     SDL_RenderPresent(engine.app.renderer);
 *     SDL_Delay(50);
 *   }
 *
 *   BasicSdlAppDestroy(&app);
 *   SDL_Quit();
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_BASIC_SDL_APP_H_
#define AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_BASIC_SDL_APP_H_

#include "SDL2/SDL.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  SDL_Window* window;              /* Application window.                */
  SDL_Renderer* renderer;          /* Object for SDL drawing operations. */
} BasicSdlApp;

/* Initializes window and renderer to null. */
void BasicSdlAppInit(BasicSdlApp* app);

/* Creates window and renderer. The window has `title` title and `width` by
 * `height` dimensions in pixels. `flags` is a bitfield of SDL_WindowFlags or'd
 * together, e.g.
 *   SDL_WINDOW_SHOWN           Window is visible.
 *   SDL_WINDOW_RESIZABLE       Window can be resized.
 *   SDL_WINDOW_FULLSCREEN      Fullscreen window.
 * Full list of flags at https://wiki.libsdl.org/SDL_WindowFlags
 *
 * Returns 1 on success, 0 on failure.
 */
int BasicSdlAppCreate(BasicSdlApp* app,
                      const char* title,
                      int width,
                      int height,
                      Uint32 flags);

/* Destroys the window and renderer. */
void BasicSdlAppDestroy(BasicSdlApp* app);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_BASIC_SDL_APP_H_ */
