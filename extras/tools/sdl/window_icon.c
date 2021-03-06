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

#include "extras/tools/sdl/window_icon.h"

#define kIconSize 16

void SetWindowIcon(SDL_Window* window) {
  /* 16x16 icon, stored as 16-bit ARGB data. */
  static const uint16_t kIconData[kIconSize * kIconSize] = {
      0x18bd, 0xb8bd, 0xf7bd, 0x08bb, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x28bd, 0xe7bd,
      0xf7bd, 0x68bd, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x28bd, 0xf8bd, 0xf8bd, 0x87bd,
      0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0x0000, 0x0000, 0x28bd, 0xf8bd, 0xf7bd, 0xb8bd, 0x0000, 0x0000,
      0x88bd, 0xb7bd, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0x28bd, 0xf8bd, 0xf8bd, 0xf8bd, 0x0000, 0x28bd, 0xd8bd, 0xe7bd,
      0x58bd, 0x0000, 0x0000, 0x27bd, 0x18bd, 0x0000, 0x0000, 0x0000, 0x88bd,
      0xf8bd, 0xf8bd, 0xf8bd, 0x0000, 0x48bd, 0xe7bd, 0xf8bd, 0x97bd, 0x0000,
      0x0000, 0xc8bd, 0x98bd, 0x18bd, 0x0000, 0x0000, 0x98bd, 0xc8bd, 0xd7bd,
      0xf7bd, 0x0000, 0x87bd, 0xf7bd, 0xf8bd, 0x97bd, 0x0000, 0x58bd, 0xf8bd,
      0xe8bd, 0x38bd, 0x0000, 0x0000, 0x98bd, 0xc7bd, 0xa7bd, 0xf8bd, 0x38bd,
      0x88bd, 0xf8bd, 0x87bd, 0xc8bd, 0x27bd, 0x77bd, 0xf7bd, 0xf8bd, 0x98bd,
      0x0000, 0x0000, 0x98bd, 0xc8bd, 0x88bd, 0xf8bd, 0x68bd, 0x98bd, 0xc8bd,
      0x88bd, 0xc8bd, 0x28bd, 0xe8bd, 0xc8bd, 0xa8bd, 0xb8bd, 0x0000, 0x0000,
      0x98bd, 0xc8bd, 0x88bd, 0xf8bd, 0x88bd, 0xc8bd, 0x98bd, 0x28bd, 0xe8bd,
      0x48bd, 0xe7bd, 0xb8bd, 0x88bd, 0xf8bd, 0x28bd, 0x88bd, 0xc8bd, 0xc8bd,
      0x88bd, 0xf8bd, 0x88bd, 0xc8bd, 0x98bd, 0x28bd, 0xf8bd, 0xa8bd, 0xf8bd,
      0x68bd, 0x58bd, 0xe8bd, 0xd8bd, 0xa8bd, 0xf7bd, 0x97bd, 0x0000, 0xf8bd,
      0xc8bd, 0xd8bd, 0x58bd, 0x28bd, 0xf8bd, 0xb8bd, 0xf8bd, 0x57bd, 0x0000,
      0xb8bd, 0xd7bd, 0x38bd, 0xf8bd, 0x98bd, 0x0000, 0xf8bd, 0xf8bd, 0xf8bd,
      0x38bd, 0x18bd, 0xa8bd, 0xf8bd, 0xf7bd, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0xf8bd, 0x98bd, 0x0000, 0xe8bd, 0xf7bd, 0xe8bd, 0x18bd, 0x0000,
      0x68bd, 0xf8bd, 0x78bd, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xf8bd,
      0x88bd, 0x0000, 0x68bd, 0xe7bd, 0x77bd, 0x00ff, 0x0000, 0x28bd, 0x98bd,
      0x28bd, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xf8bd, 0x88bd, 0x0000,
      0x28bd, 0x98bd, 0x57bd, 0x18bc, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0x0000, 0x0000, 0x0000};
  SDL_Surface* icon_surface = SDL_CreateRGBSurfaceFrom(
      (uint16_t*)kIconData, kIconSize, kIconSize, kIconSize, 2 * kIconSize,
      0x0f00, 0x00f0, 0x000f, 0xf000);
  if (icon_surface == NULL) { return; }
  SDL_SetWindowIcon(window, icon_surface);
  SDL_FreeSurface(icon_surface); /* Icon surface no longer required. */
}
