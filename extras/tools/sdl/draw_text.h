/* Copyright 2019, 2021 Google LLC
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
 * Simple function for drawing monospaced text to an SDL render target.
 *
 * SDL has functions for rendered text (e.g. with TTF_RenderText_Solid), which
 * allocate a new surface and render a given string to it. This makes sense for
 * UI with fixed strings that can be prepared up front, but is inconvenient for
 * dynamically-changing text such as meter displays.
 *
 * `DrawText()` is a simple function that is good for such dynamic text. It
 * efficiently renders given text to the current rendering target, without
 * allocating resources.
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_DRAW_TEXT_H_
#define AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_DRAW_TEXT_H_

#include "SDL2/SDL.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize font texture needed for `DrawText()`. */
int DrawTextInitFontTexture(SDL_Renderer* renderer);

/* Frees the font texture. */
void DrawTextFreeFontTexture(void);

/* Draws formatted text to the current rendering target. The text is a
 * monospaced font with characters of size 12x20 (a decently large and readable
 * size), and supports printable ASCII characters. The upper left corner of the
 * text is `x`, `y`.
 *
 * Notes:
 *  - `DrawTextInitFontTexture()` must be called first.
 *  - Use `DrawTextSetColor()` to set the text color.
 *  - Codes \x7e, \x7f, \x80, \x81 print respectively left, up, right, down
 *    arrow symbols. For instance, `DrawText(r, x, y, "Use \x7e \x80 arrows")`.
 */
void DrawText(SDL_Renderer* renderer, int x, int y, const char* format, ...);

/* Sets the text color. Returns 1 on success, 0 on failure. */
int DrawTextSetColor(uint8_t r, uint8_t g, uint8_t b);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_DRAW_TEXT_H_ */
