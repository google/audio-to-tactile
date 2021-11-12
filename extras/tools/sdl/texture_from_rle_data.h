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
 * Creates an SDL_Texture from run length-encoded (RLE) data.
 *
 * CreateTextureFromRleData() decompresses data specifying an 8-bit alpha
 * channel and a rectangle. This is returned as a newly created ARGB texture
 * with white color and an SDL_Rect. The texture can be used with
 * SDL_SetTextureColorMod to vary its color. The caller should call
 * SDL_DestroyTexture to release memory when done. On failure, the function
 * prints a message to stderr and returns NULL.
 *
 * The encoded data format is described in rle_compress_image.py.
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_TEXTURE_FROM_RLE_DATA_H_
#define AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_TEXTURE_FROM_RLE_DATA_H_

#include "SDL2/SDL.h"

#ifdef __cplusplus
extern "C" {
#endif

SDL_Texture* CreateTextureFromRleData(const uint8_t* encoded,
                                      SDL_Renderer* renderer, SDL_Rect* rect);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_EXTRAS_TOOLS_SDL_TEXTURE_FROM_RLE_DATA_H_ */
