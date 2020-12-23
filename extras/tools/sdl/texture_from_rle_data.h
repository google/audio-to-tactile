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
 * Creates an SDL_Texture from run length-encoded (RLE) data.
 *
 * CreateTextureFromRleData() decompresses data specifying an 8-bit alpha
 * channel and a rectangle. This is returned as a newly created ARGB texture
 * with white color and an SDL_Rect. The texture can be used with
 * SDL_SetTextureColorMod to vary its color. The caller should call
 * SDL_DestroyTexture to release memory when done. On failure, the function
 * prints a message to stderr and returns NULL.
 *
 * The encoded data format is as follows:
 *
 * The first 8 bytes encode a rectangle. Fields x, y, width, height are encoded
 * as big endian 16-bit unsigned integer values.
 *
 *  - The width and height fields specify the image dimensions.
 *
 *  - The x and y fields are ignored by the decompression and may be used for
 *    any purpose, e.g. screen coordinates for where to display the image.
 *
 * The image data follows the rectangle. The image is encoded in TGA format as a
 * series of "packets". There are two kinds: "run-length packets" and "raw
 * packets". Each packet starts with a one-byte packet header. The high bit
 * indicates the kind of packet (0 => raw, 1 => run length). The lower 7 bits
 * encodes the length `n` minus one, so the max possible length is 128 pixels. A
 * packet is allowed to cross scanlines.
 *
 *  - A run-length packet header is followed by a single byte, a pixel value to
 *    be repeated `n` times.
 *
 *  - A raw packet header is followed by `n` bytes for the next `n` pixels.
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
