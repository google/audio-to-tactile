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

#include "audio/tactile/sdl/texture_from_rle_data.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio/dsp/portable/serialize.h"

SDL_Texture* CreateTextureFromRleData(const uint8_t* encoded,
                                      SDL_Renderer* renderer, SDL_Rect* rect) {
  /* Decode the rectangle from the first 8 bytes. */
  rect->x = BigEndianReadU16(encoded);
  rect->y = BigEndianReadU16(encoded + 2);
  rect->w = BigEndianReadU16(encoded + 4);
  rect->h = BigEndianReadU16(encoded + 6);
  encoded += 8;

  uint32_t* argb_data = NULL;
  SDL_Texture* texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                        SDL_TEXTUREACCESS_STATIC, rect->w, rect->h);
  if (!texture) {
    fprintf(stderr, "Error: %s\n", SDL_GetError());
    goto fail;
  }

  const size_t num_pixels = (size_t)rect->w * (size_t)rect->h;
  const size_t pitch = sizeof(uint32_t) * rect->w;
  argb_data = (uint32_t*)malloc(pitch * rect->h);
  if (argb_data == NULL) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    goto fail;
  }

  /* Image data is run-length encoded in TGA format. Pixels are encoded in
   * "packets". There are two kinds: "run-length packets" and "raw packets".
   */
  size_t i;
  for (i = 0; i < num_pixels;) {
    /* Each packet starts with a one-byte packet header. The high bit indicates
     * the kind of packet. The lower 7 bits encodes the length `n` minus one.
     */
    const unsigned packet_header = *(encoded++);
    const int is_run = packet_header >> 7;
    int n = 1 + (packet_header & 0x7f);
    if (i + n > num_pixels) {
      fprintf(stderr, "Error: Corrupt image data\n");
      goto fail;
    }
    if (is_run) { /* Decode a run-length packet. */
      const uint32_t value = ((uint32_t)*(encoded++) << 24) | 0xffffffUL;
      do {
        argb_data[i++] = value;
      } while (--n);
    } else { /* Decode a raw packet. */
      do {
        const uint32_t value = ((uint32_t)*(encoded++) << 24) | 0xffffffUL;
        argb_data[i++] = value;
      } while (--n);
    }
  }

  if ((SDL_UpdateTexture(texture, NULL, argb_data, pitch) != 0) ||
      (SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND) != 0)) {
    fprintf(stderr, "Error: %s\n", SDL_GetError());
    goto fail;
  }

  free(argb_data);
  return texture;

fail:
  free(argb_data);
  if (texture) {
    SDL_DestroyTexture(texture);
    texture = NULL;
  }
  return NULL;
}
