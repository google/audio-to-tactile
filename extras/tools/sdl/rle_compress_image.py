# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Run-length encoding (RLE) image compression.

`compress_image()` compresses a grayscale image using run-length encoding,
which can then be decompressed using `decompress_image()` or the C function
`CreateTextureFromRleData()` in texture_from_rle_data.c. This compression is
useful for images where there are large contiguous regions of constant value.

For convenience, this library can also run as a command line program: the
following compresses "input.png" and prints the compressed data to stdout.

  python3 rle_compress_image.py input.png

NOTE: Don't use a lossy format like JPEG as input. It will compress poorly.

The encoded format is as follows. The first 8 bytes encode a rectangle.
Fields x, y, width, height are encoded as big endian uint16 values. Image
data within the rectangle is encoded in TGA format as a series of "packets".
There are two kinds: "run-length packets" and "raw packets". Each packet
starts with a one-byte packet header. The high bit indicates the kind of
packet (0 => raw, 1 => run length). The lower 7 bits encodes the length `n`
minus one, so the max possible length is 128 pixels. A packet is allowed to
cross scanlines.

 * A run-length packet header is followed by a single byte, a pixel value to
   be repeated `n` times.

 * A raw packet header is followed by `n` bytes for the next `n` pixels.
"""

import os.path
import struct
import textwrap
from typing import Sequence, Tuple

from absl import app
import numpy as np
from PIL import Image


def compress_image(image: np.ndarray) -> bytes:
  """Compresses an image using the RLE format described above.

  Args:
    image: 2D numpy array of uint8 values.
  Returns:
    RLE-encoded data.
  """
  image, rect = crop_to_content(image)
  return struct.pack('>HHHH', *rect) + rle_encode(image.flatten())


def decompress_image(
    encoded: bytes) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
  """Decompresses an image encoded by rle_compress_image().

  Args:
    encoded: bytes, RLE-encoded data.
  Returns:
    2-tuple of a cropped image and a rectangle (x0, y0, width, height)
    indicating the position within the original image.
  """
  rect = struct.unpack('>HHHH', encoded[:8])
  content = rle_decode(encoded[8:])
  _, _, width, height = rect
  cropped_image = content.reshape(height, width)
  return cropped_image, rect


def crop_to_content(
    image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
  """Crops image to content, trimming margins of 0-valued pixels.

  Args:
    image: 2D numpy array of uint8 values.
  Returns:
    2-tuple of a cropped image and a rectangle (x0, y0, width, height)
    indicating the position within the original image.
  """
  image = np.asarray(image)
  mask = (image > 0)

  if not np.any(mask):
    x0, y0, width, height = 0, 0, 1, 1
  else:
    x0, x1 = np.where(np.any(mask, axis=0))[0][[0, -1]]
    y0, y1 = np.where(np.any(mask, axis=1))[0][[0, -1]]
    width = x1 - x0 + 1
    height = y1 - y0 + 1

  cropped_image = image[y0:(y0 + height), x0:(x0 + width)]
  rect = x0, y0, width, height
  return cropped_image, rect


def rle_encode(values: np.ndarray) -> bytes:
  """Run-length encodes a sequence of unsigned 8-bit values.

  Args:
    values: 1D array with values in the range 0 to 255.
  Returns:
    RLE-encoded data.
  """
  values = np.asarray(values, dtype=np.uint8)
  if len(values) >= 3:
    # Encoding a run needs a 1-byte header, so it is only advantageous to encode
    # runs of 3 or more elements. The following determines `is_run` so that
    #   is_run[i] = (values[i] == values[i + 1] == values[i + 2]).
    is_run = np.convolve([1, 1, 0], np.diff(values) == 0)[1:] == 2
  else:
    is_run = [False] * len(values)

  encoded = []

  i = 0
  while i < len(values):
    # Encode either a "run length" or "raw" packet of up to 128 elements.
    if is_run[i]:  # Encode a run-length packet.
      n = 1
      while i + n < len(values) and values[i] == values[i + n] and n < 128:
        n += 1
      encoded.append(bytes([127 + n, values[i]]))
    else:  # Encode a raw packet.
      n = 1
      while i + n < len(values) and not is_run[i + n] and n < 128:
        n += 1
      encoded.append(bytes([n - 1]) + values[i:i + n].tobytes())
    i += n

  return b''.join(encoded)


def rle_decode(encoded: bytes) -> np.ndarray:
  """Decodes run-length encoded data, inverse of `rle_encode()`.

  Args:
    encoded: bytes, RLE-encoded data.
  Returns:
    1D array of decoded values.
  """
  encoded = np.frombuffer(bytes(encoded), np.uint8)
  result = []

  i = 0
  while i < len(encoded):
    packet_header = encoded[i]
    i += 1
    n = 1 + (packet_header & 0x7f)  # Number of elements in the next packet.

    if packet_header >> 7:  # Decode a run-length packet.
      content = np.full(n, encoded[i], np.uint8)
      i += 1
    else:  # Decode a raw packet.
      content = encoded[i:i + n]
      i += n

    result.append(content)

  return np.concatenate(result)


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    print('Use: rle_compress_image.py input.png')
    return

  file_name = argv[1]
  image = np.array(Image.open(file_name).convert('L'))
  data = compress_image(image)

  # To get a reasonable name for the generated array, extract the file name stem
  # and convert it to CamelCase.
  name = os.path.splitext(os.path.basename(file_name))[0]
  name = ''.join(word.title() for word in name.replace('.', '_').split('_'))
  # Generate array C code for embedding the data.
  output = 'static const uint8_t k{name}[{size}] = {{{data}}};'.format(
      name=name,
      size=len(data),
      data=', '.join(f'0x{value:02x}' for value in data))
  # Wrap lines to 80 chars.
  print(textwrap.fill(output, width=80, subsequent_indent='    ') + '\n')


if __name__ == '__main__':
  app.run(main)
