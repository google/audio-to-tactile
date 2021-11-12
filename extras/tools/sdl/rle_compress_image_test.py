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

"""Tests for rle_compress_image."""

import unittest
import numpy as np

from extras.tools.sdl import rle_compress_image


class RleCompressImageTest(unittest.TestCase):

  def test_rle_example(self):
    """Test RLE encoding and decoding on an example sequence."""
    values = np.array([9, 9, 9, 9, 170, 6, 5, 5, 4, 3, 0, 0, 0, 4], np.uint8)
    encoded = rle_compress_image.rle_encode(values)
    decoded = rle_compress_image.rle_decode(encoded)

    np.testing.assert_array_equal(
        np.frombuffer(encoded, np.uint8),
        [128 | (4 - 1), 9,             # Run of four 9's.
         (6 - 1), 170, 6, 5, 5, 4, 3,  # Raw packet [170, 6, 5, 5, 4, 3].
         128 | (3 - 1), 0,             # Run of three 0's.
         (1 - 1), 4,                   # Raw packet [4].
        ])
    np.testing.assert_array_equal(decoded, values)

  def test_rle_single_byte(self):
    """Test RLE when the content is a single byte."""
    values = np.array([42], np.uint8)
    encoded = rle_compress_image.rle_encode(values)
    decoded = rle_compress_image.rle_decode(encoded)

    np.testing.assert_array_equal(
        np.frombuffer(encoded, np.uint8), [(1 - 1), 42])  # Raw packet [42].
    np.testing.assert_array_equal(decoded, values)

  def test_rle_long_run(self):
    """Test run of 300 elements."""
    values = np.full(300, 10, dtype=np.uint8)
    encoded = rle_compress_image.rle_encode(values)
    decoded = rle_compress_image.rle_decode(encoded)

    np.testing.assert_array_equal(
        np.frombuffer(encoded, np.uint8),
        [128 | (128 - 1), 10,  # First packet of 128 elements.
         128 | (128 - 1), 10,  # Another packet of 128 elements.
         128 | (44 - 1), 10,   # A packet of 44 elements.
        ])
    np.testing.assert_array_equal(decoded, values)

  def test_rle_random_sequences(self):
    """Test on random sequences that RLE encoding+decoding recovers original."""
    np.random.seed(0)

    for _ in range(10):
      values = []
      for _ in range(10):
        n = np.random.randint(1, 9)
        values.append(np.full(n, np.random.randint(0, 256), dtype=np.uint8))
        n = np.random.randint(1, 9)
        values.append(np.random.randint(0, 256, size=n, dtype=np.uint8))
      values = np.concatenate(values)

      encoded = rle_compress_image.rle_encode(values)
      decoded = rle_compress_image.rle_decode(encoded)

      np.testing.assert_array_equal(decoded, values)

  def test_compress_image_basic(self):
    image = np.zeros((6, 8), dtype=np.uint8)
    image[1, 1:5] = 17
    image[3, 3] = 254

    encoded = rle_compress_image.compress_image(image)
    decoded, rect = rle_compress_image.decompress_image(encoded)

    x0, y0, width, height = rect
    recovered = np.zeros((6, 8), dtype=np.uint8)
    recovered[y0:(y0 + height), x0:(x0 + width)] = decoded

    np.testing.assert_array_equal(
        np.frombuffer(encoded, np.uint8),
        [0, x0, 0, y0, 0, width, 0, height,  # Rectangle.
         128 | (4 - 1), 17,  # Run of four 17's.
         128 | (6 - 1), 0,   # Run of six 0's.
         (2 - 1), 254, 0,    # Raw packet [254, 0].
        ])
    np.testing.assert_array_equal(recovered, image)


if __name__ == '__main__':
  unittest.main()
