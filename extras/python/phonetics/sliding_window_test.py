# Copyright 2020 Google LLC
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

"""Tests for tactile.python."""

import unittest

import numpy as np

from extras.python.phonetics.sliding_window import sliding_window


class SlidingWindowTest(unittest.TestCase):

  def test_sliding_window_basic(self):
    """Test basic calls of sliding_window where the input is 1D."""
    data = [1, 2, 3, 4, 5, 6, 7, 8]

    out = sliding_window(data, 3)
    np.testing.assert_array_equal(
        out, [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])

    out = sliding_window(data, 3, hop=2)
    np.testing.assert_array_equal(out, [[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    out = sliding_window(data, 4, hop=3)
    np.testing.assert_array_equal(out, [[1, 2, 3, 4], [4, 5, 6, 7]])

  def test_sliding_window_multichannel(self):
    """Test where input is 2D, representing multiple channels."""
    num_channels = 5
    data = np.arange(6 * num_channels).reshape(-1, num_channels)

    out = sliding_window(data, 3, hop=2)
    np.testing.assert_array_equal(out, [[[0, 1, 2, 3, 4],
                                         [5, 6, 7, 8, 9],
                                         [10, 11, 12, 13, 14]],
                                        [[10, 11, 12, 13, 14],
                                         [15, 16, 17, 18, 19],
                                         [20, 21, 22, 23, 24]]])

  def test_sliding_window_noncontiguous_input(self):
    """Test noncontiguous input, important because code uses stride_tricks."""
    memory = np.arange(100)
    s = memory.strides[0]  # Stride in bytes between successive elements.
    # Create a weirdly-strided 7x4 array by pointing into `memory`.
    data = np.lib.stride_tricks.as_strided(
        memory, shape=(7, 4), strides=(2 * s, 9 * s), writeable=False)
    np.testing.assert_array_equal(data, [[0, 9, 18, 27],
                                         [2, 11, 20, 29],
                                         [4, 13, 22, 31],
                                         [6, 15, 24, 33],
                                         [8, 17, 26, 35],
                                         [10, 19, 28, 37],
                                         [12, 21, 30, 39]])

    out = sliding_window(data, 3, hop=2)
    np.testing.assert_array_equal(out, [[[0, 9, 18, 27],
                                         [2, 11, 20, 29],
                                         [4, 13, 22, 31]],
                                        [[4, 13, 22, 31],
                                         [6, 15, 24, 33],
                                         [8, 17, 26, 35]],
                                        [[8, 17, 26, 35],
                                         [10, 19, 28, 37],
                                         [12, 21, 30, 39]]])

  def test_sliding_window_empty_output(self):
    """Test window size longer than the data, resulting in empty output."""
    data = np.arange(24).reshape(4, 3, 2)

    out = sliding_window(data, 5)
    self.assertEqual(out.shape, (0, 5, 3, 2))


if __name__ == '__main__':
  unittest.main()
