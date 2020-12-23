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

"""Signal processing utilities."""

from typing import Any, Iterable

import numpy as np


def sliding_window(data: Iterable[Any],
                   window_size: int,
                   hop: int = 1) -> np.ndarray:
  """Converts an array to a sequence of successive overlapping windows.

  Given a 1D data array, this function slides a window of size `window_size`
  and returns a 2D array of shape (num_windows, window_size). The ith row of the
  output is data[hop * i:hop * i + window_size].

  If `data` is N-dimensional with shape (n0, n1, ...), the window slides
  along the first axis and the returned output array is (N+1)-D with shape
  (num_windows, window_size, n1, ...).

  For memory efficiency, the implementation creates the output by pointing into
  `data` without copying [using np.lib.slide_tricks.as_strided].

  Args:
    data: N-D numpy array.
    window_size: Integer.
    hop: Integer, hop step between successive windows.
  Returns:
    Read-only (N+1)-D numpy array pointing into `data`.
  """
  data = np.asarray(data)
  window_size = int(window_size)
  hop = int(hop)
  num_windows = max(0, 1 + (len(data) - window_size) // hop)
  return np.lib.stride_tricks.as_strided(
      data,
      shape=(num_windows, window_size) + data.shape[1:],
      strides=(hop * data.strides[0],) + data.strides,
      writeable=False)
