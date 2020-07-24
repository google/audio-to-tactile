# Lint as: python3
r"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Test for read_nist_sphere.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import unittest
import numpy as np

from audio.tactile.python import read_nist_sphere


class ReadNistSphereTest(unittest.TestCase):

  def test_basic(self):
    expected = np.array([0, 1, 2, -1, -32768, 32767], np.int16)
    f = io.BytesIO(b'NIST_1A\n' +
                   b'     104\n' +
                   b'channel_count -i 1\n' +
                   b'sample_count -i 6\n' +
                   b'sample_n_bytes -i 2\n' +
                   b'sample_rate -i 8000\n' +
                   b'end_head\n\n\n' +
                   expected.tobytes())

    sample_rate_hz, samples = read_nist_sphere.read_nist_sphere(f)

    self.assertEqual(sample_rate_hz, 8000)
    np.testing.assert_equal(samples, expected)


if __name__ == '__main__':
  unittest.main()
