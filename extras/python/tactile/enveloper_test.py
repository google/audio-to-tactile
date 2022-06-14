# Copyright 2020-2021 Google LLC
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

"""Tests for Enveloper Python bindings."""

from typing import Tuple
import unittest

import numpy as np

from extras.python.tactile import enveloper


def compute_max(x: np.ndarray,
                t_range: Tuple[float, float],
                sample_rate_hz: float) -> float:
  i_start, i_end = (np.array(t_range) * sample_rate_hz).astype(int)
  return x[i_start:i_end].max()


class EnveloperTest(unittest.TestCase):

  def test_basic(self):
    for decimation_factor in (1, 2, 4):
      for sample_rate_hz in (16000.0, 44100.0, 48000.0):
        env = enveloper.Enveloper(sample_rate_hz, decimation_factor)
        output_rate = env.output_sample_rate_hz
        self.assertAlmostEqual(output_rate, sample_rate_hz / decimation_factor,
                               0.01)
        output_frames = int(0.5 * output_rate)
        num_samples = output_frames * decimation_factor
        num_channels = enveloper.NUM_CHANNELS

        for c, test_frequency in [(0, 80.0), (1, 1500.0), (3, 5000.0)]:
          radians_per_second = 2 * np.pi * test_frequency
          t = np.arange(num_samples) / sample_rate_hz
          input_samples = (
              1e-5 * (np.random.rand(num_samples) - 0.5) +
              0.2 * np.sin(radians_per_second * t) * (t > 0.25))

          env.reset()
          output = env.process_samples(input_samples)
          self.assertEqual(output.shape, (output_frames, num_channels))

          # For t < 0.05, all output is close to zero.
          for i in range(num_channels):
            max_i = compute_max(output[:, i], [0.0, 0.05], output_rate)
            self.assertLess(max_i, 0.005)

          # For t > 0.05, channel c is strongest.
          max_c = compute_max(output[:, c], [0.07, 0.28], output_rate)
          self.assertGreater(max_c, 0.2)
          for i in (0, 1, 3):
            if i != c:
              max_i = compute_max(output[:, i], [0.07, 0.28], output_rate)
              self.assertGreater(max_c, 1.1 * max_i)

        debug_out = {}
        env.reset()
        np.testing.assert_array_equal(
            output, env.process_samples(input_samples, debug_out=debug_out))
        # debug_out is nonempty and has array entries of the expected shape.
        self.assertGreater(len(debug_out), 0)
        for _, v in debug_out.items():
          self.assertTupleEqual(v.shape, output.shape)


if __name__ == '__main__':
  unittest.main()
