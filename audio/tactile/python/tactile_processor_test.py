r"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Tests for TactileProcessor Python bindings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import six

from audio.dsp.portable.python import wav_io
from audio.tactile.python import tactile_processor


class TactileProcessorTest(unittest.TestCase):

  def test_phone(self):
    for phone, intended_tactor in [('aa', 1), ('eh', 5), ('uw', 2)]:
      input_samples, input_sample_rate_hz = wav_io.read_wav_file(
          'audio/tactile/testdata/phone_%s.wav' % phone)
      input_samples = input_samples[:, 0]

      for decimation_factor in [1, 2, 4, 8]:
        processor = tactile_processor.TactileProcessor(
            input_sample_rate_hz=input_sample_rate_hz,
            decimation_factor=decimation_factor)
        block_size = processor.block_size
        output_block_size = block_size // decimation_factor

        energy = np.zeros(tactile_processor.NUM_TACTORS)

        start = 0
        while start + block_size < len(input_samples):
          block_end = start + block_size
          input_block = input_samples[start:block_end]
          # Convert to floats in [-1, 1].
          input_block = input_block.astype(np.float32) / 2.0**15

          # Run audio-to-tactile processing in a streaming manner.
          output_block = processor.process_samples(input_block)

          self.assertEqual(output_block.shape[0], output_block_size)
          self.assertEqual(output_block.shape[1], tactile_processor.NUM_TACTORS)

          # Accumulate energy for each channel.
          energy += (output_block**2).sum(axis=0)
          start = block_end

        # The intended tactor has the largest energy in the vowel cluster.
        for c in range(1, 8):
          if c != intended_tactor:
            self.assertGreaterEqual(energy[intended_tactor], 1.65 * energy[c])

  def test_streaming(self):
    np.random.seed(0)
    block_size = 16
    num_blocks = 12
    decimation_factor = 2
    input_samples = np.random.uniform(-0.1, 0.1, num_blocks * block_size)

    processor = tactile_processor.TactileProcessor(
        block_size=block_size, decimation_factor=decimation_factor)
    self.assertEqual(processor.input_sample_rate_hz, 16000.0)
    self.assertEqual(processor.output_sample_rate_hz, 8000.0)
    self.assertEqual(processor.block_size, block_size)
    self.assertEqual(processor.decimation_factor, decimation_factor)

    nonstreaming_outputs = processor.process_samples(input_samples)
    self.assertEqual(nonstreaming_outputs.shape, (
        len(input_samples) // decimation_factor, tactile_processor.NUM_TACTORS))
    for output in nonstreaming_outputs:
      self.assertTrue(np.all(np.isfinite(output)))

    processor.reset()
    streaming_outputs = np.vstack([
        processor.process_samples(input_block)
        for input_block in input_samples.reshape(num_blocks, block_size)])

    np.testing.assert_allclose(
        streaming_outputs, nonstreaming_outputs, atol=1e-9)

  def test_bad_input(self):
    processor = tactile_processor.TactileProcessor()

    with six.assertRaisesRegex(self, ValueError, 'expected 1-D array'):
      processor.process_samples(np.zeros((5, 5)))

    with six.assertRaisesRegex(self, ValueError, 'multiple of block_size'):
      processor.process_samples(np.zeros(19))


if __name__ == '__main__':
  unittest.main()
