# Copyright 2019 Google LLC
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

"""Tests for vowel embedding Python bindings."""

import unittest
import numpy as np

from extras.python import dsp
from extras.python import frontend
from extras.python.phonetics import embed_vowel


class EmbedVowelTest(unittest.TestCase):

  def _run_phone(self, phone):
    self.assertEqual(len(embed_vowel.TARGET_NAMES), embed_vowel.NUM_TARGETS)
    self.assertEqual(len(embed_vowel.TARGET_COORDS), embed_vowel.NUM_TARGETS)
    phone_index = embed_vowel.TARGET_NAMES.index(phone)

    wav_file = (
        f'extras/test/testdata/phone_{phone}.wav')
    samples, _ = dsp.read_wav_file(wav_file, dtype=np.float32)
    samples = samples.mean(axis=1)

    carl = frontend.CarlFrontend()
    self.assertEqual(carl.num_channels, embed_vowel.NUM_CHANNELS)
    samples = samples[:len(samples) - len(samples) % carl.block_size]
    frames = carl.process_samples(samples)
    coords = embed_vowel.embed_vowel(frames)

    distance_from_targets = np.linalg.norm(
        coords[:, np.newaxis, :]
        - embed_vowel.TARGET_COORDS[np.newaxis, :, :],
        axis=-1)

    # Compare `coords` with embed_vowel_scores.
    scores = embed_vowel.embed_vowel_scores(frames)
    self.assertEqual(scores.shape, (frames.shape[0], embed_vowel.NUM_TARGETS))
    np.testing.assert_array_equal(scores.argmax(axis=1),
                                  distance_from_targets.argmin(axis=1))
    np.testing.assert_allclose(
        scores, np.exp(-4.0 * distance_from_targets), atol=1e-6)

    # Compute L2 distance from intended_target.
    distance_from_intended = distance_from_targets[:, phone_index]
    # Mean distance from the intended target is not too large.
    self.assertLessEqual(distance_from_intended.mean(), 0.35)

    other = [j for j in range(embed_vowel.NUM_TARGETS) if j != phone_index]
    min_distance_from_other = distance_from_targets[:, other].min(axis=1)
    accuracy = float(np.count_nonzero(
        distance_from_intended < min_distance_from_other)) / len(frames)
    # Coordinate is closest to the indended target >= 70% of the time.
    self.assertGreaterEqual(accuracy, 0.7)

  def test_phone_aa(self):
    self._run_phone('aa')

  def test_phone_uw(self):
    self._run_phone('uw')

  def test_phone_er(self):
    self._run_phone('er')

  def test_phone_ih(self):
    self._run_phone('ih')

  def test_phone_iy(self):
    self._run_phone('iy')

  def test_phone_eh(self):
    self._run_phone('eh')

  def test_phone_ae(self):
    self._run_phone('ae')

  def test_phone_uh(self):
    self._run_phone('uh')


if __name__ == '__main__':
  unittest.main()
