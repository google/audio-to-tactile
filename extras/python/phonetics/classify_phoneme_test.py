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

"""Tests for phoneme classifier Python bindings."""

import unittest
import numpy as np

from extras.python import dsp
from extras.python import frontend
from extras.python.phonetics import classify_phoneme
from extras.python.phonetics.sliding_window import sliding_window

CLASSIFIER_INPUT_HZ = 16000


class ClassifyPhonemeTest(unittest.TestCase):

  def _run_phoneme(self, phoneme: str) -> None:
    """A forgiving test that the classifier is basically working.

    Runs CarlFrontend + ClassifyPhoneme on a short WAV recording of a pure
    phone, and checks that a moderately confident score is sometimes given to
    the correct label.

    Args:
      phoneme: String, name of the phoneme to test.
    """
    wav_file = (
        f'extras/test/testdata/phone_{phoneme}.wav'
    )
    samples, sample_rate_hz = dsp.read_wav_file(wav_file, dtype=np.float32)
    samples = samples.mean(axis=1)
    self.assertEqual(sample_rate_hz, CLASSIFIER_INPUT_HZ)

    # Run frontend to get CARL frames. The classifier expects input sample rate
    # CLASSIFIER_INPUT_HZ, block_size=128, pcen_cross_channel_diffusivity=60,
    # and otherwise the default frontend settings.
    carl = frontend.CarlFrontend(input_sample_rate_hz=CLASSIFIER_INPUT_HZ,
                                 block_size=128,
                                 pcen_cross_channel_diffusivity=60.0)
    self.assertEqual(carl.num_channels, classify_phoneme.NUM_CHANNELS)
    samples = samples[:len(samples) - len(samples) % carl.block_size]
    frames = carl.process_samples(samples)

    count_correct = 0
    count_total = 0
    for window in sliding_window(frames, classify_phoneme.NUM_FRAMES):
      scores = classify_phoneme.classify_phoneme_scores(window)
      # Count as "correct" if correct label's score is moderately confident.
      count_correct += (scores['phoneme'][phoneme] > 0.1)
      count_total += 1

    self.assertCountEqual(scores['phoneme'].keys(), classify_phoneme.PHONEMES)
    self.assertCountEqual(scores['manner'].keys(), classify_phoneme.MANNERS)
    self.assertCountEqual(scores['place'].keys(), classify_phoneme.PLACES)

    accuracy = float(count_correct) / count_total
    self.assertGreaterEqual(accuracy, 0.6)

  def test_phoneme_ae(self):
    self._run_phoneme('ae')

  def test_phoneme_er(self):
    self._run_phoneme('er')

  def test_phoneme_z(self):
    self._run_phoneme('z')

  def test_label_output(self):
    np.random.seed(0)

    for _ in range(5):
      frames = np.random.rand(classify_phoneme.NUM_FRAMES,
                              classify_phoneme.NUM_CHANNELS)

      labels = classify_phoneme.classify_phoneme_labels(frames)
      scores = classify_phoneme.classify_phoneme_scores(frames)

      self.assertIn(labels['phoneme'], classify_phoneme.PHONEMES)
      self.assertIn(labels['manner'], classify_phoneme.MANNERS)
      self.assertIn(labels['place'], classify_phoneme.PLACES)

      score_argmax = max(scores['phoneme'], key=scores['phoneme'].get)
      self.assertEqual(labels['phoneme'], score_argmax)


if __name__ == '__main__':
  unittest.main()
