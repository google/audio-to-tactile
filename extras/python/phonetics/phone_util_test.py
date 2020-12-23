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

"""Tests for phonetics.phone_util."""

import io
import threading

from absl import flags
from absl.testing import absltest
import numpy as np

from extras.python.phonetics import phone_util

flags.DEFINE_integer('dummy_test_flag', 123, '')

FLAGS = flags.FLAGS


class PhoneUtilTest(absltest.TestCase):

  def test_get_main_module_flags_dict(self):
    """Test that get_main_module_flags_dict() sees `FLAGS.dummy_test_flag`."""
    flags_dict = phone_util.get_main_module_flags_dict()
    self.assertIn('dummy_test_flag', flags_dict)
    self.assertEqual(flags_dict['dummy_test_flag'], FLAGS.dummy_test_flag)

  def test_get_phone_label_filename(self):
    """Test get_phone_label_filename()."""
    phn_file = phone_util.get_phone_label_filename('test/foo.wav')
    self.assertEqual(phn_file, 'test/foo.phn')

  def test_get_phone_times(self):
    """Test get_phone_times() on a small (in-memory) phone label file."""
    phone_times = phone_util.get_phone_times(io.StringIO(u"""
0 1502 sil
1502 2492 ae
2492 4000 sil
"""))
    self.assertListEqual(phone_times, [(0, 1502, 'sil'),
                                       (1502, 2492, 'ae'),
                                       (2492, 4000, 'sil')])

  def test_run_in_parallel(self):
    lock = threading.Lock()
    results = []

    def _fun(x):
      y = np.cos(x)

      lock.acquire()
      results.append(y)
      lock.release()

    phone_util.run_in_parallel(range(25), 4, _fun)

    # `results` should be equivalent to np.cos(np.arange(25)), though possibly
    # in a different order.
    self.assertCountEqual(results, np.cos(np.arange(25)))

  def test_balance_weights(self):
    example_counts = {'a': 1240, 'b': 871, 'c': 552, 'd': 104, 'e': 85, 'f': 99}
    fg_classes = ('d', 'a', 'f')
    fg_balance_exponent = 0.85
    bg_fraction = 0.1
    bg_balance_exponent = 0.7
    weights = phone_util.balance_weights(
        example_counts,
        fg_classes=fg_classes,
        fg_balance_exponent=fg_balance_exponent,
        bg_fraction=bg_fraction,
        bg_balance_exponent=bg_balance_exponent)

    # weights are nonnegative, and largest weight is 1.0.
    self.assertGreaterEqual(min(weights.values()), 0.0)
    self.assertAlmostEqual(max(weights.values()), 1.0)

    # Among foreground classes, `weights` is proportional to
    # example_counts**-fg_balance_exponent.
    fg_weights = np.array([weights[k] for k in fg_classes])
    expected = np.array([example_counts[k]**-fg_balance_exponent
                         for k in fg_classes])
    prop_constant = fg_weights.sum() / expected.sum()
    np.testing.assert_allclose(fg_weights, prop_constant * expected)

    # Among background classes, `weights` is proportional to
    # example_counts**-bg_balance_exponent.
    bg_classes = set(example_counts.keys()) - set(fg_classes)
    bg_weights = np.array([weights[k] for k in bg_classes])
    expected = np.array([example_counts[k]**-bg_balance_exponent
                         for k in bg_classes])
    prop_constant = bg_weights.sum() / expected.sum()
    np.testing.assert_allclose(bg_weights, prop_constant * expected)

    # `weights` should be such that num_bg / num_total == bg_fraction.
    num_total = sum(weights[k] * example_counts[k] for k in weights)
    num_bg = sum(weights[k] * example_counts[k] for k in bg_classes)
    self.assertAlmostEqual(num_bg / num_total, bg_fraction)


class DatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self.num_frames_left_context = 2
    self.num_channels = 2
    self.num_frames = self.num_frames_left_context + 1
    self.examples = {
        'ae': np.random.rand(
            70, self.num_frames, self.num_channels).astype(np.float32),
        'sil': np.random.rand(
            200, self.num_frames, self.num_channels).astype(np.float32),
        # Dataset will drop this class because it is empty.
        'z': np.zeros((0, self.num_frames, self.num_channels)),
    }
    self.metadata = {
        'num_frames_left_context': self.num_frames_left_context,
        'num_channels': self.num_channels,
    }

  def test_basic(self):
    """Test creating, writing, and reading the Dataset class."""
    # Create Dataset.
    dataset = phone_util.Dataset(self.examples, self.metadata)

    self.assertNotIn('z', dataset.examples)
    self.assertEqual(dataset.num_frames, self.num_frames)
    self.assertEqual(dataset.num_channels, self.num_channels)
    self.assertDictEqual(dataset.example_counts, {'ae': 70, 'sil': 200})

    # Write Dataset to in-memory .npz file and then read it.
    npz = io.BytesIO()
    dataset.write_npz(npz)
    npz.seek(0)  # Rewind to beginning of file for reading.
    recovered = phone_util.read_dataset_npz(npz)

    np.testing.assert_array_equal(recovered.examples['ae'],
                                  self.examples['ae'])
    np.testing.assert_array_equal(recovered.examples['sil'],
                                  self.examples['sil'])
    self.assertDictEqual(recovered.metadata, self.metadata)
    self.assertEqual(recovered.num_frames, self.num_frames)
    self.assertEqual(recovered.num_channels, self.num_channels)

  def test_get_xy_arrays(self):
    """Test Dataset.get_xy_arrays() method."""
    dataset = phone_util.Dataset(self.examples, self.metadata)

    x, y = dataset.get_xy_arrays(['sil', 'z', 'ae'])

    expected_x = np.concatenate((self.examples['sil'], self.examples['ae']))
    expected_y = np.hstack(([0] * 200, [2] * 70))

    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)

    x, y = dataset.get_xy_arrays(['sil', 'z', 'ae'], shuffle=True)

    self.assertEqual(x.shape, (270, self.num_frames, self.num_channels))

  def test_subsample(self):
    """Test Dataset.subsample() method."""
    self.examples['x'] = np.random.rand(
        20, self.num_frames, self.num_channels).astype(np.float32)
    dataset = phone_util.Dataset(self.examples, self.metadata)

    dataset.subsample({'ae': 0.6, 'sil': 0.1})

    self.assertNotIn('x', dataset.examples)
    self.assertEqual(dataset.example_counts, {'ae': round(0.6 * 70),
                                              'sil': round(0.1 * 200)})

  def test_split(self):
    """Test Dataset.split() method."""
    dataset = phone_util.Dataset(self.examples, self.metadata)

    dataset_a, dataset_b = dataset.split(0.1)

    self.assertEqual(dataset_a.example_counts, {'ae': 7, 'sil': 20})
    self.assertEqual(dataset_b.example_counts, {'ae': 63, 'sil': 180})
    self.assertEqual(dataset_a.metadata, dataset.metadata)
    self.assertEqual(dataset_b.metadata, dataset.metadata)

  def test_missing_metadata(self):
    """Test that read_dataset_npz raises an error when metadata is missing."""
    npz = io.BytesIO()
    np.savez(npz, **self.examples)
    npz.seek(0)

    with self.assertRaisesRegex(ValueError, 'dataset_metadata missing'):
      phone_util.read_dataset_npz(npz)

  def test_nonfinite(self):
    """Test that Dataset.write_npz raises if examples have nonfinite values."""
    dataset = phone_util.Dataset({}, self.metadata)
    dataset.examples['ae'] = np.full(
        (1, self.num_frames, self.num_channels), np.nan)

    with self.assertRaisesRegex(ValueError, 'nonfinite value'):
      dataset.write_npz(io.BytesIO())


if __name__ == '__main__':
  absltest.main()
