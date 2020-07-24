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


Tests for stats.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from audio.tactile.python import stats


def _one_hot(indices, depth):
  """Convert indices to one-hot representation."""
  result = np.zeros((len(indices), depth))
  np.put(result, np.arange(len(indices)) * depth + indices, 1.0)
  return result


class StatsTest(unittest.TestCase):

  def test_estimate_information_transfer(self):
    np.random.seed(0)
    for num_rows, num_cols in ((2, 2), (5, 8), (8, 5), (20, 20)):
      cm = np.random.randint(99, size=(num_rows, num_cols))**2 // 100

      it = stats.estimate_information_transfer(cm)

      n = np.sum(cm)
      expected_it = 0.0
      for i in range(num_rows):
        for j in range(num_cols):
          n_ij = cm[i, j]
          if n_ij:
            n_i = np.sum(cm[i, :])
            n_j = np.sum(cm[:, j])
            expected_it += (n_ij / n) * np.log2((n_ij * n) / (n_i * n_j))

      self.assertAlmostEqual(it, expected_it)

  def test_mds_confusion(self):
    np.random.seed(0)
    confusion_matrix = np.array([[0.7, 0.3, 0.0, 0.0, 0.0],
                                 [0.2, 0.8, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.9, 0.1, 0.0],
                                 [0.0, 0.0, 0.0, 0.6, 0.4],
                                 [0.0, 0.0, 0.0, 0.4, 0.6]])

    for _ in range(3):
      coords = stats.mds_confusion(confusion_matrix)

      # Compute distance[i, j] = Euclidean distance between ith and jth coords.
      distance = np.linalg.norm(
          coords[np.newaxis, :, :] - coords[:, np.newaxis, :], axis=-1)
      # Classes 0 and 1 are closer together than to other classes.
      self.assertLess(distance[0, 1], 0.9 * distance[:2, 2:].min())
      # Classes 3 and 4 are closer together than to other classes.
      self.assertLess(distance[3, 4], 0.9 * distance[:3, 3:].min())
      # Distance between classes 3 and 4 is less than classes 0 and 1.
      self.assertLess(distance[3, 4], 0.9 * distance[0, 1])

  def test_basic(self):
    np.random.seed(0)
    for sigma in (0.1, 0.2, 0.3):
      # Generate labeled examples of two moderately imbalanced classes. The
      # scores are around 0.7 for positive examples or 0.3 for negative examples
      # with normal noise of stddev, so there is some overlap between classes.
      labels = (np.random.rand(500) > 0.3)
      distance = 0.4
      scores = np.random.normal(
          0.3 + distance * labels, sigma).astype(np.float32)

      s = stats.BinaryClassifierStats()
      s.accum(labels, scores)  # Accumulate all the stats.

      false_positive_rate, true_positive_rate, thresholds = s.roc_curve

      expected_fpr = np.empty_like(thresholds)
      expected_tpr = np.empty_like(thresholds)
      num_positives = np.sum(labels)
      num_negatives = len(labels) - num_positives

      for i, thresh in enumerate(thresholds):
        predicted = (scores >= thresh).astype(float)
        # Manually compute the expected false positive and true positive rates.
        expected_fpr[i] = predicted.dot(np.logical_not(labels)) / num_negatives
        expected_tpr[i] = predicted.dot(labels) / num_positives

      np.testing.assert_allclose(false_positive_rate, expected_fpr, atol=1e-6)
      np.testing.assert_allclose(true_positive_rate, expected_tpr, atol=1e-6)

      # Estimated d' is within 20% of theoretically expected value.
      expected_d_prime = distance / sigma
      self.assertAlmostEqual(s.d_prime, expected_d_prime,
                             delta=0.2 * expected_d_prime)

      # Shuffle the data.
      i = np.random.permutation(len(labels))
      labels = labels[i]
      scores = scores[i]

      labels_batches = labels.reshape(100, -1)
      scores_batches = scores.reshape(100, -1)
      s.reset()
      # Accumulate stats over 100 separate batches.
      for labels_batch, scores_batch in zip(labels_batches, scores_batches):
        s.accum(labels_batch, scores_batch)

      # Results should be same as before.
      false_positive_rate2, true_positive_rate2, thresholds2 = s.roc_curve
      np.testing.assert_array_equal(
          false_positive_rate, false_positive_rate2)
      np.testing.assert_array_equal(
          true_positive_rate, true_positive_rate2)
      np.testing.assert_array_equal(thresholds, thresholds2)

  def test_multiclass_dprime(self):
    np.random.seed(0)
    num_classes = 4
    labels = np.random.randint(num_classes, size=10000)
    distance = np.array([0.3, 0.4, 0.5, 0.6])
    sigma = 0.2
    scores = _one_hot(labels, num_classes) * np.expand_dims(distance, axis=0)
    scores += sigma * np.random.randn(*scores.shape)

    s = stats.MulticlassClassifierStats(num_classes)
    self.assertEqual(s.num_classes, num_classes)
    s.accum(labels, scores)  # Accumulate all the stats.

    # Estimated mean d' is within 5% of theoretically expected value.
    expected_mean_d_prime = np.mean(distance / sigma)
    self.assertAlmostEqual(np.mean(s.d_prime), expected_mean_d_prime,
                           delta=0.05 * expected_mean_d_prime)

  def test_multiclass_confusion_matrix(self):
    labels = [1, 2, 4, 4]
    scores = [[-0.3, 0.0, 4.6, 0.0, 0.0],
              [0.0, 0.0, 3.4, 0.0, 0.1],
              [0.2, 0.0, 0.0, 0.0, 7.5],
              [0.0, 0.0, 0.0, 0.0, 7.2]]

    s = stats.MulticlassClassifierStats(5)
    np.testing.assert_array_equal(
        s.confusion_matrix, np.zeros((5, 5)))

    s.accum(labels, scores)
    np.testing.assert_array_equal(
        s.confusion_matrix,
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 2]])

    s.accum([2, 3], [[0.0, 0.1, 7.0, 0.0, 0.0],
                     [0.0, 7.0, 0.0, 0.0, 0.3]])
    np.testing.assert_array_equal(
        s.confusion_matrix,
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 2]])

  def test_random_sampler_basic(self):
    np.random.seed(0)
    sampler = stats.RandomSampler(4)
    self.assertEqual(sampler.limit, 4)
    self.assertEqual(sampler.total_count, 0)

    # Push fewer elements than the limit. The sample should just be [0, 1, 2].
    sampler.push([0, 1, 2])
    self.assertEqual(sampler.total_count, 3)
    sample = sampler.sample
    np.testing.assert_array_equal(sorted(sample), [0, 1, 2])

    # Reset and push 25 elements [0, 1, ..., 24] in several batches.
    sampler.reset()
    sampler.push([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    sampler.push([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    sampler.push([20, 21, 22, 23, 24])

    self.assertEqual(sampler.total_count, 25)
    sample = sampler.sample
    self.assertEqual(len(sample), sampler.limit)
    self.assertEqual(len(np.unique(sample)), sampler.limit)
    self.assertGreaterEqual(sample.min(), 0)
    self.assertLessEqual(sample.max(), 24)

  def test_random_sampler_uniformity(self):
    np.random.seed(0)
    # Basic check that sampling is uniform: check that each dataset element
    # appears similarly often over multiple trials.
    dataset_size = 15
    num_trials = 250
    histogram = np.zeros(dataset_size, dtype=int)
    sampler = stats.RandomSampler(3)

    for _ in range(num_trials):
      sampler.reset()
      # Push dataset as several batches so that multiple partition sorts happen.
      sampler.push([0, 1, 2, 3])
      sampler.push([4, 5, 6, 7])
      sampler.push([8, 9, 10, 11])
      sampler.push([12, 13, 14])
      histogram += np.bincount(sampler.sample, minlength=dataset_size)

    np.testing.assert_allclose(histogram, 50, atol=15)


if __name__ == '__main__':
  unittest.main()

