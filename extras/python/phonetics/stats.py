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

"""Statistics utilities."""

import collections
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np
import scipy.stats
import sklearn.manifold


class Confusion:
  """Confusion matrix."""

  def __init__(self,
               matrix: np.ndarray,
               class_names: Optional[Iterable[str]] = None) -> None:
    """Constructor.

    Args:
      matrix: 2D N-by-N integer-valued array, where the (i,j)th entry is the
        number of trials in which the ith stimulus was presented and resulted in
        the jth response.
      class_names: List of N class names.
    Raises:
      ValueError: If matrix shape mismatches class_names.
    """
    self.matrix = np.asarray(matrix)
    if class_names is None:
      class_names = [str(i) for i in range(len(self.matrix))]
    self.class_names = tuple(class_names)
    if self.matrix.shape != (len(self.class_names), len(self.class_names)):
      raise ValueError('Matrix shape mismatches class_names')

  @property
  def normalized_matrix(self) -> np.ndarray:
    """Confusion matrix normalized so that each row sums to one."""
    cm = self.matrix.astype(float)
    cm /= np.maximum(1e-12, cm.sum(axis=1, keepdims=True))
    return cm

  @property
  def stimulus_bits(self) -> float:
    """Entropy of the stimulus (input) distribution in bits."""
    hist = self.matrix.sum(axis=1).astype(float)
    return scipy.stats.entropy(hist / hist.sum(), base=2)  # pytype: disable=bad-return-type  # numpy-scalars

  @property
  def response_bits(self) -> float:
    """Entropy of the response (output) distribution in bits."""
    hist = self.matrix.sum(axis=0).astype(float)
    return scipy.stats.entropy(hist / hist.sum(), base=2)  # pytype: disable=bad-return-type  # numpy-scalars

  @property
  def transfer_bits(self) -> float:
    """Estimated information transfer in bits.

    Let X denote the input variable and Y the output variable, and define
    p[i] = P(X = i) and p[i,j] = P(X = i, Y = j). The entropy of X is

      H(X) = -sum_i p[i] log p[i],

    and entropy of X and Y considered jointly is

      H(XY) = -sum_i,j p[i,j] log p[i,j].

    Then, the information transfer from X to Y is

      T(X;Y) = H(X) + H(Y) - H(XY)
             = -sum_i,j p[i,j] log(p[i] p[j] / p[i,j]).

    Reference:
      page 348 of Miller and Nicely, "An analysis of perceptual confusions among
      some English consonants," The Journal of the Acoustical Society of America
      27.2 (1955): 338-352.

    Returns:
      Float, information transfer in units of bits.
    """
    return self.stimulus_bits + self.response_bits - scipy.stats.entropy(  # pytype: disable=bad-return-type  # numpy-scalars
        self.matrix.flatten(), base=2)

  def group_by(self, group_dict: Mapping[str, Any]) -> 'Confusion':
    """Groups classes by `group_dict`, returning a reduced confusion matrix.

    Args:
      group_dict: Dict or defaultdict whose keys are names in`class_names` and
        values are group names. Classes that fail to look up are dropped.
    Returns:
      A new Confusion matrix.
    """
    groups = collections.defaultdict(list)
    for i, class_name in enumerate(self.class_names):
      try:
        # Accumulate the list of class indices belonging to each group.
        groups[group_dict[class_name]].append(i)
      except KeyError:
        pass  # Ignore classes that group_dict can't look up.
    # Convert to a list of arrays, ordered by key.
    group_keys = sorted(groups.keys())
    groups = tuple(np.array(groups[key]) for key in group_keys)

    # Sum confusion matrix over the groups.
    grouped_matrix = np.empty((len(groups), len(groups)), self.matrix.dtype)
    for i, group_i in enumerate(groups):
      for j, group_j in enumerate(groups):
        grouped_matrix[i, j] = self.matrix[group_i, :][:, group_j].sum()

    return Confusion(grouped_matrix, tuple(map(str, group_keys)))

  def save_csv(self, filename) -> None:
    """Saves confusion matrix as CSV file with class_names in the header line.

    Args:
      filename: String or file object.
    """
    np.savetxt(filename, self.matrix, fmt='%d', delimiter=',',
               header=','.join(self.class_names))

  def mds_embedding(self, num_dims: int = 2) -> np.ndarray:
    """Performs metric MDS to embed the confusion matrix in lower dimension.

    This function performs metric MDS on a confusion matrix
    [https://en.wikipedia.org/wiki/Multidimensional_scaling]. An array is
    returned of coordinates for each class, where more often confused classes
    are closer together. This is useful to visualize confusions.

    Args:
      num_dims: Integer.
    Returns:
      2D array of shape (len(class_names), num_dims) of the embedded
      coordinates for each class.
    """
    cm = self.normalized_matrix
    cm = cm.clip(1e-4, 1.0)  # Avoid infinity in entropy calculations.

    dissimilarity_matrix = np.empty_like(cm)
    for i, pmf_i in enumerate(cm):
      for j, pmf_j in enumerate(cm):
        # Compute Jensen–Shannon divergence between row i vs. row j of the
        # confusion matrix, considering each row as a probability mass function.
        mix = 0.5 * (pmf_i + pmf_j)
        dissimilarity_matrix[i, j] = 0.5 * (scipy.stats.entropy(pmf_i, mix)
                                            + scipy.stats.entropy(pmf_j, mix))

    # MDS does `n_init` randomly-initialized runs of an "SMACOF" algorithm. For
    # more reliable results, set n_init=8 instead of the default 4.
    mds = sklearn.manifold.MDS(num_dims, dissimilarity='precomputed', n_init=8)
    return mds.fit_transform(dissimilarity_matrix)


class BinaryClassifierStats:
  """Use histograms to approximate a binary classifier's ROC, AUC, and d'.

  This class computes ROC, AUC, and d' approximately, using a random sample that
  takes a fixed amount of memory regardless of the number of examples.

  The intended use is:
  1. Create a BinaryClassifierStats instance.
  2. Evaluate your data in batches, calling `Accum()` once per batch.
  3. Get the final stats from the `.roc_curve`, `.auc`, `.d_prime` properties.
  """

  def __init__(self, sample_size: int = 5000) -> None:
    """Constructor.

    Args:
      sample_size: Positive integer, max size of the random sample.
    """
    self._sampler = RandomSampler(sample_size)

  def reset(self) -> None:
    """Resets to initial zero state."""
    self._sampler.reset()

  def accum(self, labels: Iterable[int], scores: Iterable[float]) -> None:
    """Accumulates classifier statistics.

    Args:
      labels: 1D array of binary label values.
      scores: 1D array of scores. Must be finite and have same length as labels.

    Raises:
      ValueError: if args are invalid.
    """
    labels = (np.asarray(labels) > 0)
    scores = np.asarray(scores, dtype=np.float32)

    if len(labels) != len(scores):
      raise ValueError('labels and scores must have same length')
    elif not np.all(np.isfinite(scores)):
      raise ValueError('scores must be finite')

    self._sampler.push(np.column_stack((labels, scores)))

  @property
  def roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets approximate receiver operating characteristic (ROC).

    Returns:
      (true_positive_rate, false_positive_rate, thresholds) 3-tuple, three
      arrays of the same length. For the ith threshold, the classification
      decision `score >= thresholds[i]` has true positive rate of approximately
      true_positive_rate[i] and false positive rate false_positive_rate[i].
    """
    sample = self._sampler.sample
    # Examples with the same score value (e.g. due to saturation) can create
    # misleading results. One mitigation is to randomly permute the examples,
    # which is conceptually like adding a small amount of noise to the scores.
    sample = np.random.permutation(sample)
    labels = sample[:, 0]
    scores = sample[:, 1]

    # Sort by decreasing score.
    i = np.argsort(scores)[::-1]
    labels = labels[i]
    thresholds = scores[i]

    false_positive_count = np.cumsum(1.0 - labels)
    true_positive_count = np.cumsum(labels)

    false_positive_count = np.append(0.0, false_positive_count)
    true_positive_count = np.append(0.0, true_positive_count)
    thresholds = np.append(thresholds[0] + 1.0, thresholds)

    false_positive_rate = false_positive_count / false_positive_count[-1]
    true_positive_rate = true_positive_count / true_positive_count[-1]

    return false_positive_rate, true_positive_rate, thresholds

  @property
  def auc(self) -> float:
    """Gets approximate area under the ROC curve (ROC AUC)."""
    false_positive_rate, true_positive_rate, _ = self.roc_curve
    return np.trapz(true_positive_rate, false_positive_rate)

  @property
  def d_prime(self) -> float:
    """Gets the approximate equivalent d' sensitivity index.

    Consider the idealized binary classification problem where negative examples
    are normally distributed with mean mu_- and positive examples are normally
    distributed with mean mu_+, and both classes with the same stddev sigma.
    Then the d' (or "d-prime") sensitivity index is defined as

      d' := (mu_+ - mu_-) / sigma.

    In other words, d' is the separation in units of sigmas. In this idealized
    case, d' is related to AUC by

      d' = sqrt(2) * Z(AUC),

    where Z(x) is the inverse cumulative distribution function for the standard
    normal. We use this latter formula to compute the "equivalent d'" for
    (possibly non-normal) distributions generally.

    Returns:
      The equivalent d', computed as sqrt(2) * Z(AUC).
    """
    return np.sqrt(2) * scipy.stats.norm.ppf(self.auc)


class MulticlassClassifierStats:
  """Stats for multiclass classifier.

  We view classification between N classes as N one-vs-all binary classifiers,
  and use BinaryClassifierStats to evaluate each such binary classifier.
  Additionally, we accumulate the N-by-N confusion matrix from considering the
  largest scores as label predictions.
  """

  def __init__(self, num_classes: int, sample_size: int = 5000) -> None:
    """Constructor.

    Args:
      num_classes: Number of classes.
      sample_size: Positive int, max size of the random sample for each class.
    """
    self._binary_stats = [BinaryClassifierStats(sample_size)
                          for _ in range(num_classes)]
    self._confusion_matrix = np.zeros((num_classes, num_classes), np.int64)

  @property
  def num_classes(self) -> int:
    """Number of classes."""
    return len(self._binary_stats)

  def reset(self) -> None:
    """Resets to initial zero state."""
    for binary_stats in self._binary_stats:
      binary_stats.reset()
    self._confusion_matrix[:] = 0

  def accum(self, labels: Iterable[int], scores: Iterable[float]) -> None:
    """Accumulates classifier statistics.

    Args:
      labels: 1D array of binary label values.
      scores: 2D array of scores of shape [len(labels), num_classes]. Must be
        finite.

    Raises:
      ValueError: if args are invalid.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores)

    expected_shape = (len(labels), self.num_classes)
    if scores.shape != expected_shape:
      raise ValueError('Expected scores with shape %s, got: %s'
                       % (expected_shape, scores.shape))

    for k in range(self.num_classes):
      self._binary_stats[k].accum(labels == k, scores[:, k])

    predicted_labels = scores.argmax(axis=1)
    n = self.num_classes
    self._confusion_matrix += np.bincount(
        n * labels + predicted_labels, minlength=n**2).reshape(n, n)

  @property
  def d_prime(self) -> np.ndarray:
    """Return d' values of the one-vs-all binary classifiers."""
    return np.array([binary_stats.d_prime
                     for binary_stats in self._binary_stats])

  @property
  def confusion_matrix(self) -> np.ndarray:
    """Confusion matrix from considering largest scores as label predictions.

    Returns:
      numpy array of shape (num_classes, num_classes), where rows are true
      labels and columns are predicted labels.
    """
    return self._confusion_matrix


class RandomSampler:
  """Class that randomly samples without replacement from a large dataset.

  This class extracts a uniformly random sample from a dataset without
  replacement, where the dataset does not need to fit in memory.

  Example with scalar elements:
    sampler = RandomSampler(K)
    sampler.push([0, 1, 2])
    sampler.push([3, 4, 5, 6, 7])
    sampler.push([8, 9])
    sample = sampler.sample  # Get K random elements in {0, ..., 9}.

  More generally, the dataset elements can be multidimensional arrays of any
  numpy dtype. Calling push with an array of shape [a0, a1, ..., aN] is
  interpreted as a batch of a0 elements, each having shape [a1, ..., aN].

  Algorithm: We associate a random key with each element and find the K lowest
  keys. push() appends elements to a buffer. When the buffer exceeds size 2 K,
  an O(K) partial sort is done to keep the elements with the K lowest keys.
  Supposing a dataset of N total elements is pushed in small batches, the number
  of partial sorts is about N/K, so overall cost is linear O(K N / K) = O(N).
  """

  def __init__(self, limit: int) -> None:
    """Constructor.

    Args:
      limit: Integer, the number of elements to sample.
    """
    self._limit = int(limit)
    self.reset()

  @property
  def limit(self) -> int:
    """Number of elements to sample."""
    return self._limit

  @property
  def total_count(self) -> int:
    """Total number of elements in the dataset seen so far."""
    return self._total_count

  @property
  def sample(self) -> np.ndarray:
    """The sample as array of shape [min(limit, total_count), a1, ..., aN]."""
    if self._sample_keys is None:
      return np.array([])
    elif self._sample_keys.size > self.limit:
      self._discard_excess_elements()
    return self._sample_elements

  def reset(self) -> None:
    """Resets to empty initial state."""
    self._total_count = 0
    self._sample_keys = None
    self._sample_elements = None

  def push(self, elements: Iterable[Any]) -> None:
    """Pushes a batch of elements.

    Args:
      elements: Array of shape [a0, a1, ... aN], representing a batch of a0
        elements, each having shape [a1, ..., aN].
    """
    elements = np.asarray(elements)
    num_elements = len(elements)
    if not num_elements:
      return

    # Assign a random key to each element.
    keys = np.random.rand(num_elements)
    self._total_count += num_elements

    if self._sample_keys is None:
      self._sample_keys = keys
      self._sample_elements = np.copy(elements)
    else:
      self._sample_keys = np.append(self._sample_keys, keys)
      self._sample_elements = np.concatenate((self._sample_elements, elements))

    if self._sample_keys.size >= 2 * self.limit:
      self._discard_excess_elements()

  def _discard_excess_elements(self) -> None:
    """Discard some elements when we have more than the limit."""
    assert self._sample_keys is not None and self._sample_keys.size > self.limit
    # Partial indirect sort to find the lowest `limit` keys.
    i = np.argpartition(self._sample_keys, self.limit)[:self.limit]
    # Keep only those keys.
    self._sample_keys = self._sample_keys[i]
    self._sample_elements = self._sample_elements[i]
