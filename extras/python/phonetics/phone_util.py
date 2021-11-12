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

"""Utilities for phone embedding network."""

import csv
import json
import os.path
import queue
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, TypeVar, Tuple

from absl import flags
import numpy as np

from extras.python import frontend

FLAGS = flags.FLAGS

KT = TypeVar('KT')
VT_in = TypeVar('VT_in')  # pylint:disable=invalid-name
VT_out = TypeVar('VT_out')  # pylint:disable=invalid-name


def _map_dict_values(fun: Callable[[VT_in], VT_out],
                     d: Dict[KT, VT_in]) -> Dict[KT, VT_out]:
  return {k: fun(v) for k, v in d.items()}


def get_main_module_flags_dict() -> Dict[str, Any]:
  """Gets dict of flags that were defined in the main module."""
  return{f.name: f.value for f in FLAGS.flags_by_module_dict()[sys.argv[0]]}


def get_phone_label_filename(wav_file: str) -> str:
  """Gets the phone label filename associated with `wav_file`."""
  return os.path.splitext(wav_file)[0] + '.phn'


def get_phone_times(phn_file: Any) -> List[Tuple[int, int, str]]:
  """Gets endpoint times for each phone in a recording.

  Reads phone endpoint times from .phn file. The .phn file has a simple text
  format as used in TIMIT. Each row gives start and end sample indices and label
  for one phone, '<start> <end> <label>'.

  Args:
    phn_file: String or file-like object.

  Returns:
    List of 3-tuples (start, end, label) where `start` and `end` are sample
    indices where the phone is active, and `label` is a phone string.
  """

  def _read(f):
    """Read .phn CSV data from file object `f`."""
    try:
      reader = csv.reader(f, delimiter=' ', quotechar='"')
      results = []
      for row in reader:
        if len(row) != 3:
          continue
        start, end, label = row
        results.append((int(start), int(end), label))
      return results
    except (IOError, UnicodeDecodeError, TypeError) as e:
      # If reading fails, reraise with the filename for more context.
      name = getattr(f, 'name', '(no name)')
      raise IOError(f'Error reading .phn file {name}: {e}')

  if isinstance(phn_file, str):
    with open(phn_file, 'rt') as f:
      return _read(f)
  else:
    return _read(phn_file)


def run_frontend(carl: frontend.CarlFrontend,
                 audio_samples: np.ndarray) -> np.ndarray:
  """Reset and run CarlFrontend on audio samples.

  Convenience function to reset and run a CarlFrontend on some audio samples,
  zero-padding as necessary to get a whole number of blocks.

  Args:
    carl: CarlFrontend.
    audio_samples: 1D array of audio samples of any length. [It is zero padded
      if necessary to make a whole number of blocks.]
  Returns:
    2D array of shape [num_frames, frontend.block_size] of frames.
  """
  carl.reset()
  audio_samples = np.asarray(audio_samples, dtype=np.float32)
  # Zero pad to a whole number of blocks.
  padding = (-len(audio_samples)) % carl.block_size
  audio_samples = np.append(audio_samples, np.zeros(padding, np.float32))
  return carl.process_samples(audio_samples)


ItemT = TypeVar('ItemT')


def run_in_parallel(items: Iterable[ItemT],
                    num_threads: int,
                    fun: Callable[[ItemT], Any]) -> None:
  """Run tasks concurrently on multiple threads.

  This function conceptually runs the for loop

    for item in items:
      fun(item)

  with multiple threads. Note that `fun` must release the GIL to actually get
  performance benefits [https://wiki.python.org/moin/GlobalInterpreterLock].

  Args:
    items: Iterable.
    num_threads: Integer, number of worker threads.
    fun: Function taking one item as its input.
  """
  stop_worker = object()

  # This implementation follows the example in the Queue documentation:
  # https://docs.python.org/3/library/queue.html#queue.Queue.join
  def _worker():
    """One worker thread."""
    while True:
      item = q.get()
      if item is stop_worker:
        break
      fun(item)
      q.task_done()

  q = queue.Queue()
  threads = []
  for _ in range(num_threads):
    t = threading.Thread(target=_worker)
    t.start()
    threads.append(t)

  for item in items:
    q.put(item)

  q.join()  # Block until all tasks are done.

  for _ in range(num_threads):  # Stop workers.
    q.put(stop_worker)
  for t in threads:
    t.join()


def balance_weights(example_counts: Dict[str, int],
                    fg_classes: Iterable[str],
                    fg_balance_exponent: float,
                    bg_fraction: float,
                    bg_balance_exponent: float) -> Dict[str, float]:
  """Computes weights to partially normalize for class imbalance.

  Compute balancing weights from example counts. The weights partially normalize
  for class imbalance, keeping some bias in favor of more frequent classes, like

    L. S. Yaeger, B. J. Webb, R. F. Lyon. "Combining neural networks and
    context-driven search for online, printed handwriting recognition in the
    Newton." AI Magazine 19.1 (1998): 73-73.
    http://dicklyon.com/tech/Mondello/AIMag-Lyon.pdf

  Args:
    example_counts: Dict, where `examples_counts[label]` is the number of
      available examples for class `label`.
    fg_classes: List of strings, class labels that are in the "foreground".
      Classes not in this list are "background".
    fg_balance_exponent: Float, an exponent between 0.0 and 1.0 for balancing
      foreground classes. A value of 1.0 implies full normalization.
    bg_fraction: Float between 0.0 and 1.0, the fraction of the balanced dataset
      to devote to background classes.
    bg_balance_exponent: Float, balancing exponent for background classes.
  Returns:
    weights dict, where `weights[label]` is a value between 0.0 and 1.0, the
    fraction of examples of class `label` that should be retained.
  """
  # Split phones to "foreground" classes of interest and "background" classes.
  fg_classes = list(fg_classes)
  bg_classes = list(set(example_counts.keys()) - set(fg_classes))
  fg_counts = np.array([example_counts[k] for k in fg_classes])
  bg_counts = np.array([example_counts[k] for k in bg_classes])

  fg_weights = np.maximum(0, fg_counts)**-fg_balance_exponent
  bg_weights = np.maximum(0, bg_counts)**-bg_balance_exponent

  bg_total = fg_weights.dot(fg_counts) * (bg_fraction / (1.0 - bg_fraction))
  # Normalize bg_weights such that background examples add up to bg_total.
  bg_weights *= bg_total / bg_weights.dot(bg_counts)

  weights = np.concatenate((fg_weights, bg_weights))
  weights /= weights.max()  # Rescale max weight to 1.0.
  return dict(zip(fg_classes + bg_classes, weights))


class Dataset:
  """Dataset for phone embedding model."""

  def __init__(self,
               examples: Dict[str, np.ndarray],
               metadata: Dict[str, Any]):
    """Constructor.

    Args:
      examples: Dict, where `examples[phone]` is a 3D array of examples with
        label `phone` of shape (num_examples, num_frames, num_channels).
      metadata: Dict.
    """
    self.examples = examples
    self.metadata = metadata
    self._validate()
    self._drop_empty_classes()

  def _validate(self) -> None:
    """Validates example array shapes."""
    for k, v in self.examples.items():
      if not isinstance(v, np.ndarray):
        raise TypeError(f'"{k}": expected numpy array, {type(v)} found')
      elif v.shape[1:] != (self.num_frames, self.num_channels):
        raise ValueError(f'"{k}": shape {v.shape} mismatches expected shape '
                         f'Nx{self.num_frames}x{self.num_channels}')
      elif not np.all(np.isfinite(v)):
        raise ValueError(f'"{k}": array has nonfinite value')

  def _drop_empty_classes(self) -> None:
    """Drop empty classes from `self.examples`."""
    self.examples = {k: v for k, v in self.examples.items() if len(v)}

  @property
  def num_frames(self) -> int:
    return int(self.metadata['num_frames_left_context'] + 1)

  @property
  def num_channels(self) -> int:
    return int(self.metadata['num_channels'])

  @property
  def example_counts(self) -> Dict[str, int]:
    return _map_dict_values(len, self.examples)  # pytype: disable=wrong-arg-types

  def get_xy_arrays(self,
                    phones: Iterable[str],
                    shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Gets dataset as a pair of arrays x and y, as in sklearn or tf.estimator.

    This function converts `example_dict` to two numpy arrays x and y, of
    observations and labels, as used in sklearn and tf.estimator. The x array
    is created by concatenating `example_dict[phones[i]]` along the first axis
    while the y is a 1D array of the same length of corresponding label indices,

      x = concatenate([example_dict[phones[0]], example_dict[phones[1]], ...])
      y =             [0, 0, 0, 0, ...       0, 1, 1, 1, 1, ...       1, ...]

    Args:
      phones: List of phone labels. This determines which phones are included
        in the output and the enumeration of label indices in y.
      shuffle: Bool. If true, the output is shuffled.
    Returns:
      (x, y) 2-tuple of numpy arrays.
    """
    x, y = [], []
    for i, phone in enumerate(phones):
      if phone in self.examples:
        x.append(self.examples[phone].astype(np.float32))
        y.append(np.full(len(self.examples[phone]), i, dtype=np.int32))

    x = np.concatenate(x)
    y = np.concatenate(y)

    if shuffle:
      i = np.random.permutation(len(x))
      x, y = x[i], y[i]
    return x, y

  def subsample(self, fraction: Dict[str, float]) -> None:
    """Subsamples examples according to `fraction`.

    This function randomly subsamples `examples[phone]` according to
    `fraction[phone]`. For instance if weights = {'ae': 0.6, 'sil': 0.1}, then
    the subsampling retains 60% of 'ae' examples and 10% of 'sil' examples. This
    is useful to compensate for class imbalance.

    If `examples[phone]` becomes empty after subsampling, it is deleted from the
    dict. If a particular phone is not in `fraction`, zero is assumed and
    `examples[phone]` is deleted.

    Args:
      fraction: Dict. `fraction[phone]` is a value between 0.0 and 1.0
        specifying the fraction of examples to keep for label `phone`.
    Raises:
      ValueError: If fraction is invalid.
    """
    for phone in self.examples:
      fraction_phone = fraction.get(phone, 0.0)
      if not 0.0 <= fraction_phone <= 1.0:
        raise ValueError(f'fraction["{phone}"] = {fraction_phone} is not '
                         'between 0.0 and 1.0')

      count = len(self.examples[phone])
      subsampled_count = int(round(fraction_phone * count))
      i = np.random.permutation(count)[:subsampled_count]
      self.examples[phone] = self.examples[phone][i]

    self._drop_empty_classes()

  def split(self, fraction: float) -> Tuple['Dataset', 'Dataset']:
    """Split and return two Datasets.

    Args:
      fraction: Float, a fraction between 0.0 and 1.0.
    Returns:
      2-tuple of two Datasets. The first has a random sampling of `fraction` of
      the examples for each class, and the second has the other examples.
    """
    if not 0.0 <= fraction <= 1.0:
      raise ValueError(f'fraction = {fraction} is not between 0.0 and 1.0')

    examples_a = {}
    examples_b = {}
    for phone in self.examples:
      count = len(self.examples[phone])
      split_count = int(round(fraction * count))
      i = np.random.permutation(count)
      examples_a[phone] = self.examples[phone][i[:split_count]]
      examples_b[phone] = self.examples[phone][i[split_count:]]

    return (Dataset(examples_a, self.metadata),
            Dataset(examples_b, self.metadata))

  def write_npz(self, npz_file: Any) -> None:
    """Writes Dataset to .npz file.

    Args:
      npz_file: String or file-like object.
    """
    self._validate()
    self._drop_empty_classes()
    contents = _map_dict_values(lambda v: v.astype(np.float32), self.examples)
    contents['dataset_metadata'] = json.dumps(self.metadata).encode('utf8')
    np.savez(npz_file, **contents)


def read_dataset_npz(npz_file: Any) -> Dataset:
  """Reads Dataset from .npz file.

  Args:
    npz_file: String or writeable file-like object.
  Returns:
    Dataset.
  """
  contents = np.load(npz_file)
  if 'dataset_metadata' not in contents.files:
    raise ValueError('dataset_metadata missing from NPZ file')
  metadata = json.loads(contents['dataset_metadata'].item().decode('utf8'))
  examples = {
      k: v.astype(np.float32)
      for k, v in contents.items()
      if k != 'dataset_metadata'
  }
  return Dataset(examples, metadata)
