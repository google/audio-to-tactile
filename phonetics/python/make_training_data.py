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

r"""Make training or testing examples for phonetic analysis from labeled data.

This program processes .wav recordings with .phn phone labels using the
CARL+PCEN frontend and writes a ".npz" format file with numpy.savez:
https://numpy.org/doc/stable/reference/generated/numpy.savez.html

Labels are read from .phn files in simple text format, as used in TIMIT. Each
row gives the segment start and end indices and phonetic label, e.g.

  0 1502 sil
  1502 2492 ae
  2492 4000 sil

labels samples [0, 1502) as 'sil' (silence), [1502, 2492) as 'ae', and
[2492, 4000) as 'sil'.
"""

import collections
import glob
import os.path
import threading
from typing import Any, Dict, List, Set

from absl import app
from absl import flags
import numpy as np

from dsp.python import rational_factor_resampler
from dsp.python import wav_io
from frontend.python import carl_frontend
from phonetics.python import phone_util
from phonetics.python.sliding_window import sliding_window

FLAGS = flags.FLAGS

flags.DEFINE_list('examples', None,
                  '[REQUIRED] Comma-delimited list of file globs of .wav '
                  'files to build the dataset from. Files not ending in .wav '
                  'are ignored. It is expected that for each .wav file, '
                  'there exists a .phn phone label file with the same name.')

flags.DEFINE_string('output', None,
                    '[REQUIRED] Output numpy .npz file.')

flags.DEFINE_integer('num_worker_threads', 6,
                     'Number of parallel threads.')

# NOTE: We could in principle also add frames of right context, but we don't do
# this since right context adds latency. A particular goal of Vibro is to
# complement lip reading, and in order to stay perceptually synchronized with
# vision, end-to-end system latency must stay below 50 ms.
#
# Reference:
#   Luca and Mahnan, "Perceptual Limits of Visual-Haptic Simultaneity in Virtual
#   Reality Interactions." 2019 IEEE World Haptics Conference.
#   https://doi.org/10.1109/WHC.2019.8816173
flags.DEFINE_integer('num_frames_left_context', 0,
                     'Number of frames to the left of (older than) the current '
                     'frame in network input. The total number of frames in '
                     'the input is `num_frames_left_context + 1`.')

flags.DEFINE_integer('downsample_factor', 128,
                     'Examples are downsampled by this factor from 16kHz. Must '
                     'be an integer multiple of block_size. E.g. the default '
                     'factor 128 reduces to 125Hz.')

flags.DEFINE_float('min_phone_length_s', 0.05,
                   'Min phone length in seconds. Phonetic segments shorter '
                   'than this are skipped.')

flags.DEFINE_float('phone_trim_left', 0.25,
                   'Fraction of phone segment to trim from left (beginning). '
                   'E.g 0.25 to trim 25%.')

flags.DEFINE_float('phone_trim_right', 0.25,
                   'Fraction of phone segment to trim from right (end).')

# Data augmentation parameters.
flags.DEFINE_integer('num_draws', 5,
                     'Number of "draws" to make from each recording.')

flags.DEFINE_float('max_resample_percent', 1.0,
                   'For each draw, audio is resampled to change pitch and '
                   'compress/dilate time by up to +/-max_resample_percent.')

flags.DEFINE_float('noise_stddev', 1e-4,
                   'Stddev of white noise added to the recording.')

flags.DEFINE_float('min_simulated_distance', 0.2,
                   'For each draw, audio is scaled to simulate recording at '
                   'different distances. This flag is the min distance, as a '
                   'factor relative to the original samples.')

flags.DEFINE_float('max_simulated_distance', 1.0,
                   'Max simulated distance, as a factor relative to the '
                   'original recording.')

# CarlFrontend parameters.
flags.DEFINE_integer('block_size', 64,
                     'Frontend block size parameter.')

flags.DEFINE_float('highest_pole_frequency_hz', 7000.0,
                   'Highest frequency to look at in Hz.')

flags.DEFINE_float('min_pole_frequency_hz', 100.0,
                   'Lower bound on generated pole frequencies.')

flags.DEFINE_float('step_erbs', 0.5,
                   'Channel spacing in equivalent rectangular bandwidth units.')

flags.DEFINE_float('envelope_cutoff_hz', 20.0,
                   'Cutoff frequency in Hz for smoothing energy envelopes.')

flags.DEFINE_float('pcen_time_constant_s', 0.3,
                   'PCEN denominator time constant.')

flags.DEFINE_float('pcen_cross_channel_diffusivity', 100.0,
                   'Diffusivity to smooth PCEN denominator across channels.')

flags.DEFINE_float('pcen_init_value', 1e-7,
                   'PCEN denominator initialization.')

flags.DEFINE_float('pcen_alpha', 0.7,
                   'PCEN denominator exponent.')

flags.DEFINE_float('pcen_beta', 0.2,
                   'PCEN beta (outer) exponent, applied to the ratio.')

flags.DEFINE_float('pcen_gamma', 1e-12,
                   'PCEN denominator offset.')

flags.DEFINE_float('pcen_delta', 0.001,
                   'PCEN zero offset.')

# Class balancing parameters.
flags.DEFINE_list('fg_classes',
                  ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'uw', 'er'],
                  'Class labels that are considered "foreground" classes.')

flags.DEFINE_float('fg_balance_exponent', 0.5,
                   'Foreground class balance exponent.')

flags.DEFINE_float('bg_fraction', 0.125,
                   'Fraction of the dataset to spend on background classes.')

flags.DEFINE_float('bg_balance_exponent', 0.5,
                   'Background class balance exponent.')

AUDIO_SAMPLE_RATE_HZ = 16000

# TIMIT includes many fine-grained phone labels that sound very similar, so it
# is common practice in speech literature to coalesce them. The following
# dictionary defines how we coalesce, which we apply as
#
#   phone = COALESCE_SIMILAR_PHONES.get(phone, phone)
#
# If `phone` is *not* a key in COALESCE_SIMILAR_PHONES, it is unchanged.
COALESCE_SIMILAR_PHONES = {
    'dx': 'd',
    'pcl': 'p',
    'bcl': 'b',
    'kcl': 'k',
    'gcl': 'g',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil',
    }

# Phones to exclude from the dataset. We skip many of the *x vowels, whose
# classification is less clear.
PHONES_TO_EXCLUDE_FROM_DATASET = frozenset(('ax', 'ax-h', 'axr', 'ix', 'ux'))


def get_frontend_params_from_flags() -> Dict[str, Any]:
  return {
      'input_sample_rate_hz': AUDIO_SAMPLE_RATE_HZ,
      'block_size': FLAGS.block_size,
      'highest_pole_frequency_hz': FLAGS.highest_pole_frequency_hz,
      'min_pole_frequency_hz': FLAGS.min_pole_frequency_hz,
      'step_erbs': FLAGS.step_erbs,
      'envelope_cutoff_hz': FLAGS.envelope_cutoff_hz,
      'pcen_time_constant_s': FLAGS.pcen_time_constant_s,
      'pcen_cross_channel_diffusivity': FLAGS.pcen_cross_channel_diffusivity,
      'pcen_init_value': FLAGS.pcen_init_value,
      'pcen_alpha': FLAGS.pcen_alpha,
      'pcen_beta': FLAGS.pcen_beta,
      'pcen_gamma': FLAGS.pcen_gamma,
      'pcen_delta': FLAGS.pcen_delta,
  }


def process_one_wav_file(wav_file: str) -> Dict[str, List[np.ndarray]]:
  """Processes one WAV file to create observed frames.

  Processes one TIMIT WAV file with the frontend, and uses the associated label
  file to group observed frames by phone. Segments shorter than
  FLAGS.min_phone_length_s or with labels in PHONES_TO_EXCLUDE_FROM_DATASET are
  skipped.

  Audio channels are averaged (if there are multiple channels) to reduce to mono
  before processing.

  Args:
    wav_file: String, WAV file path.
  Returns:
    Examples dict with values of shape (num_examples, num_frames, num_channels).
    `examples[phone][i]` is the input for the ith example with label `phone`.
  """
  samples_orig, sample_rate_hz = wav_io.read_wav_file(wav_file,
                                                      dtype=np.float32)
  samples_orig = samples_orig.mean(axis=1)

  phone_times = phone_util.get_phone_times(
      phone_util.get_phone_label_filename(wav_file))
  frontend = carl_frontend.CarlFrontend(**get_frontend_params_from_flags())
  examples = collections.defaultdict(list)
  translation = 0

  for draw_index in range(FLAGS.num_draws):
    samples = np.copy(samples_orig)

    # Resample from sample_rate_hz to AUDIO_SAMPLE_RATE_HZ, perturbed up to
    # +/-max_resample_percent to change pitch and compress/dilate time.
    # TODO(getreuer): For more data augmentation, consider changing pitch and
    # time stretching independently.
    dilation_factor = AUDIO_SAMPLE_RATE_HZ / sample_rate_hz
    if draw_index > 0:
      max_log_dilation = np.log(1.0 + FLAGS.max_resample_percent / 100.0)
      dilation_factor *= np.exp(
          np.random.uniform(-max_log_dilation, max_log_dilation))

    if abs(dilation_factor - 1.0) >= 1e-4:
      resampler = rational_factor_resampler.Resampler(
          1.0, dilation_factor, max_denominator=2000)
      samples = resampler.process_samples(samples)

    if draw_index > 0:
      # Prepend a random fraction of a block of silence. This randomizes the
      # input phase with respect to the frontend's decimation by block_size.
      translation = np.random.randint(FLAGS.block_size)
      samples = np.append(np.zeros(translation), samples)
      # Add white Gaussian noise.
      samples = np.random.normal(
          samples, FLAGS.noise_stddev).astype(np.float32)
      # Scale the samples to simulate the recording at a different distance.
      samples /= np.exp(np.random.uniform(
          np.log(FLAGS.min_simulated_distance),
          np.log(FLAGS.max_simulated_distance)))

    observed = phone_util.run_frontend(frontend, samples)

    for start, end, phone in phone_times:
      start = int(round(dilation_factor * start)) + translation
      end = min(int(round(dilation_factor * end)), len(samples)) + translation
      phone_length_s = float(end - start) / sample_rate_hz

      # Skip short (quickly-spoken) phone segments. They are likely influenced
      # by preceding/following phones, making classification is less clear.
      if phone_length_s < FLAGS.min_phone_length_s:
        continue  # Skip short phone.

      phone = COALESCE_SIMILAR_PHONES.get(phone, phone)

      if phone in PHONES_TO_EXCLUDE_FROM_DATASET:
        continue

      # There may be confusing transitions (or possible labeling inaccuracy)
      # near the segment endpoints, so trim a fraction from each end.
      length = end - start
      start += int(round(length * FLAGS.phone_trim_left))
      end -= int(round(length * FLAGS.phone_trim_right))

      # Convert sample indices from audio sample rate to frame rate.
      start //= FLAGS.block_size
      end //= FLAGS.block_size

      left_context = FLAGS.num_frames_left_context
      # Extract a window every `hop` frames and append to examples.
      examples[phone].append(sliding_window(
          observed[max(0, start - left_context):end],
          window_size=left_context + 1,
          hop=FLAGS.downsample_factor // frontend.block_size))

  return examples


def process_wav_files(wav_files: Set[str]) -> Dict[str, np.ndarray]:
  """Processes a list of WAV files to create observed frames."""
  all_examples = collections.defaultdict(list)
  lock = threading.Lock()
  count = [0]
  print('Making dataset:')

  def _process(wav_file):
    examples = process_one_wav_file(wav_file)

    lock.acquire()
    for phone in examples:
      all_examples[phone].extend(examples[phone])
    count[0] += 1
    print('\r  %4d / %4d' % (count[0], len(wav_files)), end='', flush=True)
    lock.release()

  phone_util.run_in_parallel(wav_files, FLAGS.num_worker_threads, _process)
  print('\n')
  return {k: np.concatenate(v) for k, v in all_examples.items()}


def balance_examples(dataset: phone_util.Dataset) -> None:
  """Balance examples for each phone."""
  original_counts = dataset.example_counts
  dataset.subsample(phone_util.balance_weights(
      original_counts,
      fg_classes=FLAGS.fg_classes,
      fg_balance_exponent=FLAGS.fg_balance_exponent,
      bg_fraction=FLAGS.bg_fraction,
      bg_balance_exponent=FLAGS.bg_balance_exponent))

  subsampled_counts = dataset.example_counts
  print('Example counts for each phone:')
  for phone in original_counts:
    print('%-4s: %7d -> %7s' % (phone, original_counts[phone],
                                subsampled_counts.get(phone, 'removed')))


def main(argv) -> int:
  if len(argv) > 1:
    print(f'WARNING: Non-flag arguments: {argv}')
  assert FLAGS.downsample_factor % FLAGS.block_size == 0

  wav_files = set()
  for glob_pattern in FLAGS.examples:
    glob_pattern = os.path.expanduser(glob_pattern)
    wav_files.update(wav_file for wav_file in glob.glob(glob_pattern)
                     if wav_file.lower().endswith('.wav'))

  if not wav_files:
    print(f'Error: No .wav files found matching {FLAGS.examples}')
    return 1

  for wav_file in wav_files:
    phn_file = phone_util.get_phone_label_filename(wav_file)
    if not os.path.isfile(phn_file):
      print(f'Error: .phn file not found: {phn_file}')
      return 1

  frontend = carl_frontend.CarlFrontend(**get_frontend_params_from_flags())
  examples = process_wav_files(wav_files)

  metadata = {
      'frontend_params': get_frontend_params_from_flags(),
      'num_channels': frontend.num_channels,
      'num_frames_left_context': FLAGS.num_frames_left_context,
      'flags': phone_util.get_main_module_flags_dict(),
  }
  dataset = phone_util.Dataset(examples, metadata)
  balance_examples(dataset)

  print(f'\nWriting dataset to {FLAGS.output}')
  dataset.write_npz(FLAGS.output)
  return 0


if __name__ == '__main__':
  flags.mark_flag_as_required('examples')
  flags.mark_flag_as_required('output')
  app.run(main)

