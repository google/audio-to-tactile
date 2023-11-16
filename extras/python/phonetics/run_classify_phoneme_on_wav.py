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

"""Program that runs classify_phoneme inference on a WAV file.

Given a WAV file of English speech, this program runs it through the classifier,
and it produces plots of the classification scores.
"""

import os.path
from typing import Any, Collection, Dict, Tuple

from absl import app
from absl import flags
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from extras.python import dsp
from extras.python import frontend
from extras.python import plot
from extras.python.phonetics import classify_phoneme
from extras.python.phonetics import phone_util
from extras.python.phonetics.sliding_window import sliding_window

FLAGS = flags.FLAGS

flags.DEFINE_string('input', None,
                    '[REQUIRED] Input WAV file.')

flags.DEFINE_string('output', None,
                    'If specified, save plot image to this filename. Otherwise '
                    'show in a figure window [assumes matplotlib uses "TkAgg" '
                    'or another backend capable of user interface].')

Figure = matplotlib.figure.Figure
CLASSIFIER_INPUT_HZ = 16000


def score_o_gram(ax: matplotlib.axes.Axes,
                 scores: Dict[str, Collection[float]],
                 t_limits: Tuple[float, float]) -> None:
  """Plots classification scores as a "score-o-gram" image.

  Each row plots scores for one label, and the horizontal dimension is time.

  Args:
    ax: Matplotlib axes.
    scores: Dict of scores timeseries.
    t_limits: (t_min, t_max) 2-tuple time range represented by `scores`.
  """
  keys = list(scores.keys())
  values = np.vstack([scores[k] for k in keys])
  extent = tuple(t_limits) + (len(keys) - 0.5, -0.5)
  ax.imshow(np.sqrt(values), interpolation='None',
            aspect='auto', cmap='density', extent=extent, vmin=0.0, vmax=1.0)
  ax.set(yticks=np.arange(len(keys)), yticklabels=keys)
  ax.set_xlabel('Time (s)')


def plot_output(frames: np.ndarray,
                frame_rate: float,
                timeseries: Dict[str, Any],
                title: str) -> Tuple[Figure, Figure]:
  """Plots frontend frames and score outputs.

  Args:
    frames: 2D numpy array, where each row is a frontend frame.
    frame_rate: Float, number of frames per second.
    timeseries: Dict of classification scores over time. For instance
      `timeseries['vad']` should be a list of VAD scores for every frame.
    title: String, title for the figure.
  Returns:
    Two matplotlib figure objects.
  """
  t_limits = (0.0, (len(frames) - 1) / frame_rate)

  # A combined figure plotting the frontend frames, VAD and voicing scores, and
  # score-o-gram for the manner classification.
  fig_combined = plt.figure(figsize=(10, 6))
  ax1, ax2, ax3 = fig_combined.subplots(3, 1, sharex=True)
  extent = t_limits + (frames.shape[1], 0)
  ax1.imshow(frames.transpose(), aspect='auto', interpolation='None',
             cmap='density', extent=extent)
  ax1.set_xlim(t_limits)
  ax1.set_title(title)
  ax1.set_ylabel('Input')

  t = (np.arange(len(timeseries['vad']))
       + classify_phoneme.NUM_FRAMES - 1) / frame_rate
  ax2.plot(t, timeseries['vad'], label='VAD')
  ax2.plot(t, timeseries['voiced'], label='Voiced')
  ax2.legend()

  score_o_gram(ax3, timeseries['manner'], (t[0], t[-1]))
  ax3.set_xlim(t_limits)
  fig_combined.set_tight_layout(True)

  # Figure plotting the score-o-gram for the phoneme classification.
  fig_phoneme = plt.figure(figsize=(10, 6))
  ax4 = fig_phoneme.subplots(1, 1)
  score_o_gram(ax4, timeseries['phoneme'], (t[0], t[-1]))
  ax4.set_xlim(t_limits)
  fig_phoneme.set_tight_layout(True)

  return fig_combined, fig_phoneme


def append_to_dict(timeseries: Any, scores: Dict[str, Any]) -> None:
  for k, v in scores.items():
    if k not in timeseries:
      timeseries[k] = {} if isinstance(v, dict) else []
    if isinstance(v, dict):
      append_to_dict(timeseries[k], v)
    else:
      timeseries[k].append(v)  # pytype: disable=attribute-error


def main(_):
  # Read WAV file.
  samples, sample_rate_hz = dsp.read_wav_file(FLAGS.input, dtype=np.float32)  # pytype: disable=wrong-arg-types  # numpy-scalars
  samples = samples.mean(axis=1)

  # Run frontend to get CARL frames. The classifier expects input sample rate
  # CLASSIFIER_INPUT_HZ, block_size=128, pcen_cross_channel_diffusivity=60, and
  # otherwise the default frontend settings.
  carl = frontend.CarlFrontend(input_sample_rate_hz=CLASSIFIER_INPUT_HZ,
                               block_size=128,
                               pcen_cross_channel_diffusivity=60.0)
  if sample_rate_hz != CLASSIFIER_INPUT_HZ:
    resampler = dsp.Resampler(sample_rate_hz, CLASSIFIER_INPUT_HZ)
    samples = resampler.process_samples(samples)
  frames = phone_util.run_frontend(carl, samples)
  # The frame rate is 125Hz (hop size of 8ms).
  frame_rate = CLASSIFIER_INPUT_HZ / carl.block_size

  timeseries = {}
  for window in sliding_window(frames, classify_phoneme.NUM_FRAMES):
    # Run classifier inference on the current window.
    scores = classify_phoneme.classify_phoneme_scores(window)
    append_to_dict(timeseries, scores)

  fig_combined, fig_phoneme = plot_output(frames, frame_rate, timeseries,
                                          os.path.basename(FLAGS.input))

  if FLAGS.output:  # Save plot as an image file.
    stem, ext = os.path.splitext(FLAGS.output)
    plot.save_figure(stem + '-combined' + ext, fig_combined)
    plot.save_figure(stem + '-phoneme' + ext, fig_phoneme)
  else:  # Show plot interactively.
    plt.show()
  return 0


if __name__ == '__main__':
  flags.mark_flag_as_required('input')
  app.run(main)
