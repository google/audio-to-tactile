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


Program that runs phone_model inference on a WAV file.

This program runs either the vowel net inference implementation in embed_vowel.c
or runs a model from a .pkl file. A plot figure is written of the resulting
posterior target probabilities.

Examples:
# Running with embed_vowel.c.
run_phone_model_on_wav --model embed_vowel.c --input in.wav --output ~/out.png

# Running with model params .pkl.
run_phone_model_on_wav --model params.pkl --input in.wav --output ~/out.png
"""

import os.path

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np

from audio.dsp.portable.python import wav_io
from audio.tactile.frontend.python import carl_frontend
from audio.tactile.jax import hk_util
from audio.tactile.phone_embedding import phone_model
from audio.tactile.phone_embedding import phone_util
from audio.tactile.phone_embedding.python import embed_vowel
from audio.tactile.python import dsp
from audio.tactile.python import plot

FLAGS = flags.FLAGS

flags.DEFINE_string('input', None,
                    '[REQUIRED] Input WAV file.')

flags.DEFINE_string('model', 'embed_vowel.c',
                    "Model .pkl file, or 'embed_vowel.c' for C implementation.")

flags.DEFINE_string('output', None,
                    'If specified, save plot image to this filename. Otherwise '
                    'show in a figure window [assumes matplotlib uses "TkAgg" '
                    'or another backend capable of user interface].')


def get_phone_net(model_file):
  """Get the phone embedding network.

  Args:
    model_file: String, either the path of a .pkl model file, or 'embed_vowel.c'
    to use the C implementation.
  Returns:
    (model_fun, target_names) 2-tuple. `model_fun(frames)` runs inference taking
    2D array `frames` as input and returning a 2D array of unnormalized scores
    as output, and `target_names` is a list of class names.
  """
  if model_file == 'embed_vowel.c':
    model_fun = embed_vowel.embed_vowel_scores
    target_names = embed_vowel.TARGET_NAMES
  else:
    model = hk_util.TrainedModel.load(
        model_file, phone_model.model_fun, phone_model.Metadata)
    target_names = model.meta.classes
    window_size = 1 + model.meta.dataset_metadata['num_frames_left_context']

    def model_fun(x):
      return model({'observed': dsp.sliding_window(x, window_size)})['scores']

  return model_fun, target_names


def plot_output(frames, frame_rate, scores, target_names, title):
  """Plot frontend frames and vowel net score outputs."""
  # Soft max to convert scores to probabilities in [0, 1].
  target_probs = np.exp2(4 * scores)
  target_probs /= target_probs.sum(axis=1, keepdims=True)

  t_limits = (0.0, (len(frames) - 1) / frame_rate)

  fig = plt.figure(figsize=(10, 6))
  ax1, ax2 = fig.subplots(2, 1, sharex=True)
  extent = t_limits + (frames.shape[1], 0)
  ax1.imshow(frames.transpose(), aspect='auto', interpolation='None',
             cmap='density', extent=extent)
  ax1.set_xlim(t_limits)
  ax1.set_title(title)
  ax1.set_ylabel('Input')

  extent = t_limits + (len(target_names) - 0.5, -0.5)
  # Plot sqrt of target_probs to enhance visualization of smaller outputs.
  ax2.imshow(np.sqrt(target_probs).transpose(), interpolation='None',
             aspect='auto', cmap='density', extent=extent, vmin=0.0, vmax=1.0)
  ax2.set(yticks=np.arange(len(target_names)),
          yticklabels=target_names)

  ax2.set_xlim(t_limits)
  ax2.set_xlabel('Time (s)')
  ax2.set_ylabel('Score')
  fig.set_tight_layout(True)
  return fig


def main(_):
  # Read WAV file.
  samples, sample_rate_hz = wav_io.read_wav_file(FLAGS.input, dtype=np.float32)
  samples = samples.mean(axis=1)

  # Make the frontend and network.
  frontend = carl_frontend.CarlFrontend(input_sample_rate_hz=sample_rate_hz)
  phone_net, target_names = get_phone_net(FLAGS.model)

  # Run audio-to-phone inference.
  frames = phone_util.run_frontend(frontend, samples)
  frame_rate = sample_rate_hz / frontend.block_size
  scores = phone_net(frames)

  fig = plot_output(frames, frame_rate, scores, target_names,
                    os.path.basename(FLAGS.input) + '\n' + FLAGS.model)

  if FLAGS.output:  # Save plot as an image file.
    plot.save_figure(FLAGS.output, fig)
  else:  # Show plot interactively.
    plt.show()
  return 0


if __name__ == '__main__':
  flags.mark_flag_as_required('input')
  app.run(main)

