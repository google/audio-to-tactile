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


Export trained JAX model parameters as C data.

This program reads a trained parameters from a .pkl pickle file and exports them
as C data in a form similar to embed_vowel_params.h. Each tensor in the model is
flattened in column-major order and written in the form

  static const float kTensorName[dim0 * dim1 * ...] = {elements... };

NOTE: This program does not convert model behavior to C; only the parameter data
is exported. It is up to the user to understand the model architecture and
parameter meanings. This may yet help in writing C implementations for inference
on device.
"""

import textwrap
from typing import Iterable

from absl import app
from absl import flags
import numpy as np

from audio.tactile.jax import hk_util
from audio.tactile.phone_embedding import phone_model

FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '[REQUIRED] Model params .pkl file.')

flags.DEFINE_string('output', '/tmp/params.h', 'Output C file.')

MAX_WIDTH = 80  # Output is wrapped to MAX_WIDTH chars.


def snake_case_to_camel(snake: str) -> str:
  """Convert snake_case_name to CamelCaseName."""
  return ''.join(w.title() for w in snake.split('_'))


def format_c_array(v: Iterable[float]) -> str:
  """Format 1D array as a C array."""
  return '{%s}' % ', '.join(['%.6ff' % x for x in v])


def export_model_as_c_data(model_file: str, output_file: str) -> None:
  """Export model as C data.

  Args:
    model_file: String, model params pickle file.
    output_file: String, output C file to write.
  """
  model = hk_util.TrainedModel.load(
      model_file, phone_model.model_fun, phone_model.Metadata)

  s = []
  print('\nModel parameters:')
  print('  %-20s %-12s %s' % ('name', 'dtype', 'shape'))
  for name, array in hk_util.params_as_list(model.params):
    name = 'k' + snake_case_to_camel(name.replace('.', '_'))
    array = np.asarray(array)
    print('  %-20s %-12s %s' % (name, array.dtype, array.shape))
    s.append(textwrap.fill(
        'static const float %s[%s] = '
        % (name, ' * '.join(map(str, array.shape)))
        + format_c_array(array.flatten(order='F')) + ';',
        MAX_WIDTH, subsequent_indent='    ') + '\n')

  with open(output_file, 'wt') as f:
    f.write('/* Model parameters. */\n\n' + '\n'.join(s))
  print('Exported to %s' % output_file)


def main(_):
  export_model_as_c_data(FLAGS.model, FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('model')
  app.run(main)

