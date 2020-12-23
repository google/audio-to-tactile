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

"""Test for phone_model."""

import os.path

from absl import flags
from absl.testing import absltest
import numpy as np

from extras.python.phonetics import hk_util
from extras.python.phonetics import phone_model
from extras.python.phonetics import phone_util

FLAGS = flags.FLAGS


def random_dataset():
  np.random.seed(0)
  num_frames_left_context = 2
  num_channels = 16
  num_frames = num_frames_left_context + 1
  examples = {
      'aa': np.random.rand(
          10, num_frames, num_channels).astype(np.float32),
      'eh': np.random.rand(
          14, num_frames, num_channels).astype(np.float32),
      'iy': np.random.rand(
          11, num_frames, num_channels).astype(np.float32),
  }
  metadata = {
      'num_frames_left_context': num_frames_left_context,
      'num_channels': num_channels,
  }
  return phone_util.Dataset(examples, metadata)


class PhoneModelTest(absltest.TestCase):

  def test_model_workflow(self):
    """Test training for a few steps, saving, loading, and eval."""
    meta = phone_model.Metadata.from_flags()
    meta.classes = ['aa', 'eh', 'iy']
    meta.hidden_units = [8, 8]
    meta.batch_size = 4
    meta.num_steps = 3

    # Load train and test datasets.
    dataset = random_dataset()
    meta.dataset_metadata = dataset.metadata

    # Train the model.
    model = phone_model.train_model(meta, dataset)

    # Save model.
    output_dir = os.path.join(FLAGS.test_tmpdir, 'phone_model_output')
    model.save(os.path.join(output_dir, 'params.pkl'))

    # Load model.
    recovered = hk_util.TrainedModel.load(
        os.path.join(output_dir, 'params.pkl'),
        phone_model.model_fun,
        phone_model.Metadata)

    # Evaluate the model.
    phone_model.eval_model(recovered, dataset, output_dir)


if __name__ == '__main__':
  absltest.main()
