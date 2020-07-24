# Lint as: python3
r"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Train and eval a network for mapping audio to 2D vowel space coordinate.
"""

import os.path

from absl import app
from absl import flags

from audio.tactile.phone_embedding import phone_model

flags.DEFINE_string('output', None,
                    '[REQUIRED] Output directory.')
flags.DEFINE_string('train_npz', None,
                    '[REQUIRED] Path of .npz file to read for training data.')
flags.DEFINE_string('test_npz', None,
                    '[REQUIRED] Path of .npz file to read for testing data.')

FLAGS = flags.FLAGS


def main(_):
  meta = phone_model.Metadata.from_flags()

  class_weights = {
      'aa': 0.4,
      'uw': 1.8,
      'ih': 1.55,
      'iy': 1.0,
      'eh': 1.7,
      'ae': 1.0,
      'uh': 1.3,
      'ah': 1.3,
      'er': 0.5,
  }
  class_weights = {k: v**0.85 for k, v in class_weights.items()}

  # Load train and test datasets.
  dataset_train = phone_model.load_dataset(
      FLAGS.train_npz, meta.classes, class_weights)
  dataset_test = phone_model.load_dataset(FLAGS.test_npz, meta.classes)
  meta.dataset_metadata = dataset_train.metadata

  # Train the model.
  model = phone_model.train_model(meta, dataset_train)
  # Save model to FLAGS.output.
  model.save(os.path.join(FLAGS.output, 'params.pkl'))

  # Evaluate the model.
  phone_model.eval_model(model, dataset_test, FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('output')
  flags.mark_flag_as_required('train_npz')
  flags.mark_flag_as_required('test_npz')
  app.run(main)

