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

"""Program that ingests the TIMIT dataset to a form that is easier to use.

The TIMIT dataset stores audio samples in an obscure file format called "NIST
SPHERE". This program converts TIMIT's sphere files to regular WAV files for
easier use. Additionally, the directory structure is flattened to "train" and
"test" folders of files of the form

  <dialectregion>_<speaker>_<recording>.wav

Instructions:

1. To use this program, first download TIMIT, e.g. from
http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC93S1

2. Extract the contents to a local directory.

3. Run this program, specifying the downloaded TIMIT directory with --input_dir
and a desired output location with --output_dir.
"""

import glob
import os
import os.path
import shutil

from absl import app
from absl import flags

from extras.python import dsp
from extras.python.phonetics import read_nist_sphere

NUM_EXAMPLES = 6300
MODES = ('TEST', 'TRAIN')

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, '[REQUIRED] Input directory.')
flags.DEFINE_string('output_dir', None, '[REQUIRED] Output directory.')


def convert_one_example(sphere_file: str, wav_file: str) -> None:
  """Converts sphere file to WAV."""
  sample_rate_hz, samples = read_nist_sphere.read_nist_sphere(sphere_file)
  dsp.write_wav_file(wav_file, samples, sample_rate_hz)

  # TIMIT includes .phn files that gives timestamps and labels of the phones
  # that occur in the recording. We copy these files to the output directory.
  src_phn = os.path.splitext(sphere_file)[0] + '.PHN'
  dest_phn = os.path.splitext(wav_file)[0] + '.phn'
  shutil.copyfile(src_phn, dest_phn)


def main(_) -> int:
  timit_dir = FLAGS.input_dir

  # If we started in TIMIT's root directory, go into the TIMIT subdirectory.
  if os.path.isdir(os.path.join(timit_dir, 'TIMIT')):
    timit_dir = os.path.join(timit_dir, 'TIMIT')
  # Check for expected subdirectories.
  for mode in MODES:
    subdir = os.path.join(timit_dir, mode)
    if not os.path.isdir(subdir):
      print('Error: Not found:', subdir)
      return 1
    if not os.path.isdir(os.path.join(FLAGS.output_dir, mode.lower())):
      os.makedirs(os.path.join(FLAGS.output_dir, mode.lower()))

  count = 0
  print('Ingesting TIMIT:')
  print('\r  %4d / %d' % (count, NUM_EXAMPLES), end='', flush=True)

  # Iterate over modes, dialect/region, speaker, recording.
  for mode in MODES:
    mode_dir = os.path.join(timit_dir, mode)
    for dr in os.listdir(mode_dir):
      dr_dir = os.path.join(mode_dir, dr)
      for speaker in os.listdir(dr_dir):
        speaker_dir = os.path.join(dr_dir, speaker)
        for sphere_file in glob.glob(os.path.join(speaker_dir, '*.WAV')):
          wav_file = os.path.join(
              FLAGS.output_dir, mode.lower(), '_'.join(
                  (dr, speaker, os.path.basename(sphere_file))).lower())
          convert_one_example(sphere_file, wav_file)
          count += 1
          print('\r  %4d / %d' % (count, NUM_EXAMPLES), end='', flush=True)

  print('')
  if count != NUM_EXAMPLES:
    print('Error: Unexpected number of examples.')
    return 1

  return 0


if __name__ == '__main__':
  flags.mark_flag_as_required('input_dir')
  flags.mark_flag_as_required('output_dir')
  app.run(main)
