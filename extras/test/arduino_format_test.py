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

"""Some limited checks that repo follows the Arduino library format.

This project has a specific directory structure in order to conform to the
Arduino library format. All embedded library code is under a srcs/ folder,
examples are in examples/, and all other non-library code (including tests) is
in extras/. For details, see
https://arduino.github.io/arduino-cli/latest/library-specification/

This format is required for publication in the Arduino Library Manager:
https://github.com/arduino/Arduino/wiki/Library-Manager-FAQ
"""

import os
import os.path
import unittest

REPO_ROOT = './'


def walk_dir_tree(directory):
  for dir_name, _, file_names in os.walk(directory):
    for file_name in file_names:
      yield os.path.join(dir_name, file_name)


class ArduinoFormatTest(unittest.TestCase):

  def _check_file_extensions(self, directory, allowed_extensions):
    """Checks that all files under `directory` have an allowed extension."""
    self.assertTrue(
        os.path.isdir(directory), msg=f'"{directory}" is not a directory')
    for filename in walk_dir_tree(directory):
      self.assertIn(
          os.path.splitext(filename)[1],
          allowed_extensions,
          msg=f'"{filename}" has bad extension')

  def test_root(self):
    """Test that repo root has no code files."""
    for filename in os.listdir(REPO_ROOT):
      self.assertNotIn(
          os.path.splitext(filename)[1].lower(),
          ('.cpp', '.c', '.h', '.cc', '.s', '.ino', '.sh'),
          msg=f'Code file "{filename}" not allowed in repo root.')

  def test_examples_dir(self):
    """Test that examples/ dir contains only C/C++ code and INO sketches."""
    examples_dir = os.path.join(REPO_ROOT, 'examples')
    self._check_file_extensions(examples_dir, ('.ino', '.cpp', '.c', '.h'))

  def test_src_dir(self):
    """Test that src/ dir contains only C/C++ library code."""
    src_dir = os.path.join(REPO_ROOT, 'src')
    self._check_file_extensions(src_dir, ('.cpp', '.c', '.h'))

    for filename in walk_dir_tree(src_dir):
      # Look for filename ending in "_test.c" or "_test.cpp".
      suffix = os.path.splitext(filename)[0].rsplit('_', 1)[-1]
      self.assertNotEqual(suffix.lower(), 'test',
                          msg=f'Test "{filename}" must go under extras/test.')

  def test_library_properties_file(self):
    """Test that `library.properties` file exists."""
    self.assertTrue(
        os.path.isfile(os.path.join(REPO_ROOT, 'library.properties')))


if __name__ == '__main__':
  unittest.main()
