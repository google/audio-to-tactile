# Copyright 2022 Google LLC
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

"""Tests for tap_out."""

import unittest

import numpy as np

from extras.python import tap_out


def convert_to_bytes(x):
  return bytes(ord(x1) if isinstance(x1, str) else x1 for x1 in x)


# Same as the test data generated in the C unit test tap_out_test.c.
TEST_DESCRIPTOR_DATA = convert_to_bytes((
    1,  # Apple descriptor.
    'A', 'p', 'p', 'l', 'e', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    9, 3, 8, 0,
    2,  # Banana descriptor.
    'B', 'a', 'n', 'a', 'n', 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0,
    11, 30, 0, 0,
    3,  # Chocolate cherry descriptor.
    'C', 'h', 'o', 'c', 'o', 'l', 'a', 't', 'e', ' ', 'c', 'h', 'e', 'r', 'r',
    5, 9, 0, 0))


class TapOutTest(unittest.TestCase):

  def test_parse_descriptors(self):
    descriptors = tap_out.parse_descriptors(TEST_DESCRIPTOR_DATA)

    self.assertIn(1, descriptors)
    self.assertEqual(descriptors[1].name, 'Apple')
    self.assertEqual(descriptors[1].dtype, tap_out.DType.FLOAT)
    self.assertEqual(descriptors[1].shape, (3, 8))

    self.assertIn(2, descriptors)
    self.assertEqual(descriptors[2].name, 'Banana')
    self.assertEqual(descriptors[2].dtype, tap_out.DType.TEXT)
    self.assertEqual(descriptors[2].shape, (30,))

    self.assertIn(3, descriptors)
    self.assertEqual(descriptors[3].name, 'Chocolate cherr')
    self.assertEqual(descriptors[3].dtype, tap_out.DType.UINT32)
    self.assertEqual(descriptors[3].shape, (9,))

  def test_read_numeric(self):
    descriptors = tap_out.parse_descriptors(TEST_DESCRIPTOR_DATA)

    # Generate and serialize random test data.
    np.random.seed(0)
    cherry = np.random.randint(1000000, size=(7, 9),
                               dtype=descriptors[3].dtype.numpy_dtype)
    apple = np.random.randn(7, 3, 8).astype(descriptors[1].dtype.numpy_dtype)
    data = b''
    for i in range(len(apple)):
      data += cherry[i].tobytes() + apple[i].tobytes()

    # Test that reader can deserialize the data.
    reader = tap_out.make_reader([descriptors[3], descriptors[1]])
    outputs = reader(data)

    np.testing.assert_array_equal(outputs[0], cherry.reshape(-1))
    np.testing.assert_array_equal(outputs[1], apple.reshape(-1, 8))

  def test_read_text(self):
    descriptors = tap_out.parse_descriptors(TEST_DESCRIPTOR_DATA)

    # Serialize test data.
    banana = ['Hey there', 'How are you?', 'Good bye']
    data = b''
    for line in banana:
      data += bytes(line, 'ascii') + b'\x00' * (30 - len(line))

    # Test that reader can deserialize the data.
    reader = tap_out.make_reader([descriptors[2]])
    outputs = reader(data)

    self.assertEqual(outputs[0], banana)


if __name__ == '__main__':
  unittest.main()
