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

import datetime
import unittest

import numpy as np

from extras.python.tactile import tap_out


def convert_to_bytes(x):
  return bytes(ord(x1) if isinstance(x1, str) else x1 for x1 in x)


# Same as the test data generated in the C unit test tap_out_test.c.
TEST_DESCRIPTOR_DATA = convert_to_bytes((
    94, 138, 52, 1,  # uint32 value encoding 20220510.
    3,  # Number of descriptors.
    1,  # Apple descriptor.
    'A', 'p', 'p', 'l', 'e', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    9, 3, 8, 0,
    2,  # Banana descriptor.
    'B', 'a', 'n', 'a', 'n', 'a', 0, 0, 0, 0, 0, 0, 0, 0, 0,
    11, 30, 0, 0,
    3,  # "Chocolate cherr" (long truncated name) descriptor.
    'C', 'h', 'o', 'c', 'o', 'l', 'a', 't', 'e', ' ', 'c', 'h', 'e', 'r', 'r',
    5, 9, 0, 0))


class FakeUart:
  """A fake UART serial object, used below in `test_capture()`."""

  def __init__(self):
    self.data_to_be_read = b''
    self.data_written = b''

  def read(self, num_bytes: int = 1) -> bytes:
    result = self.data_to_be_read[:num_bytes]
    self.data_to_be_read = self.data_to_be_read[num_bytes:]
    return result

  def write(self, data: bytes) -> None:
    self.data_written += data

  def flush(self) -> None:
    pass


class TapOutTest(unittest.TestCase):

  def test_parse_descriptors(self):
    descriptors, build_date = tap_out.parse_descriptors(TEST_DESCRIPTOR_DATA)

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

    self.assertEqual(build_date, datetime.date(2022, 5, 10))

  def test_read_numeric(self):
    descriptors, _ = tap_out.parse_descriptors(TEST_DESCRIPTOR_DATA)

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
    output = reader(data)

    np.testing.assert_array_equal(output['Chocolate cherr'], cherry.reshape(-1))
    np.testing.assert_array_equal(output['Apple'], apple.reshape(-1, 8))

  def test_read_text(self):
    descriptors, _ = tap_out.parse_descriptors(TEST_DESCRIPTOR_DATA)

    # Serialize test data.
    banana = ['Hey there', 'How are you?', 'Good bye']
    data = b''
    for line in banana:
      data += bytes(line, 'ascii') + b'\x00' * (30 - len(line))

    # Test that reader can deserialize the data.
    reader = tap_out.make_reader([descriptors[2]])
    output = reader(data)

    self.assertEqual(output['Banana'], banana)

  def test_capture(self):
    np.random.seed(0)

    uart = FakeUart()
    uart.data_to_be_read = (
        tap_out.MARKER +
        bytes([tap_out.OP_DESCRIPTORS, len(TEST_DESCRIPTOR_DATA)]) +
        TEST_DESCRIPTOR_DATA)

    comm = tap_out.TapOut(uart)
    descriptors, _ = comm.get_descriptors()

    self.assertEqual(uart.data_written,
                     tap_out.MARKER + bytes([tap_out.OP_GET_DESCRIPTORS, 0]))

    num = 7
    cherry = np.random.randint(
        1000000, size=(num, 9),
        dtype=descriptors['Chocolate cherr'].dtype.numpy_dtype)
    apple = np.random.randn(
        num, 3, 8).astype(descriptors['Apple'].dtype.numpy_dtype)
    for i in range(num):
      payload = cherry[i].tobytes() + apple[i].tobytes()
      uart.data_to_be_read += (
          tap_out.MARKER + bytes([tap_out.OP_CAPTURE, len(payload)]) + payload)
    uart.data_written = b''

    captured = comm.capture(['Chocolate cherr', 'Apple'], num)

    self.assertEqual(uart.data_written,
                     tap_out.MARKER +
                     bytes([tap_out.OP_START_CAPTURE, 2, 3, 1]) +
                     tap_out.MARKER +
                     bytes([tap_out.OP_HEARTBEAT, 0]))
    np.testing.assert_array_equal(
        captured['Chocolate cherr'], cherry.reshape(-1))
    np.testing.assert_array_equal(
        captured['Apple'], apple.reshape(-1, 8))


if __name__ == '__main__':
  unittest.main()
