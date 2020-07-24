# Lint as: python2, python3
r"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Tests for wav_io Python bindings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import os.path
import struct
import tempfile
import unittest

import numpy as np
import six

from audio.dsp.portable.python import wav_io

try:
  FileNotFoundError
except NameError:
  # Needed for Python 2 compatilibity.
  FileNotFoundError = IOError  # pylint: disable=redefined-builtin


def make_24_bit_wav(samples, sample_rate_hz):
  """Makes a 24-bit WAV."""
  num_frames, num_channels = samples.shape
  block_align = 3 * num_channels
  # Numpy doesn't have a 24-bit dtype, so serialize as int32 and remove LSBs.
  data = bytearray(samples.astype('<i4').tobytes())
  del data[::4]
  return (
      b'RIFF\x00\x00\x00\x00WAVEfmt (\x00\x00\x00\xfe\xff'
      + struct.pack('<hIIh', num_channels, sample_rate_hz,
                    block_align * sample_rate_hz, block_align)
      + b'\x18\x00\x16\x00\x18\x00\x04\x00\x00\x00\x01\x00\x00\x00\x00\x00\x10'
      + b'\x00\x80\x00\x00\xaa\x008\x9bqfact\x04\x00\x00\x00'
      + struct.pack('<I', num_frames)
      + b'data' + struct.pack('<I', len(data)) + data)


def make_float_wav(samples, sample_rate_hz):
  """Makes a 32-bit float WAV."""
  num_frames, num_channels = samples.shape
  block_align = 4 * num_channels
  data = samples.astype('<f4').tobytes()
  return (
      b'RIFF\x00\x00\x00\x00WAVEfmt \x12\x00\x00\x00\x03\x00'
      + struct.pack('<hIIh', num_channels, sample_rate_hz,
                    block_align * sample_rate_hz, block_align)
      + b' \x00\x00\x00fact\x04\x00\x00\x00'
      + struct.pack('<I', num_frames)
      + b'data' + struct.pack('<I', len(data)) + data)


class WavDifferentIoStreamsTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(WavDifferentIoStreamsTest, cls).setUpClass()
    cls.temp_dir = tempfile.mkdtemp(suffix='wav_io_test')

    # Generate 48kHz stereo WAV file with 16-bit PCM samples `wav_samples`.
    n = np.arange(200, dtype=np.int16)
    cls.wav_samples = np.column_stack((10 * n + 1, 10 * n + 2))
    cls.wav_bytes = (
        b'RIFFD\x03\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00'
        + b'\x80\xbb\x00\x00\x00\xee\x02\x00\x04\x00\x10\x00data \x03\x00\x00'
        + cls.wav_samples.astype('<i2').tobytes())

    # Write as local temp file.
    cls.read_filename = os.path.join(cls.temp_dir, 'read.wav')
    with open(cls.read_filename, 'wb') as f:
      f.write(cls.wav_bytes)

  @classmethod
  def tearDownClass(cls):
    super(WavDifferentIoStreamsTest, cls).tearDownClass()

    os.remove(cls.read_filename)
    os.rmdir(cls.temp_dir)

  def test_read_wav_given_filename(self):
    """Read WAV given a filename with read_wav_file()."""
    samples, sample_rate_hz = wav_io.read_wav_file(self.read_filename)

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_from_bytes(self):
    """Read WAV from a byte string with from_bytes()."""
    samples, sample_rate_hz = wav_io.from_bytes(self.wav_bytes)

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_read_wav_given_local_file_object(self):
    """Read WAV given a local file object."""
    with open(self.read_filename, 'rb') as f:
      samples, sample_rate_hz = wav_io.read_wav_file(f)

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_read_wav_given_memory_stream(self):
    """Read WAV from an in-memory stream."""
    samples, sample_rate_hz = wav_io.read_wav_file(io.BytesIO(self.wav_bytes))

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_write_wav_local_file(self):
    """Write WAV to a given filename with write_wav_file()."""
    try:
      write_filename = os.path.join(self.temp_dir, 'write.wav')
      wav_io.write_wav_file(write_filename, self.wav_samples, 44100)

      samples, sample_rate_hz = wav_io.read_wav_file(write_filename)
      np.testing.assert_array_equal(samples, self.wav_samples)
      self.assertEqual(sample_rate_hz, 44100)
    finally:
      if os.path.isfile(write_filename):
        os.remove(write_filename)

  def test_to_bytes(self):
    """Write WAV to byte string with to_bytes()."""
    wav_bytes = wav_io.to_bytes(self.wav_samples, 44100)

    samples, sample_rate_hz = wav_io.from_bytes(wav_bytes)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 44100)


class MockReader(object):

  def __init__(self, read_fun):
    self.read = read_fun


class MockWriter(object):

  def __init__(self, write_fun):
    self.write = write_fun


class WavIoTest(unittest.TestCase):

  def assert_equal_same_dtype(self, x, y):
    """Asserts that arrays x and y have equal elements and same dtype."""
    self.assertEqual(x.dtype, y.dtype)
    np.testing.assert_array_equal(x, y)

  def test_read_24_bit_wav(self):
    """Read a 48kHz mono WAV file with 24-bit samples."""
    np.random.seed(0)
    expected = np.random.randint(-2**23, 2**23 - 1, size=(20, 3)) * 256
    wav_bytes = make_24_bit_wav(expected, 44100)

    samples, sample_rate_hz = wav_io.from_bytes(wav_bytes)

    self.assertEqual(samples.dtype, np.int32)
    np.testing.assert_array_equal(samples, expected)
    self.assertEqual(sample_rate_hz, 44100)

    # Read with conversion to float32.
    samples, _ = wav_io.from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(
        samples, expected.astype(np.float32) / 2.0**31)

  def test_read_float_wav(self):
    """Read a 48kHz mono WAV file with 32-bit float samples."""
    np.random.seed(0)
    expected = np.random.randn(15, 4).astype(np.float32)
    wav_bytes = make_float_wav(expected, 48000)

    samples, sample_rate_hz = wav_io.from_bytes(wav_bytes)

    self.assertEqual(samples.dtype, np.float32)
    np.testing.assert_array_equal(samples, expected)
    self.assertEqual(sample_rate_hz, 48000)

  def test_read_16_bit_wav_with_dtype(self):
    """Test reading a 16-bit WAV with conversion to specified dtype."""
    samples = np.expand_dims(
        [0, 1, 2, -5, 25000, 32767, -32768], axis=1).astype(np.int16)
    wav_bytes = wav_io.to_bytes(samples, 8000)

    # int16 -> int16.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.int16)
    self.assert_equal_same_dtype(out, samples)
    # int16 -> int32.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.int32)
    self.assert_equal_same_dtype(out, samples.astype(np.int32) * 2**16)
    # int16 -> float32.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(out, samples.astype(np.float32) / 2.0**15)

  def test_read_24_bit_wav_with_dtype(self):
    """Test reading a 24-bit WAV with conversion to specified dtype."""
    samples = 256 * np.expand_dims(
        [1, -1500000, 2**23 - 1, -2**23], axis=1).astype(np.int32)
    wav_bytes = make_24_bit_wav(samples, 8000)

    # int32 -> int16.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.int16)
    self.assert_equal_same_dtype(
        out, np.expand_dims([0, -5859, 32767, -32768], axis=1).astype(np.int16))
    # int32 -> int32.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.int32)
    self.assert_equal_same_dtype(out, samples)
    # int32 -> float32.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(out, samples.astype(np.float32) / 2.0**31)

  def test_read_float_wav_with_dtype(self):
    """Test reading a float WAV with conversion to specified dtype."""
    samples = np.expand_dims(
        [0.0, 1e-6, -1e-4, 0.1, -0.5, 1.0, -1.0,
         np.inf, -np.inf, np.nan], axis=1).astype(np.float32)
    wav_bytes = make_float_wav(samples, 8000)

    # float32 -> int16.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.int16)
    self.assert_equal_same_dtype(
        out, np.expand_dims([0, 0, -3, 3277, -16384, 32767, -32768,
                             32767, -32768, 0], axis=1).astype(np.int16))
    # float32 -> int32.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.int32)
    self.assert_equal_same_dtype(
        out, np.expand_dims([
            0, 2147, -214748, 214748368, -1073741824, 2147483647,
            -2147483648, 2147483647, -2147483648, 0], axis=1).astype(np.int32))
    # float32 -> float32.
    out, _ = wav_io.from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(out, samples)

  def test_write_wav_1d_array(self):
    """Test writing a 1D array as a mono WAV file."""
    samples = np.arange(20, dtype=np.int16)
    recovered, sample_rate_hz = wav_io.from_bytes(
        wav_io.to_bytes(samples, 8000))

    np.testing.assert_array_equal(recovered, samples.reshape(-1, 1))
    self.assertEqual(sample_rate_hz, 8000)

  def test_read_wav_bad_arg(self):
    """Call where the argument is not a file-like object."""

    class Nonsense(object):
      pass

    with six.assertRaisesRegex(self, TypeError, 'Nonsense found'):
      wav_io.read_wav_file(Nonsense())

  def test_read_wav_read_not_callable(self):
    """Test where the read attribute is not callable."""
    reader = MockReader(None)
    with six.assertRaisesRegex(self, TypeError, 'not callable'):
      wav_io.read_wav_file(reader)

  def test_read_wav_reader_raises_exception(self):
    """Test where the file object read method raises an exception."""
    def _failing_read(unused_size):
      raise OSError('read method failed')
    reader = MockReader(_failing_read)
    with six.assertRaisesRegex(self, OSError, 'read method failed'):
      wav_io.read_wav_file(reader)

  def test_read_wav_reader_returns_wrong_type(self):
    """Test where the read method returns the wrong type."""
    reader = MockReader(lambda size: [0] * size)
    with six.assertRaisesRegex(self, TypeError, 'list found'):
      wav_io.read_wav_file(reader)

  def test_read_wav_reader_result_too_large(self):
    """Test where the read method returns more than requested."""
    reader = MockReader(lambda size: b'\000' * (size + 1))
    with six.assertRaisesRegex(self, ValueError, 'exceeds requested size'):
      wav_io.read_wav_file(reader)

  def test_read_wav_bad_dtype(self):
    """Test where WAV fact chunk is corrupt."""
    with six.assertRaisesRegex(self, ValueError, 'dtype must be one of'):
      wav_io.from_bytes(b'RIFF', dtype=np.uint8)

  def test_read_wav_bad_fact_chunk(self):
    """Test where WAV fact chunk is corrupt."""
    with six.assertRaisesRegex(self, OSError, 'error reading WAV header'):
      wav_io.from_bytes(b'RIFF_\000\000\000WAVEfactbusted')

  def test_write_wav_bad_arg(self):
    """write_wav_file where the argument is not a file-like object."""

    class Nonsense(object):
      pass

    with six.assertRaisesRegex(self, TypeError, 'Nonsense found'):
      wav_io.write_wav_file(Nonsense(), np.zeros((10, 1), dtype=np.int16), 8000)

  def test_write_wav_wrong_dtype(self):
    """write_wav_file where samples can't safely cast to np.int16 dtype."""
    samples = np.array([-0.2, 0.5, 0.7, 0.3, 0.1])
    with six.assertRaisesRegex(self, TypeError, 'Cannot cast array data'):
      wav_io.to_bytes(samples, 8000)

  def test_write_wav_write_not_callable(self):
    """write_wav_file where the write attribute is not callable."""
    writer = MockWriter(None)
    with six.assertRaisesRegex(self, TypeError, 'not callable'):
      wav_io.write_wav_file(writer, np.zeros((10, 1), dtype=np.int16), 8000)

  def test_write_wav_writer_raises_exception(self):
    """write_wav_file where the file object write method raises an exception."""
    def _failing_write(unused_bytes):
      raise OSError('write method failed')
    writer = MockWriter(_failing_write)
    with six.assertRaisesRegex(self, OSError, 'write method failed'):
      wav_io.write_wav_file(writer, np.zeros((10, 1), dtype=np.int16), 8000)


if __name__ == '__main__':
  unittest.main()

