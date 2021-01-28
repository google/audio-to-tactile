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

"""Tests for dsp.py."""

import fractions
import io
import os
import os.path
import struct
import tempfile
import unittest

import numpy as np

from extras.python import dsp


class FastFunTest(unittest.TestCase):

  def test_fast_log2_accuracy(self):
    # An arbitrary spot check at x=4.2 with a scalar input.
    output = dsp.fast_log2(np.float32(4.2))
    self.assertIsInstance(output, np.float32)
    self.assertAlmostEqual(output, np.log2(4.2), delta=0.002)

    # Check thoroughly at random positions with an array input. To test a wide
    # range, we first make x in [-125.5, 125.5], then make y = exp2(x), so that
    # y is distributed over most of the finite float range, excluding denormals.
    x = np.random.uniform(-125.5, 125.5, size=10000).astype(np.float32)
    y = 2.0**x
    output = dsp.fast_log2(y)
    self.assertTupleEqual(output.shape, x.shape)
    self.assertEqual(output.dtype, np.float32)
    max_abs_error = np.max(np.abs(output - x))
    self.assertLess(max_abs_error, 0.003)

  def test_fast_exp2_accuracy(self):
    # An arbitrary spot check at x=4.2 with a scalar input.
    output = dsp.fast_exp2(np.float32(4.2))
    self.assertIsInstance(output, np.float32)
    self.assertAlmostEqual(output / np.exp2(4.2), 1.0, delta=6e-4)

    x = np.random.uniform(-125.5, 125.5, size=10000).astype(np.float32)
    y = 2.0**x
    output = dsp.fast_exp2(x)
    self.assertTupleEqual(output.shape, x.shape)
    self.assertEqual(output.dtype, np.float32)
    max_rel_error = np.max(np.abs(output / y - 1.0))
    self.assertLess(max_rel_error, 0.003)

  def test_fast_pow_accuracy(self):
    # Check x^y over a 2-D grid of points 0.1 <= x <= 50, -2 <= y <= 2.
    x = (np.arange(1, 501, dtype=np.float32) * 0.1)[np.newaxis, :]
    y = (np.arange(-20, 21, dtype=np.float32) * 0.1)[:, np.newaxis]

    output = dsp.fast_pow(x, y)
    self.assertTupleEqual(output.shape, (41, 500))
    self.assertEqual(output.dtype, np.float32)
    max_rel_error = np.max(np.abs(output / x**y - 1.0))
    self.assertLess(max_rel_error, 0.005)

  def test_fast_tanh_accuracy(self):
    x = np.random.uniform(-12.0, 12.0, size=10000).astype(np.float32)

    output = dsp.fast_tanh(x)
    self.assertTupleEqual(output.shape, x.shape)
    self.assertEqual(output.dtype, np.float32)
    max_abs_error = np.max(np.abs(output - np.tanh(x)))
    self.assertLess(max_abs_error, 0.0008)

    # Check large arguments.
    self.assertEqual(dsp.fast_tanh(np.float32(0.0)), 0.0)
    self.assertEqual(dsp.fast_tanh(np.float32(1000.0)), 1.0)
    self.assertEqual(dsp.fast_tanh(np.float32(-1000.0)), -1.0)


# Tested resampling sample rates in Hz.
RATES = (12000, 16000, 32000, 44100, 48000, 16000 * np.sqrt(2))


def make_message(options):
  """Returns formatted string describing `options` dict."""
  return 'Options: ' + ', '.join('%s=%s' % (k, options[k]) for k in options)


class ResamplerKernelTest(unittest.TestCase):

  def test_resampler_kernel(self):
    """Test ResamplerKernel for various sample rates and support radii."""
    for filter_radius_factor in (5.0, 17.0):
      for input_sample_rate_hz in RATES:
        for output_sample_rate_hz in RATES:
          cutoff_proportion = 0.85
          kaiser_beta = 6.0
          options = {'input_sample_rate_hz': input_sample_rate_hz,
                     'output_sample_rate_hz': output_sample_rate_hz,
                     'filter_radius_factor': filter_radius_factor,
                     'cutoff_proportion': cutoff_proportion,
                     'kaiser_beta': kaiser_beta}
          message = make_message(options)
          kernel = dsp.ResamplerKernel(**options)
          self.assertAlmostEqual(kernel.factor * output_sample_rate_hz,
                                 input_sample_rate_hz, delta=0.005, msg=message)

          # The kernel should be zero outside of [-radius, +radius].
          self.assertEqual(kernel(-kernel.radius - 1e-6), 0.0, msg=message)
          self.assertEqual(kernel(kernel.radius + 1e-6), 0.0, msg=message)

          x = np.arange(1 + 50 * kernel.radius) / 50

          # Compare with samples of the expected kernel.
          input_nyquist = input_sample_rate_hz / 2
          output_nyquist = output_sample_rate_hz / 2
          cutoff_hz = cutoff_proportion * min(input_nyquist, output_nyquist)
          theta = cutoff_hz / input_nyquist
          support_thresh = kernel.radius * (1.0 + 100 * np.finfo(np.double).eps)
          expected_kernel = (np.abs(x) <= support_thresh) * (
              theta * np.sinc(theta * x) *
              np.i0(kaiser_beta *
                    np.sqrt(np.maximum(0, 1 - (x / kernel.radius)**2))) /
              np.i0(kaiser_beta))

          np.testing.assert_allclose(kernel(x), expected_kernel,
                                     atol=1e-6, err_msg=message)


class ResamplerTest(unittest.TestCase):

  def _reference_resampling(self, kernel, rational_factor, input_samples):
    """Reference implementation for resampling.

    Implement resampling directly according to
      x'[m] = x(m/F') = sum_n x[n] h(m F/F' - n),
    where h is the resampling kernel, F is the input sample rate, and F' is the
    output sample rate.

    Args:
      kernel: ResamplerKernel.
      rational_factor: Fraction, rational approximation of F/F'.
      input_samples: 2D numpy array.
    Returns:
      2D numpy array, resampled output.
    """
    output = []
    m = 0
    while True:
      n0 = m * rational_factor
      n_first = int(round(n0 - kernel.radius))
      n_last = int(round(n0 + kernel.radius))
      self.assertEqual(kernel(n0 - (n_first - 1)), 0.0)
      self.assertEqual(kernel(n0 - (n_last + 1)), 0.0)

      if n_last >= len(input_samples):
        break
      n = np.arange(n_first, n_last + 1)
      output.append(kernel(n0 - n).dot(
          np.expand_dims(n >= 0, axis=1) * input_samples[n]))
      m += 1

    if output:
      output = np.vstack(output)
    else:
      output = np.empty((0, input_samples.shape[1]))
    return output

  def test_compare_with_reference_resampler(self):
    """Compare Resampler to _reference_resampling() implementation."""
    np.random.seed(0)

    for filter_radius_factor in (4.0, 5.0, 17.0):
      num_channels_list = (1, 2, 3) if filter_radius_factor == 5.0 else (1,)
      for num_channels in num_channels_list:
        input_samples = -0.5 + np.random.rand(50, num_channels)
        for input_sample_rate_hz in RATES:
          for output_sample_rate_hz in RATES:
            options = {'input_sample_rate_hz': input_sample_rate_hz,
                       'output_sample_rate_hz': output_sample_rate_hz,
                       'filter_radius_factor': filter_radius_factor}
            message = make_message(options)
            resampler = dsp.Resampler(**options, num_channels=num_channels)
            self.assertEqual(resampler.num_channels, num_channels, msg=message)

            output = resampler.process_samples(input_samples)

            kernel = dsp.ResamplerKernel(**options)
            self.assertAlmostEqual(float(resampler.rational_factor),
                                   kernel.factor, delta=5e-4, msg=message)
            self.assertEqual(
                resampler.flush_frames, 2 * np.ceil(kernel.radius), msg=message)

            expected = self._reference_resampling(
                kernel, resampler.rational_factor, input_samples)
            self.assertAlmostEqual(len(output), len(expected), delta=2,
                                   msg=message)

            min_size = min(len(output), len(expected))
            np.testing.assert_allclose(
                output[:min_size], expected[:min_size], atol=5e-7,
                err_msg=message)

  def test_rational_approximation_options(self):
    """Test that rational approximation options work as expected."""
    # Request a resampling factor of pi with default options.
    resampler = dsp.Resampler(np.pi, 1.0)
    self.assertEqual(resampler.rational_factor, fractions.Fraction(355, 113))

    # Truncate continued fraction expansion at 3 terms.
    resampler = dsp.Resampler(np.pi, 1.0, rational_approximation_max_terms=3)
    self.assertEqual(resampler.rational_factor,
                     fractions.Fraction(333, 106))  # 3rd convergent [3; 7, 15].

    # Truncate when continued fraction residual is less than 0.1.
    resampler = dsp.Resampler(
        np.pi, 1.0, rational_approximation_convergence_tolerance=0.1)
    self.assertEqual(resampler.rational_factor,
                     fractions.Fraction(22, 7))  # 2nd convergent, [3; 7].

  def test_resample_sine_wave(self):
    """Test Resampler on a sine wave for various sample rates."""
    frequency = 1100.7

    for input_sample_rate_hz in RATES:
      radians_per_sample = 2 * np.pi * frequency / input_sample_rate_hz
      input_samples = np.sin(radians_per_sample * np.arange(100))

      for output_sample_rate_hz in RATES:
        options = {'input_sample_rate_hz': input_sample_rate_hz,
                   'output_sample_rate_hz': output_sample_rate_hz}
        message = make_message(options)
        resampler = dsp.Resampler(**options)
        # Run resampler on sine wave samples.
        output_samples = resampler.process_samples(input_samples)

        kernel = dsp.ResamplerKernel(
            input_sample_rate_hz, output_sample_rate_hz)
        expected_size = (len(input_samples) - kernel.radius) / kernel.factor
        self.assertAlmostEqual(len(output_samples), expected_size, delta=1.0,
                               msg=message)

        radians_per_sample = 2 * np.pi * frequency / output_sample_rate_hz
        expected = np.sin(radians_per_sample * np.arange(len(output_samples)))
        # We ignore the first few output samples because they depend on input
        # samples at negative times, which are extrapolated as zeros.
        num_to_ignore = 1 + int(kernel.radius / kernel.factor)
        np.testing.assert_allclose(output_samples[num_to_ignore:],
                                   expected[num_to_ignore:], atol=0.005,
                                   err_msg=message)

  def test_resample_chirp(self):
    """Test Resampler on a chirp signal for various sample rates."""
    duration_s = 0.025

    for input_sample_rate_hz in RATES:
      max_frequency_hz = 0.45 * input_sample_rate_hz
      chirp_slope = max_frequency_hz / duration_s

      input_size = int(duration_s * input_sample_rate_hz)
      t = np.arange(input_size) / input_sample_rate_hz
      input_samples = np.sin(np.pi * chirp_slope * t**2).astype(np.float32)

      for output_sample_rate_hz in RATES:
        options = {'input_sample_rate_hz': input_sample_rate_hz,
                   'output_sample_rate_hz': output_sample_rate_hz}
        message = make_message(options)
        resampler = dsp.Resampler(**options)
        # Run resampler on the chirp.
        output_samples = resampler.process_samples(input_samples)

        kernel = dsp.ResamplerKernel(**options)
        cutoff_hz = (kernel.radians_per_sample
                     * input_sample_rate_hz / (2 * np.pi))
        t = np.arange(len(output_samples)) / output_sample_rate_hz
        # Compute the chirp's instantaneous frequency at t.
        chirp_frequency_hz = chirp_slope * t

        # Expect output near zero when chirp frequency is above cutoff_hz.
        expected = ((chirp_frequency_hz < cutoff_hz)
                    * np.sin(np.pi * chirp_slope * t**2).astype(np.float32))
        # Skip samples in the transition between passband and stopband.
        mask = np.abs(chirp_frequency_hz - cutoff_hz) >= 0.3 * cutoff_hz

        np.testing.assert_allclose(
            output_samples[mask], expected[mask], atol=0.04, err_msg=message)

  def test_streaming_random_block_size(self):
    """Test Resampler streaming works by passing blocks of random sizes."""
    np.random.seed(0)
    input_samples = np.random.randn(500).astype(np.float32)
    max_block_size = 20

    for input_sample_rate_hz in RATES:
      for output_sample_rate_hz in RATES:
        options = {'input_sample_rate_hz': input_sample_rate_hz,
                   'output_sample_rate_hz': output_sample_rate_hz}
        message = make_message(options)
        resampler = dsp.Resampler(**options)

        # Do "streaming" resampling, passing successive blocks of input.
        streaming = []
        n = 0
        while n < len(input_samples):
          input_block_size = int(np.random.rand() * max_block_size)
          input_block = input_samples[n:n + input_block_size]
          n += input_block_size
          # Resample the block.
          output_block = resampler.process_samples(input_block)
          streaming.append(output_block)

        streaming = np.hstack(streaming)

        resampler.reset()
        # Do "nonstreaming" resampling, processing the whole input at once.
        nonstreaming = resampler.process_samples(input_samples)

        # Streaming vs. nonstreaming outputs should match.
        np.testing.assert_allclose(streaming, nonstreaming, atol=1e-6,
                                   err_msg=message)


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
    samples, sample_rate_hz = dsp.read_wav_file(self.read_filename)

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_from_bytes(self):
    """Read WAV from a byte string with read_wav_from_bytes()."""
    samples, sample_rate_hz = dsp.read_wav_from_bytes(self.wav_bytes)

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_read_wav_given_local_file_object(self):
    """Read WAV given a local file object."""
    with open(self.read_filename, 'rb') as f:
      samples, sample_rate_hz = dsp.read_wav_file(f)

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_read_wav_given_memory_stream(self):
    """Read WAV from an in-memory stream."""
    samples, sample_rate_hz = dsp.read_wav_file(io.BytesIO(self.wav_bytes))

    self.assertEqual(samples.dtype, np.int16)
    np.testing.assert_array_equal(samples, self.wav_samples)
    self.assertEqual(sample_rate_hz, 48000)

  def test_write_wav_local_file(self):
    """Write WAV to a given filename with write_wav_file()."""
    try:
      write_filename = os.path.join(self.temp_dir, 'write.wav')
      dsp.write_wav_file(write_filename, self.wav_samples, 44100)

      samples, sample_rate_hz = dsp.read_wav_file(write_filename)
      np.testing.assert_array_equal(samples, self.wav_samples)
      self.assertEqual(sample_rate_hz, 44100)
    finally:
      if os.path.isfile(write_filename):
        os.remove(write_filename)

  def test_to_bytes(self):
    """Write WAV to byte string with write_wav_to_bytes()."""
    wav_bytes = dsp.write_wav_to_bytes(self.wav_samples, 44100)

    samples, sample_rate_hz = dsp.read_wav_from_bytes(wav_bytes)
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

    samples, sample_rate_hz = dsp.read_wav_from_bytes(wav_bytes)

    self.assertEqual(samples.dtype, np.int32)
    np.testing.assert_array_equal(samples, expected)
    self.assertEqual(sample_rate_hz, 44100)

    # Read with conversion to float32.
    samples, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(
        samples, expected.astype(np.float32) / 2.0**31)

  def test_read_float_wav(self):
    """Read a 48kHz mono WAV file with 32-bit float samples."""
    np.random.seed(0)
    expected = np.random.randn(15, 4).astype(np.float32)
    wav_bytes = make_float_wav(expected, 48000)

    samples, sample_rate_hz = dsp.read_wav_from_bytes(wav_bytes)

    self.assertEqual(samples.dtype, np.float32)
    np.testing.assert_array_equal(samples, expected)
    self.assertEqual(sample_rate_hz, 48000)

  def test_read_16_bit_wav_with_dtype(self):
    """Test reading a 16-bit WAV with conversion to specified dtype."""
    samples = np.expand_dims(
        [0, 1, 2, -5, 25000, 32767, -32768], axis=1).astype(np.int16)
    wav_bytes = dsp.write_wav_to_bytes(samples, 8000)

    # int16 -> int16.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.int16)
    self.assert_equal_same_dtype(out, samples)
    # int16 -> int32.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.int32)
    self.assert_equal_same_dtype(out, samples.astype(np.int32) * 2**16)
    # int16 -> float32.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(out, samples.astype(np.float32) / 2.0**15)

  def test_read_24_bit_wav_with_dtype(self):
    """Test reading a 24-bit WAV with conversion to specified dtype."""
    samples = 256 * np.expand_dims(
        [1, -1500000, 2**23 - 1, -2**23], axis=1).astype(np.int32)
    wav_bytes = make_24_bit_wav(samples, 8000)

    # int32 -> int16.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.int16)
    self.assert_equal_same_dtype(
        out, np.expand_dims([0, -5859, 32767, -32768], axis=1).astype(np.int16))
    # int32 -> int32.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.int32)
    self.assert_equal_same_dtype(out, samples)
    # int32 -> float32.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(out, samples.astype(np.float32) / 2.0**31)

  def test_read_float_wav_with_dtype(self):
    """Test reading a float WAV with conversion to specified dtype."""
    samples = np.expand_dims(
        [0.0, 1e-6, -1e-4, 0.1, -0.5, 1.0, -1.0,
         np.inf, -np.inf, np.nan], axis=1).astype(np.float32)
    wav_bytes = make_float_wav(samples, 8000)

    # float32 -> int16.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.int16)
    self.assert_equal_same_dtype(
        out, np.expand_dims([0, 0, -3, 3277, -16384, 32767, -32768,
                             32767, -32768, 0], axis=1).astype(np.int16))
    # float32 -> int32.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.int32)
    self.assert_equal_same_dtype(
        out, np.expand_dims([
            0, 2147, -214748, 214748368, -1073741824, 2147483647,
            -2147483648, 2147483647, -2147483648, 0], axis=1).astype(np.int32))
    # float32 -> float32.
    out, _ = dsp.read_wav_from_bytes(wav_bytes, dtype=np.float32)
    self.assert_equal_same_dtype(out, samples)

  def test_write_wav_1d_array(self):
    """Test writing a 1D array as a mono WAV file."""
    samples = np.arange(20, dtype=np.int16)
    recovered, sample_rate_hz = dsp.read_wav_from_bytes(
        dsp.write_wav_to_bytes(samples, 8000))

    np.testing.assert_array_equal(recovered, samples.reshape(-1, 1))
    self.assertEqual(sample_rate_hz, 8000)

  def test_read_wav_bad_arg(self):
    """Call where the argument is not a file-like object."""

    class Nonsense(object):
      pass

    with self.assertRaisesRegex(TypeError, 'Nonsense found'):
      dsp.read_wav_file(Nonsense())

  def test_read_wav_read_not_callable(self):
    """Test where the read attribute is not callable."""
    reader = MockReader(None)
    with self.assertRaisesRegex(TypeError, 'not callable'):
      dsp.read_wav_file(reader)

  def test_read_wav_reader_raises_exception(self):
    """Test where the file object read method raises an exception."""
    def _failing_read(unused_size):
      raise OSError('read method failed')
    reader = MockReader(_failing_read)
    with self.assertRaisesRegex(OSError, 'read method failed'):
      dsp.read_wav_file(reader)

  def test_read_wav_reader_returns_wrong_type(self):
    """Test where the read method returns the wrong type."""
    reader = MockReader(lambda size: [0] * size)
    with self.assertRaisesRegex(TypeError, 'list found'):
      dsp.read_wav_file(reader)

  def test_read_wav_reader_result_too_large(self):
    """Test where the read method returns more than requested."""
    reader = MockReader(lambda size: b'\000' * (size + 1))
    with self.assertRaisesRegex(ValueError, 'exceeds requested size'):
      dsp.read_wav_file(reader)

  def test_read_wav_bad_dtype(self):
    """Test where WAV fact chunk is corrupt."""
    with self.assertRaisesRegex(ValueError, 'dtype must be one of'):
      dsp.read_wav_from_bytes(b'RIFF', dtype=np.uint8)

  def test_read_wav_bad_fact_chunk(self):
    """Test where WAV fact chunk is corrupt."""
    with self.assertRaisesRegex(OSError, 'error reading WAV header'):
      dsp.read_wav_from_bytes(b'RIFF_\000\000\000WAVEfactbusted')

  def test_write_wav_bad_arg(self):
    """write_wav_file where the argument is not a file-like object."""

    class Nonsense(object):
      pass

    with self.assertRaisesRegex(TypeError, 'Nonsense found'):
      dsp.write_wav_file(Nonsense(), np.zeros((10, 1), dtype=np.int16), 8000)

  def test_write_wav_wrong_dtype(self):
    """write_wav_file where samples can't safely cast to np.int16 dtype."""
    samples = np.array([-0.2, 0.5, 0.7, 0.3, 0.1])
    with self.assertRaisesRegex(TypeError, 'Cannot cast array data'):
      dsp.write_wav_to_bytes(samples, 8000)

  def test_write_wav_write_not_callable(self):
    """write_wav_file where the write attribute is not callable."""
    writer = MockWriter(None)
    with self.assertRaisesRegex(TypeError, 'not callable'):
      dsp.write_wav_file(writer, np.zeros((10, 1), dtype=np.int16), 8000)

  def test_write_wav_writer_raises_exception(self):
    """write_wav_file where the file object write method raises an exception."""
    def _failing_write(unused_bytes):
      raise OSError('write method failed')
    writer = MockWriter(_failing_write)
    with self.assertRaisesRegex(OSError, 'write method failed'):
      dsp.write_wav_file(writer, np.zeros((10, 1), dtype=np.int16), 8000)


if __name__ == '__main__':
  unittest.main()
