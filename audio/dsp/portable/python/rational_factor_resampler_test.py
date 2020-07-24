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


Tests for RationalFactorResampler Python bindings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fractions
import unittest

import numpy as np

from audio.dsp.portable.python import rational_factor_resampler

# Tested sample rates in Hz.
RATES = (12000, 16000, 32000, 44100, 48000, 16000 * np.sqrt(2))


def make_message(options):
  """Returns formatted string describing `options` dict."""
  return 'Options: ' + ', '.join('%s=%s' % (k, options[k]) for k in options)


class KernelTest(unittest.TestCase):

  def test_kernel(self):
    """Test Kernel object for various sample rates and support radii."""
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
          kernel = rational_factor_resampler.Kernel(**options)
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
          expected_kernel = (
              theta * np.sinc(theta * x)
              * np.i0(kaiser_beta * np.sqrt(1 - (x / kernel.radius)**2))
              / np.i0(kaiser_beta))

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
      input_samples: 1D numpy array.
    Returns:
      1D numpy array, resampled output.
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
      output.append(((n >= 0) * input_samples[n]).dot(kernel(n0 - n)))
      m += 1

    return np.array(output)

  def test_compare_with_reference_resampler(self):
    """Compare Resampler to _reference_resampling() implementation."""
    np.random.seed(0)
    input_samples = -0.5 + np.random.rand(50)

    for filter_radius_factor in (4.0, 5.0, 17.0):
      for input_sample_rate_hz in RATES:
        for output_sample_rate_hz in RATES:
          options = {'input_sample_rate_hz': input_sample_rate_hz,
                     'output_sample_rate_hz': output_sample_rate_hz,
                     'filter_radius_factor': filter_radius_factor}
          message = make_message(options)
          resampler = rational_factor_resampler.Resampler(**options)

          output = resampler.process_samples(input_samples)

          kernel = rational_factor_resampler.Kernel(**options)
          self.assertAlmostEqual(float(resampler.rational_factor),
                                 kernel.factor, delta=5e-4, msg=message)
          self.assertEqual(resampler.flush_size, 2 * kernel.radius, msg=message)

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
    resampler = rational_factor_resampler.Resampler(np.pi, 1.0)
    self.assertEqual(resampler.rational_factor, fractions.Fraction(355, 113))

    # Truncate continued fraction expansion at 3 terms.
    resampler = rational_factor_resampler.Resampler(
        np.pi, 1.0, rational_approximation_max_terms=3)
    self.assertEqual(resampler.rational_factor,
                     fractions.Fraction(333, 106))  # 3rd convergent [3; 7, 15].

    # Truncate when continued fraction residual is less than 0.1.
    resampler = rational_factor_resampler.Resampler(
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
        resampler = rational_factor_resampler.Resampler(**options)
        # Run resampler on sine wave samples.
        output_samples = resampler.process_samples(input_samples)

        kernel = rational_factor_resampler.Kernel(
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
        resampler = rational_factor_resampler.Resampler(**options)
        # Run resampler on the chirp.
        output_samples = resampler.process_samples(input_samples)

        kernel = rational_factor_resampler.Kernel(**options)
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
        resampler = rational_factor_resampler.Resampler(**options)

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


if __name__ == '__main__':
  unittest.main()

