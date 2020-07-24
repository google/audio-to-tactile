# Lint as: python2,python3
"""Resample audio by a rational factor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fractions
from typing import Iterable, Union   # pylint:disable=unused-import
import numpy as np  # pylint:disable=unused-import

import audio.dsp.portable.python.rational_factor_resampler_python_bindings as python_bindings


class Resampler(object):
  """Python bindings for RationalFactorResampler."""

  def __init__(self,
               input_sample_rate_hz,
               output_sample_rate_hz,
               max_denominator=1000,
               rational_approximation_max_terms=47,
               rational_approximation_convergence_tolerance=1e-9,
               filter_radius_factor=5.0,
               cutoff_proportion=0.9,
               kaiser_beta=5.658,
               max_input_size=1024):
    # type: (float, float, int, int, float, float, float, float, int) -> None
    """Constructor, wraps `RationalFactorResamplerMake()` in the C library.

    Args:
      input_sample_rate_hz: Float, input audio sample rate in Hz.
      output_sample_rate_hz: Float, output audio sample rate in Hz.
      max_denominator: Integer, determines the max allowed denominator, which
        is also the max number of filters.
      rational_approximation_max_terms: Integer, in approximating the resampling
        factor with a rational, the max number of continued fraction terms used.
      rational_approximation_convergence_tolerance: Float. Truncate continued
        fraction when residual is less than this tolerance.
      filter_radius_factor: Float, scales the nonzero support radius of the
        resampling kernel. Larger radius improves filtering quality but
        increases computation and memory cost.
      cutoff_proportion: Float, antialiasing cutoff frequency as a proportion
        of min(input_sample_rate_hz, output_sample_rate_hz) / 2. The default
        is 0.9, meaning the cutoff is at 90% of the input Nyquist frequency or
        the output Nyquist frequency, whichever is smaller.
      kaiser_beta: Float, the positive beta parameter for the Kaiser window
        shape, where larger value yields a wider transition band and stronger
        attenuation. The default 5.658 gives a stopband of -60 dB.
      max_input_size: Integer, max input size for internal buffering. Input
        exceeding this size is passed over multiple calls to the C library.
    Raises:
      ValueError: if parameters are invalid. (In this case, the C library may
        write additional details to stderr.)
    """
    self._impl = python_bindings.ResamplerImpl(
        input_sample_rate_hz,
        output_sample_rate_hz,
        max_denominator=max_denominator,
        rational_approximation_max_terms=rational_approximation_max_terms,
        rational_approximation_convergence_tolerance=
        rational_approximation_convergence_tolerance,
        filter_radius_factor=filter_radius_factor,
        cutoff_proportion=cutoff_proportion,
        kaiser_beta=kaiser_beta,
        max_input_size=max_input_size)

  def reset(self):
    # type: () -> None
    """Resets to initial state."""
    self._impl.reset()

  def process_samples(self, samples):
    # type: (Iterable) -> np.ndarray
    """Processes samples in a streaming manner.

    Args:
      samples: 1-D numpy array with np.float32 dtype, input samples.
    Returns:
      1-D numpy array of resampled output samples.
    """
    return self._impl.process_samples(samples)

  @property
  def rational_factor(self):
    # type: () -> fractions.Fraction
    """Rational used to approximate of the requested resampling factor."""
    return fractions.Fraction(*self._impl.rational_factor)

  @property
  def flush_size(self):
    # type: () -> int
    """Number of zero-valued input samples sufficient to flush the resampler."""
    return self._impl.flush_size


class Kernel(object):
  """Python bindings for RationalFactorResamplerKernel, Kaiser-windowed sinc."""

  def __init__(self,
               input_sample_rate_hz,
               output_sample_rate_hz,
               filter_radius_factor=5.0,
               cutoff_proportion=0.9,
               kaiser_beta=5.658):
    # type: (float, float, float, float, float) -> None
    """Constructor, wraps `RationalFactorResamplerKernelInit()`.

    See Resampler.__init__ for details.

    Args:
      input_sample_rate_hz: Float, input audio sample rate in Hz.
      output_sample_rate_hz: Float, output audio sample rate in Hz.
      filter_radius_factor: Float, support radius scale factor.
      cutoff_proportion: Float, antialiasing cutoff frequency.
      kaiser_beta: Float, Kaiser window beta parameter.
    """
    self._impl = python_bindings.KernelImpl(
        input_sample_rate_hz,
        output_sample_rate_hz,
        filter_radius_factor=filter_radius_factor,
        cutoff_proportion=cutoff_proportion,
        kaiser_beta=kaiser_beta)

  def __call__(self, x):
    # type: (Union[float, Iterable]) -> np.ndarray
    """Evaluate the kernel at x.

    Args:
      x: Scalar or numpy array.
    Returns:
      Numpy array of the same shape, kernel evaluated at x.
    """
    return self._impl(x)

  @property
  def factor(self):
    # type: () -> float
    """Requested resampling factor."""
    return self._impl.factor

  @property
  def radius(self):
    # type: () -> float
    """Kernel radius in units of input samples."""
    return self._impl.radius

  @property
  def radians_per_sample(self):
    # type: () -> float
    """Frequency of kernel sinc in radians per input sample."""
    return self._impl.radians_per_sample

