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

"""Resample audio by a rational factor."""

import fractions
from typing import Any, Iterable, Union
import numpy as np

import dsp.python.rational_factor_resampler_python_bindings as python_bindings


class Resampler:
  """Python bindings for RationalFactorResampler."""

  def __init__(self,
               input_sample_rate_hz: float,
               output_sample_rate_hz: float,
               max_denominator: int = 1000,
               rational_approximation_max_terms: int = 47,
               rational_approximation_convergence_tolerance: float = 1e-9,
               filter_radius_factor: float = 5.0,
               cutoff_proportion: float = 0.9,
               kaiser_beta: float = 5.658,
               max_input_size: int = 1024):
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

  def reset(self) -> None:
    """Resets to initial state."""
    self._impl.reset()

  def process_samples(self, samples: np.ndarray) -> np.ndarray:
    """Processes samples in a streaming manner.

    Args:
      samples: 1-D numpy array with np.float32 dtype, input samples.
    Returns:
      1-D numpy array of resampled output samples.
    """
    return self._impl.process_samples(samples)

  @property
  def rational_factor(self) -> fractions.Fraction:
    """Rational used to approximate of the requested resampling factor."""
    return fractions.Fraction(*self._impl.rational_factor)

  @property
  def flush_size(self) -> int:
    """Number of zero-valued input samples sufficient to flush the resampler."""
    return self._impl.flush_size


class Kernel:
  """Python bindings for RationalFactorResamplerKernel, Kaiser-windowed sinc."""

  def __init__(self,
               input_sample_rate_hz: float,
               output_sample_rate_hz: float,
               filter_radius_factor: float = 5.0,
               cutoff_proportion: float = 0.9,
               kaiser_beta: float = 5.658):
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

  def __call__(self, x: Union[float, Iterable[Any]]) -> np.ndarray:
    """Evaluate the kernel at x.

    Args:
      x: Scalar or numpy array.
    Returns:
      Numpy array of the same shape, kernel evaluated at x.
    """
    return self._impl(x)

  @property
  def factor(self) -> float:
    """Requested resampling factor."""
    return self._impl.factor

  @property
  def radius(self) -> float:
    """Kernel radius in units of input samples."""
    return self._impl.radius

  @property
  def radians_per_sample(self) -> float:
    """Frequency of kernel sinc in radians per input sample."""
    return self._impl.radians_per_sample

