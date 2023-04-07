# Copyright 2020-2021, 2023 Google LLC
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
"""Python bindings for dsp."""

import fractions
import io
from typing import Any, IO, Iterable, Optional, Tuple, Union

import numpy as np

from extras.python import fast_fun_python_bindings
from extras.python import q_resampler_python_bindings
from extras.python import wav_io_python_bindings


def fast_log2(x):
  """Fast log base 2. Wraps `FastLog2()` in the C library.

  An approximation of log2() is accurate to about 0.003 max abs error.

  Limitations:
   - x is assumed to be positive and finite.
   - If x is a denormal, i.e. a small value less than about 1e-38, the result
     is less accurate.

  Args:
    x: Scalar or numpy array of np.float32 dtype.
  Returns:
    np.float32 or numpy array of the same shape.
  """
  return fast_fun_python_bindings.fast_log2_impl(x)


def fast_exp2(x):
  """Fast 2^x. Wraps `FastExp2()` in the C library.

  An approximation of exp2() accurate to about 0.3% relative error.

  Limitations:
    - Assumes |x| <= 126. (Otherwise, result may be nonsensical.)

  Args:
    x: Scalar or numpy array of np.float32 dtype.
  Returns:
    np.float32 or numpy array of the same shape.
  """
  return fast_fun_python_bindings.fast_exp2_impl(x)


def fast_pow(x, y):
  """Fast power x^y for x > 0. Reproduces `FastPow()` in the C library.

  An approximation of x^y with about 0.5% relative error.

  Limitations:
   - Assumes x is positive and finite.
   - Assumes 1e-37 <= |x^y| <= 1e37, i.e. that the exact result would be
     neither very large nor very close to zero.

  Otherwise, result may be nonsensical.

  Args:
    x: Scalar or numpy array of np.float32 dtype.
    y: Scalar or numpy array of np.float32 dtype.
  Returns:
    np.float32 or numpy array of the same shape.
  """
  # Reproduce the computation used in the C library:
  #   float FastPow(float x, float y) { return FastExp2(FastLog2(x) * y); }
  return fast_fun_python_bindings.fast_exp2_impl(
      fast_fun_python_bindings.fast_log2_impl(x) * np.asarray(y, np.float32))


def fast_tanh(x):
  """Fast tanh(x). Wraps `FastTanh()` in the C library.

  An approximation of tanh() accurate to about 0.0008 max abs error.
  The result is valid for non-NaN x, even for large x.

  Args:
    x: Scalar or numpy array of np.float32 dtype.
  Returns:
    np.float32 or numpy array of the same shape.
  """
  return fast_fun_python_bindings.fast_tanh_impl(x)


def fast_sigmoid(x):
  """Fast sigmoid (logistic) function. Wraps `FastSigmoid()` in the C library.

  An approximation of 1/(1+exp(-x)) accurate to about 0.0004 max abs error. The
  result is valid for non-NaN x, even for large x.

  Args:
    x: Scalar or numpy array of np.float32 dtype.
  Returns:
    np.float32 or numpy array of the same shape.
  """
  return fast_fun_python_bindings.fast_sigmoid_impl(x)


class Resampler:
  """Python bindings for QResampler."""

  def __init__(self,
               input_sample_rate_hz: float,
               output_sample_rate_hz: float,
               num_channels: int = 1,
               max_denominator: int = 1000,
               rational_approximation_max_terms: int = 47,
               rational_approximation_convergence_tolerance: float = 1e-9,
               filter_radius_factor: float = 5.0,
               cutoff_proportion: float = 0.9,
               kaiser_beta: float = 5.658,
               max_input_size: int = 1024):
    """Constructor, wraps `QResamplerMake()` in the C library.

    Args:
      input_sample_rate_hz: Float, input audio sample rate in Hz.
      output_sample_rate_hz: Float, output audio sample rate in Hz.
      num_channels: Integer, number of channels.
      max_denominator: Integer, determines the max allowed denominator, which is
        also the max number of filters.
      rational_approximation_max_terms: Integer, in approximating the resampling
        factor with a rational, the max number of continued fraction terms used.
      rational_approximation_convergence_tolerance: Float. Truncate continued
        fraction when residual is less than this tolerance.
      filter_radius_factor: Float, scales the nonzero support radius of the
        resampling kernel. Larger radius improves filtering quality but
        increases computation and memory cost.
      cutoff_proportion: Float, antialiasing cutoff frequency as a proportion of
        min(input_sample_rate_hz, output_sample_rate_hz) / 2. The default is
        0.9, meaning the cutoff is at 90% of the input Nyquist frequency or the
        output Nyquist frequency, whichever is smaller.
      kaiser_beta: Float, the positive beta parameter for the Kaiser window
        shape, where larger value yields a wider transition band and stronger
        attenuation. The default 5.658 gives a stopband of -60 dB.
      max_input_size: Integer, max input size for internal buffering. Input
        exceeding this size is passed over multiple calls to the C library.

    Raises:
      ValueError: if parameters are invalid. (In this case, the C library may
        write additional details to stderr.)
    """
    self._impl = q_resampler_python_bindings.ResamplerImpl(
        input_sample_rate_hz,
        output_sample_rate_hz,
        num_channels=num_channels,
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
      samples: 2D numpy array with np.float32 dtype of input samples with
        num_channels columns. The array may be 1D if num_channels = 1.

    Returns:
      2D numpy array of resampled output. Or if num_channels = 1 and `samples`
      is 1D, then the output is 1D.
    """
    return self._impl.process_samples(samples)

  @property
  def rational_factor(self) -> fractions.Fraction:
    """Rational used to approximate of the requested resampling factor."""
    return fractions.Fraction(*self._impl.rational_factor)

  @property
  def num_channels(self) -> int:
    """Number of channels."""
    return self._impl.num_channels

  @property
  def flush_frames(self) -> int:
    """Number of zero-valued input frames sufficient to flush the resampler."""
    return self._impl.flush_frames


class ResamplerKernel:
  """Python bindings for QResamplerKernel, Kaiser-windowed sinc."""

  def __init__(self,
               input_sample_rate_hz: float,
               output_sample_rate_hz: float,
               filter_radius_factor: float = 5.0,
               cutoff_proportion: float = 0.9,
               kaiser_beta: float = 5.658):
    """Constructor, wraps `QResamplerKernelInit()`.

    See Resampler.__init__ for details.

    Args:
      input_sample_rate_hz: Float, input audio sample rate in Hz.
      output_sample_rate_hz: Float, output audio sample rate in Hz.
      filter_radius_factor: Float, support radius scale factor.
      cutoff_proportion: Float, antialiasing cutoff frequency.
      kaiser_beta: Float, Kaiser window beta parameter.
    """
    self._impl = q_resampler_python_bindings.KernelImpl(
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


def read_wav_file(filename: Union[str, IO[bytes]],
                  dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, int]:
  """Reads a WAV audio file.

  This function uses dsp/read_wav_file_generic.c to read WAV files. Unlike
  `wave` and `scipy.io.wavfile`, this library can read WAVs with 24-bit, float,
  and mulaw encoded samples. It also tolerates WAVs with a truncated data chunk,
  which is a common problem that confuses other WAV software
  (scipy.io.wavfile raises an exception).

  The returned sample array has shape [num_frames, num_channels]. By default, it
  has dtype according to the encoding in the file:
   * np.int16 for 16-bit PCM or mulaw sample encoding.
   * np.int32 for 24-bit PCM encoding.
   * np.float32 for float encoding.

  If `dtype` is specified, the samples are converted to `dtype` according to the
  nominal range of the dtype. The supported dtypes and their nominal ranges are

    np.int16     [-2^15, 2^15)
    np.int32     [-2^31, 2^31)
    np.float32   [-1.0, 1.0)

  For instance int16 -> int32 conversion is casting then scaling by 2^16. In
  floating -> integer conversions, out-of-range values are saturated.

  Args:
    filename: String or open file-like object.
    dtype: Numpy dtype, or None to return samples without conversion.

  Returns:
    (samples, sample_rate_hz) 2-tuple.
  """
  if isinstance(filename, str):
    with open(filename, 'rb') as f:
      return wav_io_python_bindings.read_wav_impl(f, dtype)
  else:
    return wav_io_python_bindings.read_wav_impl(filename, dtype)


def write_wav_file(filename: Union[str, IO[bytes]], samples: Iterable[Any],
                   sample_rate_hz: int) -> None:
  """Writes a 16-bit WAV audio file.

  Args:
    filename: String or open writeable file-like object.
    samples: 2D array of shape [num_frames, num_channels] and np.int16 dtype. If
      samples is 1D, it is interpreted as a single channel.
    sample_rate_hz: Integer, sample rate in Hz.
  """
  sample_rate_hz = int(sample_rate_hz)

  if isinstance(filename, str):
    with open(filename, 'wb') as f:
      wav_io_python_bindings.write_wav_impl(f, samples, sample_rate_hz)
  else:
    wav_io_python_bindings.write_wav_impl(filename, samples, sample_rate_hz)


def read_wav_from_bytes(
    wav_bytes: bytes,
    dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, int]:
  """Reads a WAV from bytes.

  Args:
    wav_bytes: Bytes object.
    dtype: Numpy dtype, or None to return samples without conversion.

  Returns:
    (samples, sample_rate_hz) 2-tuple.
  """
  return wav_io_python_bindings.read_wav_impl(io.BytesIO(wav_bytes), dtype)


def write_wav_to_bytes(samples: Iterable[Any], sample_rate_hz: int) -> bytes:
  """Writes a 16-bit WAV to bytes.

  Args:
    samples: 2D array of shape [num_frames, num_channels] and np.int16 dtype. If
      samples is 1D, it is interpreted as a single channel.
    sample_rate_hz: Integer, sample rate in Hz.

  Returns:
    Bytes object.
  """
  sample_rate_hz = int(sample_rate_hz)

  f = io.BytesIO()
  wav_io_python_bindings.write_wav_impl(f, samples, sample_rate_hz)
  return f.getvalue()
