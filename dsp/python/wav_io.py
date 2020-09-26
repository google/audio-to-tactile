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

"""Read and write WAV audio files.

This library uses the read_wav_file_generic.c and write_wav_file_generic.c for
WAV reading and writing.

Unlike `wave` and `scipy.io.wavfile`, this library can read WAVs with 24-bit,
float, and mulaw encoded samples. It also tolerates WAVs with a truncated data
chunk, which is a common problem that confuses other WAV software
(scipy.io.wavfile raises an exception).
"""

import io
from typing import Any, IO, Iterable, Optional, Tuple, Union

import numpy as np

from dsp.python import wav_io_python_bindings


def read_wav_file(filename: Union[str, IO[bytes]],
                  dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, int]:
  """Reads a WAV audio file.

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


def write_wav_file(filename: Union[str, IO[bytes]],
                   samples: Iterable[Any],
                   sample_rate_hz: int) -> None:
  """Writes a 16-bit WAV audio file.

  Args:
    filename: String or open writeable file-like object.
    samples: 2D array of shape [num_frames, num_channels] and np.int16 dtype.
      If samples is 1D, it is interpreted as a single channel.
    sample_rate_hz: Integer, sample rate in Hz.
  """
  sample_rate_hz = int(sample_rate_hz)

  if isinstance(filename, str):
    with open(filename, 'wb') as f:
      wav_io_python_bindings.write_wav_impl(f, samples, sample_rate_hz)
  else:
    wav_io_python_bindings.write_wav_impl(filename, samples, sample_rate_hz)


def from_bytes(wav_bytes: bytes,
               dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, int]:
  """Reads a WAV from bytes.

  Args:
    wav_bytes: Bytes object.
    dtype: Numpy dtype, or None to return samples without conversion.
  Returns:
    (samples, sample_rate_hz) 2-tuple.
  """
  return wav_io_python_bindings.read_wav_impl(io.BytesIO(wav_bytes), dtype)


def to_bytes(samples: Iterable[Any], sample_rate_hz: int) -> bytes:
  """Writes a 16-bit WAV to bytes.

  Args:
    samples: 2D array of shape [num_frames, num_channels] and np.int16 dtype.
      If samples is 1D, it is interpreted as a single channel.
    sample_rate_hz: Integer, sample rate in Hz.
  Returns:
    Bytes object.
  """
  sample_rate_hz = int(sample_rate_hz)

  f = io.BytesIO()
  wav_io_python_bindings.write_wav_impl(f, samples, sample_rate_hz)
  return f.getvalue()
