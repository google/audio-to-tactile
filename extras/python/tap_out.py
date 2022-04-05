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

"""A library to "tap out" intermediate outputs for analysis and debugging.

It is useful when debugging to capture intermediate signals, for instance
the microphone input or Enveloper noise estimates. This Python library receives
the data encoded by the C library src/dsp/tap_out.h. The intended use is:

1. The device first sends a buffer of descriptor metadata, which is then
   parsed here using `parse_descriptors()`. A Python program could then show
   this information in a UI to let the user select among the available outputs.

2. The Python program communicates back to the device which outputs to capture.
   In preparation to receive the data, the Python program calls `make_reader()`,
   which makes a reader function to deserializes the selected outputs.

3. The device begins to send back buffers of output. The data is deserialized
   here using the reader function.
"""

import dataclasses
import enum
from typing import Callable, Dict, List, Sequence, Type, TypeVar, Tuple, Union

import numpy as np

# Tap out buffer capacity.
BUFFER_CAPACITY = 256
# Max number of dimensions in an output.
MAX_DIMS = 3
# Max number of simultaneous outputs.
MAX_OUTPUTS = 4
# Max name length before it is truncated in the binary format.
MAX_NAME_LENGTH = 15
# Number of bytes to encode a descriptor.
BYTES_PER_DESCRIPTOR = 1 + MAX_NAME_LENGTH + 1 + MAX_DIMS

OutputData = Union[np.ndarray, List[str]]


class DType(enum.Enum):
  """Supported dtypes."""

  UINT8 = 1
  INT8 = 2
  UINT16 = 3
  INT16 = 4
  UINT32 = 5
  INT32 = 6
  UINT64 = 7
  INT64 = 8
  FLOAT = 9
  DOUBLE = 10
  TEXT = 11

  @property
  def is_numeric(self) -> bool:
    return 1 <= int(self.value) <= 10

  @property
  def numpy_dtype(self) -> np.dtype:
    return np.dtype(('<u1', '<i1', '<u2', '<i2', '<u4', '<i4', '<u8', '<i8',
                     '<f4', '<f8')[self.value - 1])

  @property
  def itemsize(self) -> int:
    if self.is_numeric:
      return self.numpy_dtype.itemsize
    else:
      return 1


T = TypeVar('T', bound='Descriptor')


@dataclasses.dataclass
class Descriptor:
  """Descriptor metadata for one tap-out output."""

  name: str
  dtype: DType
  shape: Tuple[int]

  @classmethod
  def parse(cls: Type[T], data: bytes) -> T:
    """Parse a Descriptor from bytes."""
    expected_len = MAX_NAME_LENGTH + 1 + MAX_DIMS
    if len(data) != expected_len:
      raise ValueError(f'Expected {expected_len} bytes, got {len(data)}')

    name = str(data[:MAX_NAME_LENGTH], 'ascii').rstrip('\x00')
    dtype = DType(data[MAX_NAME_LENGTH])

    shape = []
    for i in range(MAX_DIMS):
      dim = data[MAX_NAME_LENGTH + 1 + i]
      if dim == 0:
        break
      shape.append(dim)

    if not shape:
      raise ValueError(f'{name}: Empty shape')

    return cls(name=name, dtype=dtype, shape=tuple(shape))

  @property
  def num_bytes(self) -> int:
    """Gets the number of bytes described by the dtype and shape."""
    return self.dtype.itemsize * np.prod(self.shape)


def parse_descriptors(data: bytes) -> Dict[int, Descriptor]:
  """Parse a set of Descriptors from bytes."""
  num_descriptors = len(data) // BYTES_PER_DESCRIPTOR
  descriptors = {}
  for i in range(num_descriptors):
    bytes_i = data[i * BYTES_PER_DESCRIPTOR:(i + 1) * BYTES_PER_DESCRIPTOR]
    descriptors[bytes_i[0]] = Descriptor.parse(bytes_i[1:])
  return descriptors


def deserialize_output(data: np.ndarray, descriptor: Descriptor) -> OutputData:
  """Deserializes one output."""
  if descriptor.dtype.is_numeric:
    # Reinterpert dtype and reshape according to the descriptor.
    output = np.frombuffer(data.tobytes('C'), descriptor.dtype.numpy_dtype)
    output = output.reshape((-1,) + descriptor.shape[1:])
  elif descriptor.dtype.name == 'TEXT':
    output = [str(line.tobytes(), 'ascii').rstrip('\x00') for line in data]
  else:
    raise ValueError(f'Invalid dtype: {descriptor.dtype}')

  return output


def make_reader(
    selected: Sequence[Descriptor]) -> Callable[[bytes], Tuple[OutputData]]:
  """Creates a reader function for a selected list of outputs."""
  descriptors = tuple(selected)
  slices = []
  start = 0

  for descriptor in descriptors:
    num_bytes = descriptor.num_bytes
    stop = start + num_bytes
    slices.append(slice(start, stop))
    start = stop

  bytes_per_buffer = start

  def reader(data: bytes) -> Tuple[OutputData]:
    data = np.frombuffer(data, np.uint8).reshape(-1, bytes_per_buffer)
    return tuple(deserialize_output(data[:, slices[i]], descriptor)
                 for i, descriptor in enumerate(descriptors))

  return reader
