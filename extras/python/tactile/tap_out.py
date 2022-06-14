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
import datetime
import enum
from typing import Callable, Dict, Iterable, List, Type, TypeVar, Tuple, Union

import numpy as np

# Tap out buffer capacity in bytes.
BUFFER_CAPACITY = 256
# Max number of dimensions in an output.
MAX_DIMS = 3
# Max number of simultaneous outputs.
MAX_OUTPUTS = 4
# Max name length before it is truncated in the binary format.
MAX_NAME_LENGTH = 15
# Number of bytes to encode a descriptor.
BYTES_PER_DESCRIPTOR = 1 + MAX_NAME_LENGTH + 1 + MAX_DIMS

# Marker byte to help detect and skip across extra bytes.
MARKER = b'\xfe'

# "Heartbeat" with empty payload to indicate that receiver is listening.
OP_HEARTBEAT = 0x01
# Request with empty payload to get the tap out descriptors.
OP_GET_DESCRIPTORS = 0x02
# Message containing the descriptors.
OP_DESCRIPTORS = 0x03
# Request to begin capture. Payload specifies which outputs.
OP_START_CAPTURE = 0x04
# Message containing captured tap out output.
OP_CAPTURE = 0x05

HEARTBEAT_BYTES = MARKER + bytes([OP_HEARTBEAT, 0])
BUFFERS_PER_HEARTBEAT = 100

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

  def __str__(self):
    return f'{self.name:<15} {self.dtype.name.lower():<8} {self.shape}'

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


def parse_descriptors(
    data: bytes) -> Tuple[Dict[int, Descriptor], datetime.date]:
  """Parse a set of Descriptors from bytes."""
  if not data:
    raise ValueError('Descriptors data is empty')
  build_date_u32 = int(np.frombuffer(data[:4], '<u4'))
  build_date = datetime.date(build_date_u32 // 10000,
                             (build_date_u32 // 100) % 100,
                             build_date_u32 % 100)

  num_descriptors = int(data[4])
  data = data[5:]
  expected_size = num_descriptors * BYTES_PER_DESCRIPTOR
  if expected_size > len(data):
    raise ValueError('Descriptors data size mismatch: ' +
                     f'{expected_size} vs. {len(data)}')

  descriptors = {}
  for i in range(num_descriptors):
    bytes_i = data[i * BYTES_PER_DESCRIPTOR:(i + 1) * BYTES_PER_DESCRIPTOR]
    descriptors[bytes_i[0]] = Descriptor.parse(bytes_i[1:])
  return descriptors, build_date


def deserialize_output(data: np.ndarray, descriptor: Descriptor) -> OutputData:
  """Deserializes one output."""
  if descriptor.dtype.is_numeric:
    # Reinterpert dtype and reshape according to the descriptor.
    output = np.frombuffer(data.tobytes('C'), descriptor.dtype.numpy_dtype)
    output = output.reshape((-1,) + descriptor.shape[1:])
  elif descriptor.dtype == DType.TEXT:
    output = [str(line.tobytes(), 'ascii').rstrip('\x00') for line in data]
  else:
    raise ValueError(f'Invalid dtype: {descriptor.dtype}')

  return output


def make_reader(
    selected: Iterable[Descriptor]) -> Callable[[bytes], Dict[str, OutputData]]:
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

  def reader(data: bytes) -> Dict[str, OutputData]:
    data = np.frombuffer(data, np.uint8).reshape(-1, bytes_per_buffer)
    return {
        descriptor.name: deserialize_output(data[:, slices[i]], descriptor)
        for i, descriptor in enumerate(descriptors)
    }

  return reader


class TapOut:
  """Object managing tap_out receiver communication."""

  def __init__(self, uart):
    self._uart = uart
    self._descriptors = {}  # type: Dict[int, Descriptor]

  def get_descriptors(self) -> Tuple[Dict[str, Descriptor], datetime.date]:
    """Sends Get Descriptors message to the device and returns the response."""
    self._write_message(OP_GET_DESCRIPTORS)

    op, payload = self._read_message()
    if op != OP_DESCRIPTORS:
      raise ValueError(f'Expected descriptors message, got op = {op}')
    self._descriptors, build_date = parse_descriptors(payload)

    return {v.name: v for v in self._descriptors.values()}, build_date

  def capture(self, selected: Iterable[str],
              num_buffers: int) -> Dict[str, OutputData]:
    """Captures `num_buffers` of data from the device.

    Args:
      selected: List of strings, the names of the outputs to capture.
      num_buffers: Int, the number of buffers to capture.

    Returns:
      Dictionary of the captured data.
    """
    tokens = [self._find_output_by_name(name) for name in selected]
    capture_reader = make_reader(self._descriptors[token] for token in tokens)

    self._write_message(OP_START_CAPTURE, bytes(tokens))

    captured_raw = []
    while num_buffers > 0:
      for _ in range(min(BUFFERS_PER_HEARTBEAT, num_buffers)):
        op, payload = self._read_message()
        if op != OP_CAPTURE:
          raise ValueError(f'Expected capture message, got op = {op}')
        captured_raw.append(payload)

      self._uart.write(HEARTBEAT_BYTES)
      num_buffers -= BUFFERS_PER_HEARTBEAT

    return capture_reader(b''.join(captured_raw))

  def _write_message(self, op: int, payload: bytes = b'') -> None:
    """Writes a tap_out message to UART serial connection."""
    if len(payload) > 255:
      raise ValueError('Payload length is too large to serialize.')
    self._uart.write(MARKER + bytes([op, len(payload)]) + payload)
    self._uart.flush()

  def _read_message(self) -> Tuple[int, bytes]:
    """Reads a tap_out message from UART serial connection."""

    # Read until finding the message marker byte.
    b = self._uart.read()
    if not b:  # Reading timed out.
      raise ValueError('No response from device.')
    elif b != MARKER:
      # If the byte is not the expected marker, read one byte at a time until we
      # find it. At most, `in_waiting + 1` iterations are made, scanning all
      # buffered input plus attempting one additional read.
      for _ in range(self._uart.in_waiting + 1):
        if b == MARKER:
          break
        b = self._uart.read()
      if b != MARKER:
        raise ValueError('Tap out response not found.')

    # Read tap_out message header (op, payload_size) bytes.
    header = self._uart.read(2)
    if len(header) != 2:
      raise ValueError('Timed out waiting for data.')
    op = int(header[0])
    payload_size = int(header[1])

    # Read message payload bytes.
    payload = self._uart.read(payload_size)
    if len(payload) != payload_size:
      raise ValueError('Timed out waiting for data.')

    return op, payload

  def _find_output_by_name(self, name: str) -> int:
    """Finds the token corresponding to a given output name."""
    for k, v in self._descriptors.items():
      if v.name == name:
        return k
    raise KeyError(f'Invalid output: "{name}"')
