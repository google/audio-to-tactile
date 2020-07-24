# Lint as: python3
r"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Reads the NIST SPHERE audio file format used by the TIMIT dataset.

NOTE: SoX [http://sox.sourceforge.net] can also read NIST SPHERE format.

Reference:
https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p04.html
"""

import typing
from typing import Any, Text, Tuple  # pylint:disable=unused-import

import numpy as np


def read_nist_sphere(sphere_file):
  # type: (Text) -> Tuple[int, np.ndarray]
  """Reads NIST SPHERE file.

  NOTE: Only mono 16-bit PCM data is supported.

  Args:
    sphere_file: String or file object.
  Returns:
    (sample_rate_hz, samples) where `samples` is a 1D numpy array.
  """
  if hasattr(sphere_file, 'read'):
    return _read_file(sphere_file)
  else:
    with open(sphere_file, 'rb') as f:
      return _read_file(f)


def _read_header_line(f):
  # type: (typing.BinaryIO) -> Text
  """Reads characters up to the next null or newline."""
  line = b''
  while True:
    c = f.read(1)
    if c in b'\0\n':
      break
    line += c
  return line.decode('utf-8')


def _read_file(f):
  # type: (typing.BinaryIO) -> Tuple[int, np.ndarray]
  """Reads the sphere file."""
  magic = _read_header_line(f)
  if magic.upper() != 'NIST_1A':
    raise ValueError('Not a NIST Sphere file')

  try:
    header_size = int(_read_header_line(f))
  except ValueError:
    raise ValueError('Error reading Sphere header')
  if header_size < 16:
    raise ValueError('Error reading Sphere header')
  fields = {
      'channel_count': 0,
      'sample_byte_format': '01',
      'sample_coding': 'pcm',
      'sample_count': 0,
      'sample_n_bytes': 0,
      'sample_rate': 0,
  }

  # Parse NIST header fields.
  for line in f.read(header_size - 16).decode('utf-8').split('\n'):
    if line == 'end_head':
      break
    try:
      field_name, field_type, value = line.split()
      if field_name not in fields:
        continue
      elif field_type == '-i':
        value = int(value)
    except ValueError:
      raise ValueError('Error reading Sphere header')
    if not isinstance(value, type(fields[field_name])):
      raise ValueError('Error reading Sphere header')
    fields[field_name] = value

  if (fields['channel_count'] != 1 or
      fields['sample_coding'] != 'pcm' or
      fields['sample_n_bytes'] != 2):
    raise ValueError('Only mono 16-bit PCM data supported')

  num_samples = fields['sample_count']
  little_endian = fields['sample_byte_format'] == '01'

  # Read waveform data.
  samples = np.frombuffer(f.read(num_samples * fields['sample_n_bytes']),
                          dtype='<i2' if little_endian else '>i2')
  return int(fields['sample_rate']), samples
