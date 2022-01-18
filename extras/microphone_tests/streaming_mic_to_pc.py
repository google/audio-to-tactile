# Copyright 2021 Google LLC
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
"""This scripts gets audio data from slim board over USB.

The hardware should be running mics_streaming_test.ino sketch
"""

import argparse
import numpy as np
import scipy.io.wavfile
import serial
from serial.tools import list_ports

# This rate is set in the hardware.
SAMPLE_RATE_HZ = 15625.0


def main():

  # Serial port access.
  # Print list of serial ports.
  ports = list(list_ports.comports())
  for i, port in enumerate(ports):
    print(f'[{i}] {port}')

  # parse the input arguments.
  parser = argparse.ArgumentParser(
      description='Record wav from the slim board microphone.')
  parser.add_argument('wavfile', type=str, help='Output WAV file path.')
  parser.add_argument(
      'recording_duration_sec',
      type=float,
      help=('Specify the length of recording in seconds'))
  parser.add_argument(
      'port_index',
      type=int,
      help=('Specify the USB port index, as listed in the table.'
            'The correct port should say - USB Serial'))
  args = parser.parse_args()

  # Open serial port. The port name will need to be changed in a different
  # computer.
  # In MacOS port will be /dev/tty.usbmodemXXXXXX
  port_string = str(ports[args.port_index])
  # Remove the extra text from the port name (e.g, "-Feather nRF52840 Express").
  port_string = port_string.split(' ', 1)[0]
  print('Connecting to port: ', port_string)
  # Add timeout to avoid infinite loop, while waiting for end of line.
  ser = serial.Serial(port_string, timeout=0.1)

  num_samples = int(round(args.recording_duration_sec * SAMPLE_RATE_HZ))

  print('Recording...')
  lines = b''.join(ser.readline() for _ in range(num_samples))
  print('Done.')

  ser.close()

  samples = np.fromstring(lines.decode('ascii'), np.int16, sep='\n')

  print('Raw microphone data snippet: ', samples)

  if len(samples) < num_samples:
    print('Warning: Data is incomplete. ' +
          f'Received {len(samples)} out of {num_samples} samples.')

  scipy.io.wavfile.write(args.wavfile, int(round(SAMPLE_RATE_HZ)), samples)

if __name__ == '__main__':
  main()
