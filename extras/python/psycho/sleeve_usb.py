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

"""Code to talk to the sleeve using a serial protocol."""

import numpy as np

import serial
import serial.tools.list_ports

# Tactor layout on the sleeve:
#                    (4)-(1)
#                    /     \      (2)
#            (3)   (8) (6) (5)           Control box
#                    \     /      (9)
#                    (7)-(10)


class SleeveUSB(object):
  """Allows us to talk to the tactile sleeve using a serial protocol."""

  # Constants needed for the hardware.
  TACTILE_FRAMES_IN_ONE_PACKET = 1
  BYTES_IN_HEADER = 4
  PWM_VALUES_IN_TACTILE_FRAME = 8
  TACTILE_CHANNELS = 12

  SAMPLE_RATE = 2000

  def __init__(self, debug=False):
    self._debug = debug
    self._modem = None

  def find_usb_modem(self) -> str:
    """Searches through USB devices looking for serial port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
      if 'usbmodem' in str(p):
        dev, _ = str(p).split(' ', 1)
        self._modem = dev
        return
    raise ValueError('Could not find modem in this list of ports: %s' % ports)

  def open_serial_port(self) -> serial.Serial:
    """Open serial port.

    The port name will need to be changed in a different computer.
    In MacOS port will be /dev/tty.usbmodemXXXXXX

    Returns:
        A serial port object
    """
    if not self._modem:
      self.find_usb_modem()
    print('Connecting to port: ', self._modem)
    return serial.Serial(self._modem)

  def wait_for_reception(self, ser: serial.Serial):
    while True:
      read_raw = ser.readline()
      read_out = read_raw.decode('ascii')
      read_out = read_out.replace('\n', '').replace('\r', '')
      if self._debug:
        print('received <' + read_out + '>')
      if 'buffer_copied' in read_out:
        return True
    return False

  def send_waves_to_tactors(self, waveforms: np.ndarray):
    """Sends wave (np array) to tactors. Data must be limited to +/-1."""
    assert waveforms.shape[1] == SleeveUSB.TACTILE_CHANNELS, (
        'Waveforms does not have %d channels: %s' %
        (self.TACTILE_CHANNELS, waveforms.shape))
    # So we stop all the tactors at the end
    waveforms[-SleeveUSB.PWM_VALUES_IN_TACTILE_FRAME:, :] = 0.0

    with self.open_serial_port() as ser:
      for frame_start in range(0, waveforms.shape[0],
                               SleeveUSB.PWM_VALUES_IN_TACTILE_FRAME):
        num_frames = min(SleeveUSB.PWM_VALUES_IN_TACTILE_FRAME,
                         waveforms.shape[0] - frame_start)
        # A byte array to send to the sleeve.
        tactile_frame_array = bytearray(SleeveUSB.TACTILE_CHANNELS*num_frames +
                                        SleeveUSB.BYTES_IN_HEADER)
        # Set the bytes for the header. Defined by serial protocol in
        # serial_puck_sleeve.h
        tactile_frame_array[0] = 200  # Start packet
        tactile_frame_array[1] = 201  # Start packet
        tactile_frame_array[2] = 17  # Playback code
        tactile_frame_array[3] = 128  # number of samples
        for i in range(num_frames):
          for j in range(SleeveUSB.TACTILE_CHANNELS):
            w = waveforms[frame_start + i, j] * 128 + 128
            w = int(min(max(w, 0), 255))
            tactile_frame_array[SleeveUSB.BYTES_IN_HEADER +
                                i * SleeveUSB.TACTILE_CHANNELS + j] = w
        try:
          ser.write(tactile_frame_array)
        except serial.SerialException:
          print('error sending')
        ser.flush()

        if False and frame_start:
          ser.reset_input_buffer()
          ser.reset_output_buffer()

        if not self.wait_for_reception(ser):
          print('Got a reception error. Aborting.')
          break
