# Copyright 2020, 2021 Google LLC
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
"""This is a script to send wav data to tactors.

The wav file should be: 2000 Hz, signed 16-bit, PCM
"""

import argparse
import struct
import wave
# TODO Use custon wav parser in third_party.

import serial
from serial import SerialException
import serial.tools.list_ports

# Constants.
TACTILE_FRAMES_IN_ONE_PACKET = 1
BYTES_IN_HEADER = 4
PWM_VALUES_IN_TACTILE_FRAME = 8
TACTILE_CHANNELS = 12


def convert_16bit_pcm_to_byte(old_value: int) -> int:
  """Converts PCM value into a byte."""

  old_min = -32768  # original 16-bit PCM wave audio minimum.
  old_max = 32768  # original 16-bit PCM wave audio max.
  new_min = 0  # new 8-bit PCM wave audio min.
  new_max = 255  # new 8-bit PCM wave audio max.
  byte_value = int((((old_value - old_min) * (new_max - new_min)) /
                    (old_max - old_min)) + new_min)
  return byte_value


def get_next_audio_frames_in_byte_array(opened_wave_file) -> bytearray:
  """Helper function to extract data from audio frames."""

  # A byte array to send to the sleeve.
  tactile_frame_array = bytearray(TACTILE_CHANNELS *
                                  PWM_VALUES_IN_TACTILE_FRAME + BYTES_IN_HEADER)
  for i in range(PWM_VALUES_IN_TACTILE_FRAME * TACTILE_FRAMES_IN_ONE_PACKET):

    # Read one frame as a string (frame contains n channels)
    string = opened_wave_file.readframes(1)
    # Unpack values in the frame from little-endian signed-16 integers.
    # Nice explanation on formatting wave:
    # https://www.cameronmacleod.com/blog/reading-wave-python

    audio_frame = struct.unpack('<hhhhhhhhhhhh', string)

    # Convert to bytes and add to byte array.
    for x in range(TACTILE_CHANNELS):
      new_value = int(convert_16bit_pcm_to_byte(audio_frame[x]))
      tactile_frame_array[BYTES_IN_HEADER + x + i * 12] = new_value

  # Set the bytes for the header. Defined by serial protocol in
  # serial_puck_sleeve.h
  tactile_frame_array[0] = 200
  tactile_frame_array[1] = 201
  tactile_frame_array[2] = 17
  tactile_frame_array[3] = 128

  return tactile_frame_array


def main():

  # parse the input arguments
  parser = argparse.ArgumentParser(description='Play wav file on the tactors.')
  parser.add_argument('wavfile', type=str, help='Input WAV file filepath')
  parser.add_argument(
      'port_index',
      type=int,
      help=('Specify the USB port index, as listed in the table.'
            'The correct port should say - USB Serial'))
  parser.add_argument(
      '--trigger', help='Wait for the audio trigger', action='store_true')
  args = parser.parse_args()
  # TODO Add trigger hardware and functionality for video sync.

  # open the wav file.
  wav = wave.open(args.wavfile, 'r')

  (nchannels, sampwidth, framerate, nframes, comptype,
   compname) = wav.getparams()
  # print the wav file details
  print('Channels: ', nchannels)
  print('Sample width: ', sampwidth)
  print('Framerate: ', framerate)
  print('Number frames: ', nframes)
  print('Compression type: ', comptype)
  print('Compression name: ', compname)

  # Serial port access.
  # Print list of serial ports.
  ports = list(serial.tools.list_ports.comports())
  index_i = 0
  for p in ports:
    print('[', index_i, ']', p)
    index_i = index_i + 1

  # Open serial port. The port name will need to be changed in a different
  # computer.
  # In MacOS port will be /dev/tty.usbmodemXXXXXX
  port_string = str(ports[args.port_index])
  terminator = port_string.index(' ')
  port_string = port_string[:terminator]
  print('Connecting to port: ', port_string)
  # Add timeout to avoid infinite loop, while waiting for end of line.
  ser = serial.Serial(port_string, timeout=0.1)

  # Counter to keep track of how many frames already played.
  frame_playback_counter = 0

  print('----Start sending to sleeve----')
  # Send the first frame to kickstart the synchronization.
  ser.write(get_next_audio_frames_in_byte_array(wav))

  retry_index = 0

  # Loop for checking if there is a request for new frames.
  while True:
    retry_index = retry_index + 1
    read_raw = ser.readline()
    read_out = read_raw.decode('ascii')
    read_out = read_out.replace('\n', '')
    read_out = read_out.replace('\r', '')
    print('received<' + read_out + '>')
    # If the serial port times out waiting for a endline, force asking for a
    # packet to start transmission again.
    # Serial port is not always 100% reliable, and has occasional errors on
    # physical layer.
    if retry_index > 2:
      ser.write(get_next_audio_frames_in_byte_array(wav))
      frame_playback_counter = (
          frame_playback_counter + PWM_VALUES_IN_TACTILE_FRAME)
    if read_out == 'buffer_copied':

      frame_playback_counter = (
          frame_playback_counter + PWM_VALUES_IN_TACTILE_FRAME)

      print(f'frames played: {frame_playback_counter} / {nframes}')

      try:
        ser.write(get_next_audio_frames_in_byte_array(wav))
        retry_index = 0
      except SerialException:
        print('error sending')

      ser.flush()
      ser.reset_input_buffer()
      ser.reset_output_buffer()

      # Quit if all frames are sent.
      if frame_playback_counter >= (nframes - 2 * PWM_VALUES_IN_TACTILE_FRAME):
        print('----Sent all frames. quit!----')
        ser.close()
        break


if __name__ == '__main__':
  main()
