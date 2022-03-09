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

"""Script to flash new firmware to the tactile device.

Use: python3 flash_firmware.py [port] firmware.ino.zip

This script uses the nordicsemi.dfu Python library to flash a new ino.zip
firmware binary to a connected tactile device.

Instructions:

1. Install adafruit-nrfutil with `pip3 install --user adafruit-nrfutil` or for
   other options see the installation instructions at
   https://github.com/adafruit/Adafruit_nRF52_nrfutil#installation

2. Connect the tactile device to the computer with a USB cable.

3. Turn on the device. We assume the device has a USB bootloader (Adafruit
   Feather variant), but this might not always be the case.

4. Run the script like

   $ python3 flash_firmware.py firmware.ino.zip

   The script auto-detects the port where the tactile device is connected. It
   looks for a device with "nrf52" in its name, otherwise it attempts to
   flash to the first available port. If this port selection fails for some
   reason, you can specify the port by running the script like

   $ python3 flash_firmware.py /dev/tty1 firmware.ino.zip

   The script prints "Device programmed" when firmware flashed successfully.
"""

import logging
import os.path
import sys
from typing import Any, List, Sequence

from nordicsemi.dfu.dfu import Dfu
from nordicsemi.dfu.dfu_transport import DfuEvent
from nordicsemi.dfu.dfu_transport_serial import DfuTransportSerial
import serial.tools.list_ports


def list_ports() -> List[Any]:
  """Get a list of available ports."""
  available_ports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
  if not available_ports:
    print('No available ports found.\n')
    print('Please make sure the device is connected and turned on.')
    sys.exit(1)

  print('Available ports:')
  for port in available_ports:
    print('%-20s %s' % port[:2])
  print('')
  return available_ports


def select_port(available_ports: List[Any]) -> str:
  """Simple heuristic to select an available port."""
  selected_port = available_ports[0][0]  # Default to selecting first port.
  for port in available_ports:
    # If there is an nRF52 device, pick that instead.
    if 'nrf52' in port[1].lower():
      selected_port = port[0]
      break

  return selected_port


def main(argv: Sequence[str]) -> int:
  # Set verbose logging level.
  logging.basicConfig(format='%(message)s', level=logging.INFO)

  if len(argv) == 2:  # Script called with only the package arg.
    _, package = argv
    port = None
  elif len(argv) == 3:  # Called with both port and package args.
    _, port, package = argv
  else:
    print('Use: python3 flash_firmware.py [port] firmware.ino.zip')
    return 1

  if not os.path.exists(package):  # Fail early if package zip isn't found.
    print(f'File "{package}" does not exist.')
    return 1

  available_ports = list_ports()
  if port is None:
    port = select_port(available_ports)
  elif not any(port == p[0] for p in available_ports):
    print(f'Port "{port}" is not available.')
    return 1

  baudrate = 115200
  flowcontrol = False
  singlebank = False
  touch = 1200
  serial_transport = DfuTransportSerial(port, baudrate, flowcontrol, singlebank,
                                        touch)

  def update_progress(progress=0, done=False, log_message=''):
    del log_message

    if done:
      print('\nDone.')
    elif progress:
      print('#', flush=True, end='' if progress % 40 else '\n')

  serial_transport.register_events_callback(DfuEvent.PROGRESS_EVENT,
                                            update_progress)
  dfu = Dfu(package, dfu_transport=serial_transport)

  print(f'Upgrading target on {port} with DFU package {package}.')
  dfu.dfu_send_images()  # Perform the DFU.
  print('Device programmed.')
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
