# flash_firmware

[flash_firmware](/extras/tools/flash_firmware.py)
is a Python script that flashes a compiled firmware ino.zip file to the device
over a USB connection. The script assumes your computer has Python and
the `adafruint-nrfutil` library installed, but does not require Arduino.

## Instructions

1. Install the
   [adafruit-nrfutil](https://github.com/adafruit/Adafruit_nRF52_nrfutil) Python
   library with shell command

    ```{.sh}
    pip3 install --user adafruit-nrfutil
    ```

    or see the [installation
    instructions](https://github.com/adafruit/Adafruit_nRF52_nrfutil#installation)
    for other options.

2. Download the [flash_firmware.py script](/extras/tools/flash_firmware.py)
   and new firmware ino.zip to your computer.

3. Use a micro USB cable to connect the device to the computer.

4. Turn on the device.

5. Navigate to the directory with the script and firmware, and run the script
   with shell command

    ```{.sh}
    python3 flash_firmware.py firmware.ino.zip
    ```

   The script prints "`Device programmed`" if successful.


A successful run looks like

```
$ python3 flash_firmware.py ~/Downloads/firmware.ino.zip
Available ports:
/dev/ttyACM1         Feather nRF52840 Express - TinyUSB Serial
/dev/ttyACM0         QT Py M0

Upgrading target on /dev/ttyACM1 with DFU package /home/getreuer/Downloads/firmware.ino.zip.
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
#########
Activating new firmware
Device programmed.
```

## Troubleshooting

The script auto-detects the port where the device is connected. It looks for
a device with "nrf52" in its name, otherwise it just attempts to use the
first available port. If this port selection fails for some reason, you can
specify the port with the following command, replacing the port name with the
correct port:

```{.sh}
python3 flash_firmware.py /dev/ttyACM1 firmware.ino.zip
```
