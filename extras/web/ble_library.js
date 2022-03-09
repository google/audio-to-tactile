/**
 * Copyright 2021-2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Library functions for BLE communication with the haptic device.
 * Defines a BLEManager class that contains state information for
 * the bluetooth connection.  UI update functions can be passed
 * to the BLEManager on instantiation depending on the needs of
 * the calling page.
 *
 * The BLE connect function must remain outside of the class in order
 * for it to work properly.
 *
 */


/**
 * Global constants.
 * Message codes should match the MessageType enum in cpp/message.h.
 */
const MESSAGE_TYPE_TEMPERATURE = 16;
const MESSAGE_TYPE_TUNING = 18;
const MESSAGE_TYPE_TACTILE_PATTERN = 19;
const MESSAGE_TYPE_GET_TUNING = 20;
const MESSAGE_TYPE_CHANNEL_MAP = 24;
const MESSAGE_TYPE_GET_CHANNEL_MAP = 25;
const MESSAGE_TYPE_STATS_RECORD = 26;
const MESSAGE_TYPE_CHANNEL_GAIN_UPDATE = 27;
const MESSAGE_TYPE_BATTERY_VOLTAGE = 28;
const MESSAGE_TYPE_DEVICE_NAME = 29;
const MESSAGE_TYPE_GET_DEVICE_NAME = 30;
const MESSAGE_TYPE_OTA_BOOTLOADMODE = 31;
const MESSAGE_TYPE_CALIBRATE_CHANNEL = 35;

const NUM_TACTORS = 10;
const ENVELOPE_TRACKER_RECORD_POINTS = 33;
const ENVELOPE_TRACKER_MEASUREMENT_PERIOD_MS = 30;


/**
 * Linearly maps `value_in` from [0, 255] to [min_in, min_out].
 * @param {number} value_in Number to map.
 * @param {number} min_out Minimum value in mapping range.
 * @param {number} max_out Maximum value in mapping range.
 * @return {number} Result of linear mapping.
 */
function lin_mapping(value_in, min_out, max_out) {
  return min_out + ((max_out - min_out) / 255) * value_in;
}

/**
 * Logarithmically maps `value_in` from [0, 255] to [min_in, min_out].
 * @param {number} value_in
 * @param {number} min_out Minimum value in mapping range.
 * @param {number} max_out Maximum value in mapping range.
 * @return {number} Result of logarithmic mapping
 */
function log_mapping(value_in, min_out, max_out) {
  return Math.exp(lin_mapping(value_in, Math.log(min_out), Math.log(max_out)));
}

/** Function that does nothing, for use as a default UI function. */
function noOp() {
  return;
}

/**
 * Connects BLE to device that both has a name starting with 'Audio-to-Tactile'
 * and is advertising the Nordic UART Service, after user input.
 * On a successful connection, sends a "get tuning" request.
 * Must be a stand-alone function, outside of a class.
 *
 * @param {!BLEManager} bleManager  An instance of the BLEManager class that
 *    will store BLE state created or modified in this function.
 */
function connect(bleManager) {
  // UUIDs for the BLE Nordic UART Service (NUS). We use NUS to send messages
  // between this web app and the device.
  //
  // Reference:
  // https://infocenter.nordicsemi.com/index.jsp?topic=%2Fsdk_nrf5_v16.0.0%2Fble_sdk_app_nus_eval.html
  const NUS_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
  const NUS_RX_CHARACTERISTIC_UUID = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';
  const NUS_TX_CHARACTERISTIC_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';
  let nusService;
  if (!navigator.bluetooth) {
    bleManager.log('WebBluetooth API is not available.\r\n' +
      'Please make sure the Web Bluetooth flag is enabled.');
    return;
  }
  navigator.bluetooth.requestDevice({  // Scan for a matching BLE device.
    filters: [
      {namePrefix: 'Audio-to-Tactile'},
      {services: [NUS_SERVICE_UUID]},
    ],
    optionalServices: [
      NUS_SERVICE_UUID,
      NUS_RX_CHARACTERISTIC_UUID,
      NUS_TX_CHARACTERISTIC_UUID
    ],
  })
    .then(device => {
      bleManager.bleDevice = device;
      bleManager.bleDevice.addEventListener('gattserverdisconnected',
          bleManager.onDisconnected.bind(bleManager));
      // Connect to the GATT server.
      return device.gatt.connect();
    })
    .then(server => {
      // Locate the NUS service.
      service = server.getPrimaryService(NUS_SERVICE_UUID);
      return service;
    }).then(service => {
      nusService = service;
      // Locate Rx characteristic.
      return nusService.getCharacteristic(NUS_RX_CHARACTERISTIC_UUID);
    })
    .then(characteristic => {
      bleManager.nusRx = characteristic;
      // Locate Tx characteristic.
      return nusService.getCharacteristic(NUS_TX_CHARACTERISTIC_UUID);
    })
    .then(characteristic => {
      bleManager.nusTx = characteristic;
      // Listen for messages sent from the device.
      bleManager.nusTx.addEventListener('characteristicvaluechanged',
          bleManager.onReceivedMessage.bind(bleManager));
      return bleManager.nusTx.startNotifications();
    })
    .then(() => {
      bleManager.log('BLE connected to ' + bleManager.bleDevice.name);
      bleManager.connected = true;
      bleManager.onConnectionUIUpdate(bleManager.connected);
      // Send "get tuning" request to the device.
      bleManager.requestGetTuning();
    })
    .catch(error => {
      bleManager.log('Error: ' + error);
      if (bleManager.bleDevice && bleManager.bleDevice.gatt.connected) {
        bleManager.bleDevice.gatt.disconnect();
      }
    });
}


/**
 * A wrapper for all BLE communication with the haptic devices,
 * except the connection action.
 */
class BleManager {
  /**
   * @param {function(): undefined=} loggingFunction A function to call for
   *    user-facing logging.
   * @param {function(): undefined=} onConnectionUIUpdate A function that
   *    updates the UI when the device's bluetooth connection state changes.
   * @param {function(): undefined=} batteryVoltageUIUpdate A function that
   *    updates the UI when a new battery reading is received.
   * @param {function(): undefined=} temperatureUIUpdate  A function that
   *    updates the UI when a new temperature reading is received.
   * @param {function(): undefined=} channelUIUpdate  A function that updates
   *    the UI when channelData changes.
   * @param {function(): undefined=} tuningKnobsUIUpdate  A function that
   *    updates the UI when tuningKnobs changes.
   * @param {function(): undefined=} updateStatsUI  A function that updates
   *    the UI when a stats record message is prcoessed.
   * @param {function(): undefined=} updateNameUI  A function that updates
   *    the UI when a new device name is received.
   */
  constructor(loggingFunction=noOp, onConnectionUIUpdate=noOp,
      batteryVoltageUIUpdate=noOp,temperatureUIUpdate=noOp,
      channelUIUpdate=noOp, tuningKnobsUIUpdate=noOp,
      updateStatsUI=noOp, updateNameUI=noOp) {
    this.log = loggingFunction;
    this.onConnectionUIUpdate = onConnectionUIUpdate;
    this.batteryVoltageUIUpdate = batteryVoltageUIUpdate;
    this.temperatureUIUpdate = temperatureUIUpdate;
    this.channelUIUpdate = channelUIUpdate;
    this.tuningKnobsUIUpdate = tuningKnobsUIUpdate;
    this.updateStatsUI = updateStatsUI;
    this.updateNameUI = updateNameUI;

    this.bleDevice = null;
    this.nusRx = null;
    this.nusTx = null;
    this.connected = false;

    // State informatin for channels.
    this.channelData = [];
    for (let c = 0; c < NUM_TACTORS; c++) {
      this.channelData.push({
        source: c + 1,
        enabled: true,
        gain: 63
      });
    }

    // Definitions and state of all the tuning knobs.
    // This should match the definitions in tactile/tuning.c.
    this.tuningKnobs = [
      {label: 'Input gain',
        default: 127,
        currentValue: 127,
        mapping: (x) => {
          return lin_mapping(x, -40.0, 40.315)
              .toFixed(1)
              .replace('-', '&minus;');
        },
        units: 'dB',
      },
      {label: 'Output gain',
        default: 191,
        currentValue: 191,
        mapping: (x) => {
          return lin_mapping(x, -18.0, 6.0).toFixed(1).replace('-', '&minus;');
        },
        units: 'dB',
      },
      {label: 'Energy smoothing',
        default: 85,
        currentValue: 85,
        mapping: (x) => { return log_mapping(x, 0.001, 1.0).toFixed(3); },
        units: 's',
      },
      {label: 'Noise adaptation',
        default: 127,
        currentValue: 127,
        mapping: (x) => { return log_mapping(x, 0.2, 20.0).toFixed(1); },
        units: 'dB/s',
      },
      {label: 'AGC strength',
        default: 191,
        currentValue: 191,
        mapping: (x) => { return lin_mapping(x, 0.1, 0.9).toFixed(2); },
      },
      {label: 'Compressor',
        default: 96,
        currentValue: 96,
        mapping: (x) => { return lin_mapping(x, 0.1, 0.5).toFixed(2); },
      },
    ];
  }

  /** Toggle the BLE connection. */
  connectionToggle() {
    if (this.connected) {
      this.disconnect();
    } else {
      connect(this);
    }
  }

  /** Disconnect the BLE device and update the UI. */
  disconnect() {
    if (this.bleDevice && this.bleDevice.gatt.connected) {
      this.bleDevice.gatt.disconnect();
      this.connected = false;
      this.onConnectionUIUpdate(this.connected);
    }
  }

  /**
   * Send a request to the device to play a specific tactile pattern.
   * @param {string} pattern String representation of tactile pattern.
   */
  requestPlayTactilePattern(pattern) {
    if (!this.connected) { return; }
    let enc = new TextEncoder();
    let messagePayload =
         enc.encode(pattern);
    this.writeMessage(MESSAGE_TYPE_TACTILE_PATTERN, messagePayload);
  }

  /**
   * Constructs and writes a message to the device to trigger
   * a tactile playback with a calibration pattern at the
   * specified amplitude.
   * @param {number} referenceChannel First channel index
   * @param {number} testChannel Second channel index
   * @param {number} amplitude Amplitude for playback, between [0.0, 1.0]
   */
  requestSetChannelCalibrate(referenceChannel, testChannel, amplitude) {
    if (!this.connected) { return; }
    let size = 4;
    let messagePayload = new Uint8Array(size);
    let messageType = MESSAGE_TYPE_CALIBRATE_CHANNEL;
    let referenceChannelData = this.channelData[referenceChannel].source - 1;
    let testChannelData = this.channelData[testChannel].source - 1;

    // Write channels in the first byte.
    messagePayload[0] = (referenceChannelData & 15) | (testChannelData << 4);

    // Write gain for test channel
    messagePayload[1] = this.channelData[testChannel].gain;

    // Convert amplitude to an integer
    amplitude = Math.max(0.0, Math.min(amplitude, 1.0));
    amplitude = Math.floor(amplitude * 65535);
    messagePayload[2] = amplitude;
    messagePayload[3] = amplitude >> 8;

    this.writeMessage(messageType, messagePayload);
  }

  /** Send a request to device to update channel gains. */
  requestSetChannelGainUpdate(c1, c2) {
    this.requestSetChannelMapOrGainUpdate([c1, c2],
                                          MESSAGE_TYPE_CHANNEL_GAIN_UPDATE);
  }

  /** Send a request to device to update channel map. */
  requestSetChannelMap() {
    this.requestSetChannelMapOrGainUpdate([], MESSAGE_TYPE_CHANNEL_MAP);
  }

  /** Send a request to device to update tuning. */
  requestSetTuning() {
    if (!this.connected) { return; }
    let messagePayload = new Uint8Array(this.tuningKnobs.length);
    for (let i = 0; i < this.tuningKnobs.length; i++) {
      messagePayload[i] = this.tuningKnobs[i].currentValue;
    }
    this.writeMessage(MESSAGE_TYPE_TUNING, messagePayload);
  }

  /**
   * Resets local channel data to default values and updates the UI.
   * Also requests a device update.
   */
  resetChannelMap() {
    for (let c = 0; c < NUM_TACTORS; c++) {
      this.channelData[c].source = c + 1;
      this.channelData[c].gain = 63;
      this.channelData[c].enabled = true;
    }
    this.channelUIUpdate();
    this.requestSetChannelMap();
  }

  /**
   * Sets a new name on the device.
   * @param {string} name The new name to set.
   */
  requestSetDeviceName(name) {
    if (!this.connected) { return; }
    let enc = new TextEncoder();
    let messagePayload = enc.encode(name);
    this.writeMessage(MESSAGE_TYPE_DEVICE_NAME, messagePayload);
  }

  /**
   * Resets local tuning data to default values and updates the UI.
   * Does not update the device.
   */
  resetTuning() {
    for (let i = 0; i < this.tuningKnobs.length; i++) {
      this.setTuningData(i, this.tuningKnobs[i].default);
      this.tuningKnobsUIUpdate();
    }
  }

  /**
   * Send a command to go into OTA bootloading mode.
   */
  sendBootloadModeCommand() {
    if (!this.connected) { return; }
    this.writeMessage(MESSAGE_TYPE_OTA_BOOTLOADMODE, new Uint8Array(0));
  }

  /**
   * Sets values in channelData.
   * @param {number} c Index of channel.
   * @param {string} field Name of field to set.
   * @param {number|boolean} val Value to assign to the field.  Can be a source
   *    index, a gain value, or a boolean for enabled.
   */
  setChannelData(c, field, val) {
    if (field == 'source' || field == 'gain') {
      val = parseInt(val);
    }
    this.channelData[c][field] = val;
  }

  /**
   * Sets the currentValue in tuningKnobs.
   * @param {number} c Index of tuning knob.
   * @param {number} val Value to assign to the tuning knob.
   */
  setTuningData(c, val) {
    this.tuningKnobs[c].currentValue = parseInt(val);
  }

  /**
   * Update the connected flag when disconnected, then update the UI.
   * @private
   */
  onDisconnected() {
    this.connected = false;
    this.log('BLE disconnected.');
    this.onConnectionUIUpdate(this.connected);
  }

  /**
   * Handles a new BLE message from the device by parsing message type and
   * calling the appropriate handler.
   * @param {!Event} event Event containing message information.
   * @private
   */
  onReceivedMessage(event) {
    let bytes = event.target.value;
    if (bytes.byteLength < 4 || bytes.byteLength != 4 + bytes.getUint8(3)) {
      this.log('Received invalid message.');
    }
    let messageType = bytes.getUint8(2);
    let messagePayload = new Uint8Array(bytes.getUint8(3));
    for (let i = 0; i < messagePayload.byteLength; i++) {
      messagePayload[i] = bytes.getUint8(4 + i);
    }
    this.log('Got type: ' + messageType + ', [' +
      messagePayload.join(', ') + ']');
    switch (messageType) {
      case MESSAGE_TYPE_TUNING:
        if (messagePayload.byteLength == this.tuningKnobs.length) {
          for (let i = 0; i < this.tuningKnobs.length; i++) {
            this.setTuningData(i, messagePayload[i]);
          }
          this.tuningKnobsUIUpdate();
          this.requestGetDeviceName();
        }
        break;
      case MESSAGE_TYPE_DEVICE_NAME:
        let name = String.fromCharCode.apply(null, messagePayload);
        this.updateNameUI(name);
        this.requestGetChannelMap();
        break;
      case MESSAGE_TYPE_CHANNEL_MAP:
        this.receiveChannelMap(messagePayload);
        break;
      case MESSAGE_TYPE_STATS_RECORD:
        this.receiveStatsRecord(messagePayload);
        break;
      case MESSAGE_TYPE_BATTERY_VOLTAGE:
        this.receiveBatteryVoltage(messagePayload);
        break;
      case MESSAGE_TYPE_TEMPERATURE:
        this.receiveTemperature(messagePayload);
        break;
      default:
        this.log('Unsupported message type.');
    }
  }

  /**
   * Handles a battery voltage message by parsing the input and updating the UI.
   * @param {!Uint8Array} messagePayload A byte array containing the
   *    new battery voltage.
   * @private
   */
  receiveBatteryVoltage(messagePayload) {
    let view = new DataView(messagePayload.buffer);
    let num = view.getFloat32(0, /*littleEndian=*/true);
    this.batteryVoltageUIUpdate(num);
  }

  /**
   * Handles a channel map message by parsing the input, recording the new
   * values, and updating the channel UI.
   * @param {!Uint8Array} messagePayload A byte array containing the channel
   *    data.
   * @private
   */
  receiveChannelMap(messagePayload) {
    if (messagePayload.length == 0) { return; }
    const numInput = 1 + ((messagePayload[0] + 15) & 15);
    const numOutput = messagePayload[0] >> 4;
    if (messagePayload.length != 1 + Math.ceil(numOutput / 2) +
                                 3 * Math.ceil(numOutput / 4)
        || numInput != NUM_TACTORS
        || numOutput != NUM_TACTORS) {
      return;
    }
    let i = 1;
    for (let c = 0; c < numOutput; c += 2, i++) {
      this.channelData[c].source =
          (messagePayload[i] & 15) + 1;
      this.channelData[c + 1].source =
          (messagePayload[i] >> 4) + 1;
    }
    const setGain = ((c, value, BLEInstance) => {
      if (c < NUM_TACTORS) {
        BLEInstance.channelData[c].enabled = (value > 0);
        if (value > 0) {
          BLEInstance.channelData[c].gain = value;
        }
      }
    });
    for (let c = 0; c < numOutput; c += 4, i += 3) {
      const pack24 = messagePayload[i]
                   | messagePayload[i + 1] << 8
                   | messagePayload[i + 2] << 16;
      setGain(c, pack24 & 63, this);
      setGain(c + 1, (pack24 >> 6) & 63, this);
      setGain(c + 2, (pack24 >> 12) & 63, this);
      setGain(c + 3, (pack24 >> 18) & 63, this);
    }
    this.channelUIUpdate();
  }

  /**
   * Handles a stats record message by parsing the input, calculating
   * the energy and updating the UI.
   * @param {!Uint8Array} messagePayload A byte array containing the new
   *    data points.
   * @private
   */
  receiveStatsRecord(messagePayload) {
    const decodeEnergy = ((value) => Math.pow(value / 255.0, 6.0));
    const decodeDelta = ((code) => [0, 1, 4, 11, 0, -1, -4, -11][code]);
    let cumulative = messagePayload[0];
    let out = new Float32Array(ENVELOPE_TRACKER_RECORD_POINTS);
    out[0] = decodeEnergy(cumulative);
    let k = 1;
    for (let i = 1; i < ENVELOPE_TRACKER_RECORD_POINTS; i += 8) {
      let pack24 = messagePayload[k]
        | messagePayload[k + 1] << 8
        | messagePayload[k + 2] << 16;
      k += 3;
      for (let j = 0; j < 8; ++j, pack24 >>= 3) {
        cumulative += decodeDelta(pack24 & 7);
        out[i + j] = decodeEnergy(cumulative);
      }
    }
    this.updateStatsUI(out);
  }

  /**
   * Handles a temperature message by parsing the input and updating the UI.
   * @param {!Uint8Array} messagePayload A byte array containing the
   *    new temperature reading.
   * @private
   */
  receiveTemperature(messagePayload) {
    let view = new DataView(messagePayload.buffer);
    let num = view.getFloat32(0, /*littleEndian=*/true);
    this.temperatureUIUpdate(num);
  }

  /**
   * Sends a request to the device for the current channel settings.
   * @private
   */
  requestGetChannelMap() {
    if (!this.connected) { return; }
    this.writeMessage(MESSAGE_TYPE_GET_CHANNEL_MAP, new Uint8Array(0));
  }

  /**
   * Requests the current name of the device.
   * @private
   */
  requestGetDeviceName() {
    if (!this.connected) { return; }
    this.writeMessage(MESSAGE_TYPE_GET_DEVICE_NAME, new Uint8Array(0));
  }

  /**
   * Sends a request to the device for the current tuning.
   * @private
   */
  requestGetTuning() {
    if (!this.connected) { return; }
    this.writeMessage(MESSAGE_TYPE_GET_TUNING, new Uint8Array(0));
  }

  /**
   * Constructs and writes a message to the device to update the
   * channel gain or channel map to the current values.
   * @param {!Array<number>} testChannels Array containing 0 or 2
   *    indices into the channel array.
   * @param {number} messageType Message code integer for
   *    MESSAGE_TYPE_CHANNEL_GAIN_UPDATE or MESSAGE_TYPE_CHANNEL_MAP.
   * @private
   */
  requestSetChannelMapOrGainUpdate(testChannels, messageType) {
    if (!this.connected) { return; }
    const numInput = NUM_TACTORS;
    const numOutput = NUM_TACTORS;
    let size = 1 + 3 * Math.ceil(numOutput / 4) +
        (testChannels.length ? 1 : Math.ceil(numOutput / 2));
    let messagePayload = new Uint8Array(size);
    // Write number of input and output channels in the first byte.
    messagePayload[0] = (numInput & 15) | numOutput << 4;
    let i = 1;
    if (testChannels.length) {
      let testChannelData0 = this.channelData[testChannels[0]].source - 1;
      let testChannelData1 = this.channelData[testChannels[1]].source - 1;
      messagePayload[i] = (testChannelData0 & 15) | testChannelData1 << 4;
      i++;
    } else {
      // Write source mapping, 4 bits per channel.
      for (let c = 0; c < numOutput; c += 2, i++) {
        let source0 = this.channelData[c].source - 1;
        let source1 = this.channelData[c + 1].source - 1;
        messagePayload[i] = (source0 & 15) | (source1 & 15) << 4;
      }
    }
    // Write gains, 6 bits per channel in little endian order.
    const getGain = ((c, BLEInstance) => {
      if (c < NUM_TACTORS &&
          BLEInstance.channelData[c].enabled) {
        return BLEInstance.channelData[c].gain;
      } else {
        return 0;
      }
    });
    for (let c = 0; c < numOutput; c += 4, i += 3) {
      const pack24 = getGain(c, this) | getGain(c + 1, this) << 6
          | getGain(c + 2, this) << 12 | getGain(c + 3, this) << 18;
      messagePayload[i] = pack24 & 255;
      messagePayload[i + 1] = (pack24 >> 8) & 255;
      messagePayload[i + 2] = (pack24 >> 16) & 255;
    }
    this.writeMessage(messageType, messagePayload);
  }

  /**
   * Writes a message to the device.
   * @param {number} messageType Code indicating the message type.
   * @param {!Uint8Array} messagePayload Contents to send to device.
   * @private
   */
  writeMessage(messageType, messagePayload) {
    if (!this.connected) { return; }
    this.log('Sent type: ' + messageType + ', [' +
             messagePayload.join(', ') + ']');
    let bytes = new Uint8Array(4 + messagePayload.byteLength);
    bytes[2] = messageType;
    bytes[3] = messagePayload.byteLength;
    for (let i = 0; i < messagePayload.byteLength; i++) {
      bytes[4 + i] = messagePayload[i];
    }
    // Compute Fletcher-16 checksum.
    let sum1 = 1;
    let sum2 = 0;
    for (let i = 2; i < bytes.length; i++) {
      sum1 += bytes[i];
      sum2 += sum1;
    }
    bytes[0] = sum1 % 255;
    bytes[1] = sum2 % 255;
    this.nusRx.writeValue(bytes);
  }
}
