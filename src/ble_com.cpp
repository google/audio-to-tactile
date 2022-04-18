// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ble_com.h"  // NOLINT(build/include)

#include "cpp/std_shim.h"

namespace audio_tactile {

AudioTactileBleCom BleCom;

void OnBleConnect(uint16_t connection_handle) {
  BleCom.event_ = BleEvent::kConnect;
  BleCom.event_fun_();
}

void OnBleDisconnect(uint16_t connection_handle, uint8_t reason) {
  BleCom.event_ = BleEvent::kDisconnect;
  BleCom.event_fun_();
}

void OnBleUartRx(uint16_t connection_handle) {
  BleCom.ReadFromBleUart();
}

namespace {
// By default, the SoftDevice has memory for an advertising name of up to
// BLE_GAP_DEVNAME_DEFAULT_LEN (= 31 chars). Otherwise, either `vloc` in the
// ble_gap_cfg_device_name_t struct must be set to BLE_GATTS_VLOC_USER, or the
// attribute table size must be increased. It doesn't seem that Bluefruit
// handles this or exposes an API for either of these things. If we try to set a
// name longer than 31 chars, it uses the name "Feather nRF52840 Express".
constexpr int kMaxAdvertisingNameLength = BLE_GAP_DEVNAME_DEFAULT_LEN;

// Makes the name to use for BLE advertising. We concatenate "Audio-to-Tactile "
// with the customizable `device_name` as the name used in BLE advertising,
// truncating as needed. The Android app identifies our devices from the
// "Audio-to-Tactile" prefix. The `advertising_name` output buffer must have
// space for (kMaxAdvertisingNameLength + 1) chars.
void MakeAdvertisingName(const char* device_name, char* advertising_name) {
  // TODO: Investigate instead using `BLEDis` to identify devices.
  constexpr const char* kPrefix = "Audio-to-Tactile ";
  const int kPrefixLength = strlen(kPrefix);
  // Calculate device_name length that keeps the result under the max length.
  const int truncated_device_name_length = std_shim::min<int>(
      kMaxAdvertisingNameLength - kPrefixLength, strlen(device_name));

  memcpy(advertising_name, kPrefix, kPrefixLength);
  advertising_name += kPrefixLength;
  memcpy(advertising_name, device_name, truncated_device_name_length);
  advertising_name += truncated_device_name_length;
  *advertising_name = '\0';
}
}  // namespace

void AudioTactileBleCom::Init(const char* device_name, void (*event_fun)()) {
  event_fun_ = event_fun;

  Bluefruit.autoConnLed(false);
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.begin();
  Bluefruit.setTxPower(-8);
  char advertising_name[kMaxAdvertisingNameLength + 1];
  MakeAdvertisingName(device_name, advertising_name);
  Bluefruit.setName(advertising_name);
  Bluefruit.Periph.setConnectCallback(OnBleConnect);
  Bluefruit.Periph.setDisconnectCallback(OnBleDisconnect);

  ble_uart_.begin();
  ble_uart_.setRxCallback(OnBleUartRx, true);

  // Start the OTA (Over-the-air) DFU (Device Firmware Update) functionality.
  bledfu_.begin();

  // Start advertising.
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(ble_uart_);
  Bluefruit.ScanResponse.addName();
  // Start advertising again if disconnected.
  Bluefruit.Advertising.restartOnDisconnect(true);
  // To balance between quick discovery vs. battery cost, we first do "fast"
  // advertising with an interval of 20 ms for 30 s. Then we switch to slower
  // interval of 318.75 ms for another 270 s, then stop advertising. After that
  // point, BLE will no longer connect until the device is reset.
  //
  // Reference, Apple's recommendations for BLE advertising:
  // https://developer.apple.com/library/archive/qa/qa1931/_index.html
  Bluefruit.Advertising.setInterval(32, 510);  // Units of 0.625 ms.
  Bluefruit.Advertising.setFastTimeout(30);  // Units of seconds.
  Bluefruit.Advertising.start(300);  // Units of seconds.

  Serial.print("BleCom: Advertising as \"");
  Serial.print(advertising_name);
  Serial.println("\"");
}

void AudioTactileBleCom::ReadFromBleUart() {
  constexpr int kHeaderSize = Message::kHeaderSize;
  constexpr int kMaxPayloadSize = Message::kMaxPayloadSize;
  const int num_received_bytes = ble_uart_.available();
  uint8_t* dest = rx_message_.data();
  int payload_size;

  // Read from ble_uart_, writing directly into rx_message_. We first read the
  // 4-byte message header to get the payload size, then read the payload.
  //
  // When BLE receives a packet, BLEUart appears to buffer the whole packet and
  // then call the `OnBleUartRx()` callback above. Then, this function can
  // instantly read the whole packet without waiting. This has been reliable in
  // all tests with it so far, in which BLE transferred at most 20 bytes at a
  // time. I'd expect that with a sufficiently large transmission, NUS will
  // break it into multiple packets sent over multiple BLE time intervals. Then
  // this receiving code would need some streaming or waiting logic to get the
  // whole transmission. But this is complicated and not needed so far.
  if (num_received_bytes >= kHeaderSize &&  // Read the message header.
      ble_uart_.read(dest, kHeaderSize) == kHeaderSize &&
      dest[2] >= 1 &&  // Check type field.
      (payload_size = dest[3]) <= kMaxPayloadSize &&  // Check size field.
      num_received_bytes >= kHeaderSize + payload_size &&  // Read the payload.
      ble_uart_.read(dest + kHeaderSize, payload_size) == payload_size &&
      // Verify the checksum.
      rx_message_.VerifyChecksum()) {
    event_ = BleEvent::kMessageReceived;
  } else {
    event_ = BleEvent::kInvalidMessage;
  }

  ble_uart_.flush();  // Flush in case a partial or failed message remains.
  event_fun_();
}

void AudioTactileBleCom::SendTxMessage() {
  tx_message_.SetBleHeader();
  ble_uart_.write(tx_message_.data(), tx_message_.size());
}

}  // namespace audio_tactile
