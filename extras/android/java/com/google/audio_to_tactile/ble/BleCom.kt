/* Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** Interfaces for BLE communication. */
package com.google.audio_to_tactile.ble

import android.app.Activity
import android.bluetooth.BluetoothDevice
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import androidx.annotation.StringRes
import com.google.audio_to_tactile.Message
import com.google.audio_to_tactile.R

/** Possible errors or other reasons for disconnection passed to `onDisconnect()`. */
enum class DisconnectReason(@StringRes val stringRes: Int) {
  /** Catch-all case for when a BLE operation failed, but we don't have a specific explanation. */
  OPERATION_FAILED(R.string.ble_operation_failed),
  /** The device disconnected. */
  DEVICE_DISCONNECTED(R.string.ble_device_disconnected),
  /** Failed to negotiate Maximum Transmission Unit (MTU) with the device. */
  MTU_NEGOTIATION_FAILED(R.string.ble_mtu_negotiation_failed),
  /** The user told the app disconnect. */
  APP_DISCONNECTED(R.string.ble_app_disconnected),
  /** Connected to the GATT server, but service discovery failed. */
  SERVICE_DISCOVERY_FAILED(R.string.ble_service_discovery_failed),
  /** The Nordic UART Service (NUS) was not found. The connected devices is likely incompatible. */
  NUS_SERVICE_NOT_FOUND(R.string.ble_nus_service_not_found),
  /** NUS was found, but one of its characteristics. Shouldn't happen unless device is buggy. */
  CHARACTERISTIC_NOT_FOUND(R.string.ble_characteristic_not_found),
}

/** Callbacks for responding to BLE events. */
interface BleComCallback {
  /** Called once BLE connection, service discovery, and NUS Tx notifications are established. */
  fun onConnect()

  /** Called when BLE disconnects, where `reason` is an explanation why it disconnected. */
  fun onDisconnect(reason: DisconnectReason)

  /** Called when the app receives a message from the connected device through NUS Tx. */
  fun onRead(message: Message)
}

/**
 * An abstraction layer for scanning, connecting, and communicating through the BLE Nordic UART
 * Service (NUS). The real implementation is [BleComImpl].
 */
interface BleCom {
  /** Callbacks for responding to BLE events. */
  var callback: BleComCallback

  /** Connected device's name. */
  val deviceName: String

  /** Connected device unique address. Returns a string such as E1:A7:79:EB:A0:2E */
  val deviceAddress: String

  /**
   * Initiates a scan for BLE devices and displays a list to allow the user to select a devices.
   * Scan results are filtered to show only devices with name beginning with "Audio-to-Tactile" and
   * that have the Nordic UART Service (NUS).
   */
  fun scanForDevices(activity: Activity, launcher: ActivityResultLauncher<IntentSenderRequest>)

  /**
   * Initiates connection to `device`. `callback.onConnect` will be called if successful, or
   * `callback.onDisconnect` will called with a `DisconnectReason` on failure.
   */
  fun connect(device: BluetoothDevice)

  /** Disconnects, if connected. */
  fun disconnect()

  /** Sends `message` to the connected device through NUS Rx. */
  fun write(message: Message)
}
