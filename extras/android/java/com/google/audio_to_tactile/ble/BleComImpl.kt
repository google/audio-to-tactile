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

/** BLE communication implementation. */
package com.google.audio_to_tactile.ble

import android.app.Activity
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanFilter
import android.companion.AssociationRequest
import android.companion.BluetoothLeDeviceFilter
import android.companion.CompanionDeviceManager
import android.content.Context
import android.content.IntentSender
import android.os.Handler
import android.os.Looper
import android.os.ParcelUuid
import android.util.Log
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import com.google.audio_to_tactile.Message
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.UUID
import java.util.regex.Pattern
import javax.inject.Inject

/** Real implementation of [BleCom]. */
class BleComImpl @Inject internal constructor(@ApplicationContext private val context: Context) :
  BleCom {
  private val handler = Handler(Looper.getMainLooper())
  private var connectedGatt: BluetoothGatt? = null
  /** NUS Tx characteristic, for receiving messages from the connected device. */
  private var nusTx: BluetoothGattCharacteristic? = null
  /** NUS Rx characteristic, for sending messages to the connected device. */
  private var nusRx: BluetoothGattCharacteristic? = null

  override var callback: BleComCallback =
    object : BleComCallback {
      override fun onConnect() {}
      override fun onDisconnect(reason: DisconnectReason) {}
      override fun onRead(message: Message) {}
    }

  private val gattCallback =
    object : BluetoothGattCallback() {
      /** `onConnectionStateChange` is called when BLE connects or disconnects. */
      override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
        Log.i(TAG, "onConnectionStateChange (status = $status, state = $newState).")

        when (status) {
          BluetoothGatt.GATT_SUCCESS ->
            when (newState) {
              BluetoothProfile.STATE_CONNECTED ->
                if (!gatt.requestMtu(DESIRED_MTU_SIZE)) {
                  error(DisconnectReason.OPERATION_FAILED, "Failed to initiate MTU request.")
                } else {
                  Log.i(TAG, "Initiating MTU request.")
                }
              BluetoothProfile.STATE_DISCONNECTED ->
                error(DisconnectReason.DEVICE_DISCONNECTED, "Device disconnected.")
            }
          STATUS_CONNECTION_TIMEOUT ->
            error(DisconnectReason.DEVICE_DISCONNECTED, "Connection timed out.")
          STATUS_DEVICE_DISCONNECTED ->
            error(DisconnectReason.DEVICE_DISCONNECTED, "Connection terminated by device.")
          STATUS_LOCAL_HOST_DISCONNECTED ->
            error(DisconnectReason.OPERATION_FAILED, "Connection terminated by local host.")
          else -> error(DisconnectReason.OPERATION_FAILED, "BLE error.")
        }
      }

      /** `onMtuChanged` is called when the Maximum Transmission Unit (MTU) is changed. */
      override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
        Log.i(TAG, "onMtuChanged (mtu = $mtu, status = $status).")
        if (mtu == DESIRED_MTU_SIZE && status == BluetoothGatt.GATT_SUCCESS) {
          // Initiate GATT service discovery.
          if (!gatt.discoverServices()) {
            error(DisconnectReason.OPERATION_FAILED, "Failed to initiate service discovery.")
          } else {
            Log.i(TAG, "Initiating service discovery.")
          }
        } else {
          error(DisconnectReason.MTU_NEGOTIATION_FAILED, "MTU negotiation failed.")
        }
      }

      /** `onServicesDiscovered` is called when service discovery completes. */
      override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
        Log.i(TAG, "onServicesDiscovered (status = $status).")

        if (status == BluetoothGatt.GATT_SUCCESS) {
          // Locate the NUS service.
          val service = gatt.getService(UUID.fromString(NUS_SERVICE_UUID))
          if (service == null) {
            error(DisconnectReason.NUS_SERVICE_NOT_FOUND, "NUS service not found.")
            return
          }

          // Locate NUS Tx and Rx characteristics.
          val nusTx = service.getCharacteristic(UUID.fromString(NUS_TX_CHARACTERISTIC_UUID))
          val nusRx = service.getCharacteristic(UUID.fromString(NUS_RX_CHARACTERISTIC_UUID))
          if (nusTx == null || nusRx == null) { // Should not happen unless the BLE device is buggy.
            error(
              DisconnectReason.CHARACTERISTIC_NOT_FOUND,
              "Found NUS service, but not NUS Tx and Rx characteristics."
            )
            return
          }

          // Enable notifications of NUS Tx.
          if (!gatt.setCharacteristicNotification(nusTx, true)) {
            error(DisconnectReason.OPERATION_FAILED, "Failed to enable notifications on NUS Tx.")
            return
          }
          val descriptor =
            nusTx.getDescriptor(UUID.fromString(CLIENT_CHARACTERISTIC_CONFIG_DESCRIPTOR_UUID))
          if (descriptor == null) { // Should not happen unless the BLE device is buggy.
            error(
              DisconnectReason.CHARACTERISTIC_NOT_FOUND,
              "Missing the CCCD for the NUS Tx characteristic."
            )
            return
          }
          descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
          gatt.writeDescriptor(descriptor)

          this@BleComImpl.nusTx = nusTx
          this@BleComImpl.nusRx = nusRx
        } else {
          error(
            DisconnectReason.SERVICE_DISCOVERY_FAILED,
            "Service discovery failed (status = $status)."
          )
        }
      }

      /** `onDescriptorWrite` is called when notifications are set on NUS Tx. */
      override fun onDescriptorWrite(
        gatt: BluetoothGatt,
        descriptor: BluetoothGattDescriptor,
        status: Int
      ) {
        Log.i(TAG, "onDescriptorWrite (status = $status).")

        if (status == BluetoothGatt.GATT_SUCCESS) {
          Log.i(TAG, "Successfully established connection.")
          _deviceName = gatt.getDevice()?.name.orEmpty()
          _deviceAddress = gatt.getDevice()?.address.toString().orEmpty()
          handler.post() { callback.onConnect() }
        } else {
          error(DisconnectReason.OPERATION_FAILED, "Write descriptor failed.")
        }
      }

      /** `onCharacteristicChanged` is called when the app receives a message on NUS Tx. */
      override fun onCharacteristicChanged(
        gatt: BluetoothGatt,
        characteristic: BluetoothGattCharacteristic
      ) {
        if (characteristic == nusTx) {
          val bytes: ByteArray = characteristic.value
          val message: Message? = Message.deserialize(bytes)
          if (message != null) {
            handler.post() { callback.onRead(message) }
          } else {
            Log.e(TAG, "Received invalid Message: ${byteArrayToString(bytes)}")
          }
        }
      }
    }

  override fun scanForDevices(
    activity: Activity,
    launcher: ActivityResultLauncher<IntentSenderRequest>
  ) {
    // If `connectedGatt` is nonnull, BLE is already connected or in the process of connecting.
    if (connectedGatt != null) {
      return
    }

    val deviceManager =
      activity.getSystemService(Context.COMPANION_DEVICE_SERVICE) as CompanionDeviceManager

    // Define device scan filtering criteria:
    // - name starts with "Audio-to-Tactile"
    // - supports the NUS service
    val deviceFilter: BluetoothLeDeviceFilter =
      BluetoothLeDeviceFilter.Builder()
        .setNamePattern(Pattern.compile("Audio-to-Tactile.*"))
        .setScanFilter(
          ScanFilter.Builder().setServiceUuid(ParcelUuid.fromString(NUS_SERVICE_UUID)).build()
        )
        .build()

    // Initiate device scan.
    deviceManager.associate(
      AssociationRequest.Builder().addDeviceFilter(deviceFilter).build(),
      object : CompanionDeviceManager.Callback() {
        override fun onDeviceFound(chooserLauncher: IntentSender) {
          launcher.launch(IntentSenderRequest.Builder(chooserLauncher).build())
        }

        override fun onFailure(error: CharSequence?) {
          // `onFailure` is called if there was an error looking for devices. There is nothing more
          // we can do here, but the user may press the "Connect" button to try again.
        }
      },
      null
    )
  }

  override fun connect(device: BluetoothDevice) {
    // If `connectedGatt` is nonnull, BLE is already connected or in the process of connecting.
    if (connectedGatt != null) {
      return
    }

    Log.i(TAG, "Connecting to ${device.name}")
    connectedGatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
  }

  override fun disconnect() {
    disconnectImpl(DisconnectReason.APP_DISCONNECTED)
  }

  private fun disconnectImpl(reason: DisconnectReason) {
    connectedGatt?.let {
      Log.i(TAG, "Disconnecting.")
      it.disconnect()
      it.close()
      handler.post { callback.onDisconnect(reason) }
    }
    connectedGatt = null
  }

  override fun write(message: Message) {
    writeBytes(message.serialize())
  }

  // Device name and address are stored when GATT is connected.
  private var _deviceName = ""
  override val deviceName: String
    get() = _deviceName

  private var _deviceAddress = "00:00:00:00:00:00"
  override val deviceAddress: String
    get() = _deviceAddress

  /** Writes raw bytes to NUS Rx. */
  private fun writeBytes(bytes: ByteArray) {
    val connectedGatt = this.connectedGatt
    val nusRx = this.nusRx
    if (connectedGatt == null || nusRx == null) {
      return
    }

    nusRx.value = bytes

    if (!connectedGatt.writeCharacteristic(nusRx)) {
      error(DisconnectReason.OPERATION_FAILED, "Failed to initiate write to NUS Rx.")
    }
  }

  /** Signals an error and disconnects. */
  private fun error(reason: DisconnectReason, logMessage: String) {
    Log.e(TAG, logMessage)
    disconnectImpl(reason)
  }

  /** Turns a `ByteArray` to string for logging. */
  private fun byteArrayToString(bytes: ByteArray) =
    "[" + bytes.joinToString(", ") { (it.toInt() and 0xff).toString() } + "]"

  private companion object {
    /** Tag for log messages generated by this class. */
    const val TAG = "BleComImpl"
    /**
     * MTU size to request, large enough that the largest packet we can send is 128 + 3 bytes, or a
     * message payload of 124 bytes, the bottleneck being the 128-byte buffer used for messages on
     * the firmware side. Android supports an MTU up to 517, the nRF52 supports an MTU up to 247,
     * and most microcontrollers support at least 180.
     */
    const val DESIRED_MTU_SIZE = 144
    /** UUID for the BLE Nordic UART Service (NUS). */
    const val NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
    /** NUS Rx characteristic, used for sending data from the app to the device. */
    const val NUS_RX_CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
    /** NUS Tx characteristic, used for sending data from the device to the app. */
    const val NUS_TX_CHARACTERISTIC_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
    /** Client characteristic config descriptor (CCCD), used to enable notifications on NUS Tx. */
    const val CLIENT_CHARACTERISTIC_CONFIG_DESCRIPTOR_UUID = "00002902-0000-1000-8000-00805f9b34fb"

    // These codes don't seem to be exposed in the Android's BLE APIs, but can be observed when
    // certain errors occur. Their definitions can be found in:
    // https://github.com/espressif/esp-idf/blob/master/components/bt/host/bluedroid/stack/include/stack/gatt_api.h

    /** Status code when BLE timed out. It's likely the device powered off or went out of range. */
    const val STATUS_CONNECTION_TIMEOUT = 0x08
    /** Status code when the device terminated the connection. */
    const val STATUS_DEVICE_DISCONNECTED = 0x13
    /** Status code when the local host terminated the connection. */
    const val STATUS_LOCAL_HOST_DISCONNECTED = 0x16
  }
}
