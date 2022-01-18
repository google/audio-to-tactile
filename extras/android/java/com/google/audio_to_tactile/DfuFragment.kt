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

package com.google.audio_to_tactile

import android.app.Activity.RESULT_OK
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.textfield.TextInputLayout
import dagger.hilt.android.AndroidEntryPoint
import no.nordicsemi.android.dfu.DfuProgressListener
import no.nordicsemi.android.dfu.DfuProgressListenerAdapter
import no.nordicsemi.android.dfu.DfuServiceInitiator
import no.nordicsemi.android.dfu.DfuServiceListenerHelper

/**
 * Define the DFU fragment, accessed from the nav drawer. This fragment will enable the user to do a
 * BLE device firmware update (DFU), sending new firmware to the wearable device.
 *
 * TODO: Add a test for DfuFragment.
 */
@AndroidEntryPoint class DfuFragment : Fragment() {
  // Those views will be initialized later in onCreateView.
  private lateinit var dfuStatusText: TextView
  private lateinit var dfuProgressBar: LinearProgressIndicator
  private lateinit var dfuCurrentlyConnectedDeviceCheckBox: CheckBox
  private lateinit var dfuForceBootloaderCheckBox: CheckBox
  private lateinit var dfuOtherDeviceCheckBox: CheckBox
  private lateinit var dfuOtherDeviceAddressText: TextInputLayout

  private val bleViewModel: BleViewModel by activityViewModels()

  private var connectedAddress = "00:00:00:00:00:00" // Placeholder value for device address.

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_dfu, container, false)
    val dfuStartButton: Button = root.findViewById(R.id.dfu_start_button)
    dfuProgressBar = root.findViewById(R.id.dfu_progress_bar)
    dfuStatusText = root.findViewById(R.id.dfu_status_text)
    dfuCurrentlyConnectedDeviceCheckBox = root.findViewById(R.id.dfu_upload_to_connected_checkbox)
    dfuForceBootloaderCheckBox = root.findViewById(R.id.prepare_for_bootloading_checkbox)
    dfuOtherDeviceCheckBox = root.findViewById(R.id.dfu_to_other_device_checkbox)
    dfuOtherDeviceAddressText = root.findViewById(R.id.dfu_alternative_address)

    dfuProgressBar.max = 100

    dfuStartButton.setOnClickListener {
      // Choose the firmware file (has to be a zip file).
      val intent = Intent().setType("application/zip").setAction(Intent.ACTION_GET_CONTENT)
      startActivityForResult(Intent.createChooser(intent, "Select a file"), PICK_ZIP_FILE)

      // Start the DFU notifications.
      DfuServiceInitiator.createDfuNotificationChannel(requireContext())
      DfuServiceListenerHelper.registerProgressListener(requireContext(), dfuProgressListener)

      // If forcing bootloader is enabled, send the forcing command now, before DFU is started.
      if (dfuForceBootloaderCheckBox.isChecked() && bleViewModel.isConnected.value == true) {
        bleViewModel.prepareForBootloading()
      }
    }

    // Do some UI logic to make sure that uploading to connected device and unconnected device
    // at same time is not possible.
    dfuOtherDeviceCheckBox.setOnClickListener {
      if (dfuOtherDeviceCheckBox.isChecked() == false) {
        dfuForceBootloaderCheckBox.setEnabled(true)
        dfuOtherDeviceAddressText.setEnabled(false)
      }
      if (dfuOtherDeviceCheckBox.isChecked() == true) {
        dfuForceBootloaderCheckBox.setEnabled(false)
        dfuOtherDeviceAddressText.setEnabled(true)
        dfuCurrentlyConnectedDeviceCheckBox.setChecked(false)
        dfuForceBootloaderCheckBox.setChecked(false)
      }
    }
    dfuCurrentlyConnectedDeviceCheckBox.setOnClickListener {
      if (dfuCurrentlyConnectedDeviceCheckBox.isChecked() == true) {
        dfuOtherDeviceCheckBox.setChecked(false)
        dfuForceBootloaderCheckBox.setEnabled(true)
        dfuOtherDeviceAddressText.setEnabled(false)
      }
    }
    return root
  }

  private val dfuProgressListener: DfuProgressListener =
    object : DfuProgressListenerAdapter() {
      override fun onDeviceConnecting(deviceAddress: String) {
        Toast.makeText(context, "Connecting", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Connecting"
      }

      override fun onDeviceConnected(deviceAddress: String) {
        Toast.makeText(context, "Connected", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Connected"
      }

      override fun onDfuProcessStarting(deviceAddress: String) {
        Toast.makeText(context, "Starting DFU", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Starting DFU"
      }

      override fun onDfuProcessStarted(deviceAddress: String) {
        Toast.makeText(context, "Started DFU", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Started DFU"
      }

      override fun onProgressChanged(
        deviceAddress: String,
        percent: Int,
        speed: Float,
        avgSpeed: Float,
        currentPart: Int,
        partsTotal: Int
      ) {
        dfuProgressBar.progress = percent
        dfuStatusText.text = "Uploading Code..."
      }

      override fun onDeviceDisconnecting(deviceAddress: String?) {
        Toast.makeText(context, "Disconnecting", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Disconnecting"
      }

      override fun onDeviceDisconnected(deviceAddress: String) {
        Toast.makeText(context, "Disconnected", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Disconnected"
      }

      override fun onDfuCompleted(deviceAddress: String) {
        Toast.makeText(context, "Finished", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Finished"
      }

      override fun onDfuAborted(deviceAddress: String) {
        Toast.makeText(context, "Aborted", Toast.LENGTH_SHORT).show()
        dfuStatusText.text = "Aborted"
      }

      override fun onError(deviceAddress: String, error: Int, errorType: Int, message: String?) {
        dfuStatusText.text = "Error $error, $errorType, $message"
        Toast.makeText(context, "Error", Toast.LENGTH_SHORT).show()
      }
    }

  // Actions when the firmware file is chosen.
  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)

    if (requestCode == PICK_ZIP_FILE && resultCode == RESULT_OK) {
      // The URI with the location of the file. Returns on null.
      val selectedFile = data?.data ?: return

      val address =
        if (dfuCurrentlyConnectedDeviceCheckBox.isChecked() &&
            bleViewModel.isConnected.value == true
        ) {
          bleViewModel.deviceBleAddress
        } else if (dfuOtherDeviceCheckBox.isChecked()) {
          dfuOtherDeviceAddressText.getEditText()?.getText().toString()
        } else {
          "00:00:00:00:00:00" // Placeholder.
        }

      // Make sure the address is defined and is the right size before uploading to device.
      val ADDRESS_LENGTH = 17
      if (address == "00:00:00:00:00:00" || address.length != ADDRESS_LENGTH) {
        Toast.makeText(context, "Invalid address-$address", Toast.LENGTH_LONG).show()
      } else {
        // Start the upload.
        val starter =
          DfuServiceInitiator(address)
            .setKeepBond(false)
            .setForceDfu(true)
            .setPacketsReceiptNotificationsEnabled(false)
            .setUnsafeExperimentalButtonlessServiceInSecureDfuEnabled(true)
            .setZip(selectedFile, null)
        starter.start(requireContext(), DfuService::class.java)
      }
    }
  }
  private companion object {
    const val PICK_ZIP_FILE = 111
    const val TAG = "dfuFragment"
  }
}
