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

import android.Manifest
import android.app.Activity
import android.bluetooth.le.ScanResult
import android.companion.CompanionDeviceManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.util.TypedValue
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.AttrRes
import androidx.annotation.ColorInt
import androidx.annotation.StringRes
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.android.material.textfield.TextInputLayout
import dagger.hilt.android.AndroidEntryPoint

/**
 * Define the Home fragment, shown when the app first opens. The Home fragment is for connecting or
 * disconnecting the BLE connection to an audio-to-tactile wearable and shows the main info and
 * actions for interacting with the device.
 */
@AndroidEntryPoint class HomeFragment : Fragment() {
  private val bleViewModel: BleViewModel by activityViewModels()
  private val selectDeviceActivityLauncher: ActivityResultLauncher<IntentSenderRequest> =
    registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { result ->
      if (result.resultCode == Activity.RESULT_OK) { // User selected a device from the menu.
        // Unpack device from `result.data`.
        val scanResult: ScanResult? =
          result.data?.getParcelableExtra(CompanionDeviceManager.EXTRA_DEVICE)
        scanResult?.device?.let { device -> bleViewModel.connect(device) }
      }
    }

  private val handler = Handler(Looper.getMainLooper())

  /** Gets the color associated with a resource, with its alpha component set to `alpha`. */
  @ColorInt
  private fun getColor(@AttrRes colorRes: Int, alpha: Int): Int {
    val value = TypedValue()
    requireContext().theme.resolveAttribute(colorRes, value, true)
    val color = (value.data and 0xffffff) or (alpha shl 24)
    return color
  }

  /** Converts a measurement to string according to `format`. */
  private fun measurementToString(@StringRes format: Int, measurement: Float?) =
    if (measurement != null && measurement >= 0f) {
      resources.getString(format, measurement)
    } else {
      "\u2013" // En dash ("--") if value is unavailable.
    }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_home, container, false)

    // Device name edit box.
    val deviceNameTextField: TextInputLayout = root.findViewById(R.id.device_name_text_field)
    deviceNameTextField.editText?.addTextChangedListener(object : TextWatcher {
      override fun afterTextChanged(s: Editable) {}

      override fun beforeTextChanged(s: CharSequence, start: Int, count: Int, after: Int) {}

      /** Called when text in the edit box changes. */
      override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) {
        // To avoid spamming BLE communication, update device name after a delay.
        val newName = s.toString()
        handler.removeCallbacksAndMessages(null) // If a previous update is pending, cancel it.
        handler.postDelayed({ bleViewModel.setDeviceName(newName) }, 1000L /* milliseconds */)
      }
    })
    bleViewModel.deviceName.observe(viewLifecycleOwner) { name ->
      handler.removeCallbacksAndMessages(null) // If a previous update is pending, cancel it.
      deviceNameTextField.editText?.apply {
        // Setting the EditText's text causes the cursor to jump. So only set if it differs.
        if (text.toString() != name) {
          setText(name)
        }
      }
    }

    // Display battery voltage and temperature.
    val batteryVoltageText: TextView = root.findViewById(R.id.battery_voltage)
    bleViewModel.batteryVoltage.observe(viewLifecycleOwner) {
      batteryVoltageText.text = measurementToString(R.string.battery_voltage_display, it)
    }
    val temperatureText: TextView = root.findViewById(R.id.temperature)
    bleViewModel.temperatureCelsius.observe(viewLifecycleOwner) {
      temperatureText.text = measurementToString(R.string.temperature_display, it)
    }

    // Create the input level plot.
    val inputLevelPlot: ImageView = root.findViewById(R.id.input_level_plot)
    inputLevelPlot.setImageDrawable(
      TimeSeriesPlot(
        bleViewModel.inputLevel,
        System.currentTimeMillis(),
        getColor(android.R.attr.colorPrimary, alpha = 255),
        getColor(android.R.attr.colorPrimary, alpha = 16),
        getColor(android.R.attr.colorForeground, alpha = 80)
      )
    )

    // Implement tactile pattern playback button.
    val tactilePatternTextField: TextInputLayout =
      root.findViewById(R.id.tactile_pattern_text_field)
    tactilePatternTextField.isEndIconVisible = true
    tactilePatternTextField.setEndIconOnClickListener {
      tactilePatternTextField.editText?.text?.let { text ->
        bleViewModel.playTactilePattern(text.toString().toByteArray())
      }
    }

    val connectButton: Button = root.findViewById(R.id.connect_button)

    // Create an object, which when "launched", asks the user for BLE
    // permissions and then scans for devices. Permissions are remembered once
    // they are granted (even if the app is closed or reinstalled), so the user
    // only needs to confirm them the first time running the app.
    val askPermissionAndScan =
      registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        Log.i(TAG, "Permissions granted: $grants")
        // Check whether the user agreed to allow the permissions.
        if (grants.all { (_, granted) -> granted }) {
          // Scan for BLE devices.
          bleViewModel.scanForDevices(requireActivity(), selectDeviceActivityLauncher)
        }
      }

    // Observe [BleViewModel.isConnected].
    bleViewModel.isConnected.observe(viewLifecycleOwner) { isConnected ->
      val enable = (isConnected == true)
      deviceNameTextField.isEnabled = enable
      tactilePatternTextField.isEnabled = enable

      // Implement connectButton. If there is currently no connection, clicking it initiates
      // a BLE scan. Otherwise, it disconnects.
      if (enable) {
        // Clicking `connectButton` disconnects BLE.
        connectButton.setText(R.string.disconnect_button_text)
        connectButton.setOnClickListener { bleViewModel.disconnect() }
      } else { // BLE is disconnected.
        // Clicking `connectButton` initiates a BLE scan.
        connectButton.setText(R.string.connect_button_text)
        connectButton.setOnClickListener { askPermissionAndScan.launch(blePermissions)
        }
      }
    }

    return root
  }

  private companion object {
    /** Tag for log messages generated by this class. */
    const val TAG = "HomeFragment"

    /** BLE-related permissions to request when connect button is pressed. */
    val blePermissions =
      arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
  }
}
