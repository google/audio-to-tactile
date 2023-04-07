/* Copyright 2021-2022 Google LLC
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

import android.app.Activity
import android.bluetooth.BluetoothDevice
import android.os.Build
import android.util.Log
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import androidx.annotation.VisibleForTesting
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.google.audio_to_tactile.ble.BleCom
import com.google.audio_to_tactile.ble.BleComCallback
import com.google.audio_to_tactile.ble.DisconnectReason
import dagger.hilt.android.lifecycle.HiltViewModel
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.util.Calendar
import javax.inject.Inject
import kotlin.math.sqrt

/** Notifies MutableLiveData observers by assigning its value to itself. */
fun <T> MutableLiveData<T>.notifyObservers() { setValue(getValue()) }

/** `ViewModel` for BLE communication. */
@HiltViewModel
class BleViewModel @Inject constructor(private val bleCom: BleCom) : ViewModel() {
  /** Date when the app was built. */
  var appBuildDate: LocalDate? = null

  private val _isConnected = MutableLiveData<Boolean>(false)
  /**
   * Indicates whether BLE is connected. UI fragments can `.observe` this variable for notifications
   * when BLE connects or disconnects.
   */
  val isConnected: LiveData<Boolean>
    get() = _isConnected

  /** Name used by the connected device for BLE advertising. */
  val deviceBleName: String
    get() = bleCom.deviceName

  /** BLE address of connected device. */
  val deviceBleAddress: String
    get() = bleCom.deviceAddress

  private val _disconnectReason = MutableLiveData<DisconnectReason>()
  /** When BLE disconnects, `disconnectReason` gives a reason to explain why. */
  val disconnectReason: LiveData<DisconnectReason>
    get() = _disconnectReason

  private val _tuning = MutableLiveData<Tuning>(Tuning())
  /** Algorithm tuning knobs. */
  val tuning: LiveData<Tuning>
    get() = _tuning

  private val _channelMap =
    MutableLiveData<ChannelMap>(ChannelMap(numInputChannels = 10, numOutputChannels = 10))
  /** Algorithm tuning knobs. */
  val channelMap: LiveData<ChannelMap>
    get() = _channelMap

  private val _firmwareBuildDate = MutableLiveData<LocalDate>()
  /** Date when device firmware was built. */
  val firmwareBuildDate: LiveData<LocalDate>
    get() = _firmwareBuildDate

  private val _deviceName = MutableLiveData<String>("")
  /** User-customizable device name. */
  val deviceName: LiveData<String>
    get() = _deviceName

  private val _batteryVoltage = MutableLiveData<Float>(-1f)
  /** Device battery voltage from the latest received [BatteryVoltageMessage]. */
  val batteryVoltage: LiveData<Float>
    get() = _batteryVoltage

  private val _temperatureCelsius = MutableLiveData<Float>(-1f)
  /** Device thermistor temperature from the latest received [TemperatureMessage]. */
  val temperatureCelsius: LiveData<Float>
    get() = _temperatureCelsius

  private val _inputLevel = TimeSeries(samplePeriodMs = 30L, windowDurationMs = 6000L)
  /** Mic input level time series received from [StatsRecordMessage]. */
  val inputLevel: TimeSeries
    get() = _inputLevel

  private val _flashMemoryWriteStatus = MutableLiveData<FlashMemoryStatus>()
  /** Status returned by latest attempt to write to device's flash memory */
  val flashMemoryWriteStatus: LiveData<FlashMemoryStatus>
    get() = _flashMemoryWriteStatus

  private val _tactilePattern = MutableLiveData(TactilePattern())
  /** Tactile pattern shown in the pattern editor. */
  val tactilePattern: LiveData<TactilePattern>
    get() = _tactilePattern

  private val _tactilePatternSelectedIndex = MutableLiveData<Int>(-1)
  /** The index of the currently selected tactile pattern op, or -1 if none is selected. */
  val tactilePatternSelectedIndex: LiveData<Int>
    get() = _tactilePatternSelectedIndex

  /** The currently selected tactile pattern op, or null if none is selected. */
  val tactilePatternSelectedOp: TactilePattern.Op?
    get() =
      _tactilePattern.value?.ops?.let { ops ->
        _tactilePatternSelectedIndex.value?.let { i ->
          if (i in ops.indices) { ops[i] } else { null }
        }
      }

  data class LogLine(val timestamp: String, val message: String)
  private val _logLines = MutableLiveData(mutableListOf<LogLine>())
  /** Log lines, to be displayed in the Log fragment. */
  val logLines: LiveData<MutableList<LogLine>>
    get() = _logLines

  /** Format for log timestamps. */
  private val logTimestampFormat = SimpleDateFormat("HH:mm:ss.SSS")

  init {
    // Callbacks for responding to BLE communication events.
    bleCom.callback =
      object : BleComCallback {
        override fun onConnect() {
          _isConnected.setValue(true)
          log("BLE successfully established connection.")

          // On connection, ask device for an on-connection batch message. The firmware expects this
          // to be the first message sent, and uses it as a cue to begin sending data to the app.
          sendMessageIfConnected(GetOnConnectionBatchMessage())
        }

        override fun onDisconnect(reason: DisconnectReason) {
          _isConnected.setValue(false)
          _disconnectReason.setValue(reason)
          log("BLE disconnected")
        }

        override fun onRead(message: Message) {
          // StatsRecordMessages are received frequently, about once per second. So we don't
          // log this kind of message to reduce log spam.
          if (!(message is StatsRecordMessage)) {
            log("Got $message")
          }

          when (message) {
            is OnConnectionBatchMessage -> {
              message.firmwareBuildDate?.let {
                log("Device firmware build date: $it")
                _firmwareBuildDate.setValue(it)
              }
              _batteryVoltage.setValue(message.batteryVoltage)
              _temperatureCelsius.setValue(message.temperatureCelsius)

              message.deviceName?.let { _deviceName.setValue(it) }
              message.tuning?.let { _tuning.setValue(it) }
              message.channelMap?.let { channelMap ->
                _channelMap.value?.let {
                  if (it.sameNumChannelsAs(channelMap)) {
                    _channelMap.setValue(channelMap)
                  }
                }
              }
            }
            is BatteryVoltageMessage -> {
              _batteryVoltage.setValue(message.voltage)
            }
            is TemperatureMessage -> {
              _temperatureCelsius.setValue(message.celsius)
            }
            is StatsRecordMessage -> {
              // A fudge factor for plotting: [TimeSeriesPlot] expects values in [0, 1], but the
              // input amplitude is usually below 0.2. So we scale by 5 to fill the box better.
              val Y_SCALE = 5.0f
              // Apply sqrt to convert energy to amplitude.
              val amplitude =
                message.inputLevel.map { energy -> Y_SCALE * sqrt(energy) }.toFloatArray()
              _inputLevel.add(System.currentTimeMillis(), amplitude)
            }
            is FlashMemoryStatusMessage -> {
              _flashMemoryWriteStatus.setValue(message.status)
            }
            else -> {
              // No action on other messages.
            }
          }
        }
      }
  }

  /** This is called by the main activity, passing the app build time as a Unix timestamp. */
  fun onActivityStart(appBuildTimestamp: Long) {
    if (appBuildDate == null) {
      appBuildDate = LocalDateTime.ofEpochSecond(appBuildTimestamp, 0, ZoneOffset.UTC).toLocalDate()
      log("App build date: $appBuildDate")
      log("Android version: ${Build.VERSION.RELEASE}")
    }
  }

  /** Adds a message to `logLines`. Also prints it to the console log. */
  fun log(message: String) {
    val timestamp = logTimestampFormat.format(Calendar.getInstance().getTime())
    _logLines.value!!.add(LogLine(timestamp, message))
    _logLines.value = _logLines.value // Notify observers.
    // Print to console log. This API includes a timestamp on its own, so just log the message.
    // TODO: Consider switching logging to Google Flogger.
    Log.i(TAG, message)
  }

  /** Initiates a scan for BLE devices. */
  fun scanForDevices(activity: Activity, launcher: ActivityResultLauncher<IntentSenderRequest>) {
    bleCom.scanForDevices(activity, launcher)
  }

  /** Initiates connection to `device` */
  fun connect(device: BluetoothDevice) {
    log("BLE connecting to ${device.name}")
    bleCom.connect(device)
  }

  /** Disconnects BLE, if connected. */
  fun disconnect() {
    bleCom.disconnect()
  }

  /** Tells the device to update the device name. */
  fun setDeviceName(name: String) {
    if (name.isNotEmpty()) {
      _deviceName.value?.let {
        // Before setting, check whether `name` differs to avoid infinite observer loop.
        if (it == name) { return }
        _deviceName.value = name

        sendMessageIfConnected(DeviceNameMessage(name))
      }
    }
  }

  /** Tells the device to play a tactile pattern. */
  fun playTactilePattern(pattern: ByteArray) {
    if (pattern.isNotEmpty()) {
      sendMessageIfConnected(TactilePatternMessage(pattern))
    }
  }

  /** Resets all tuning knobs to default settings. */
  fun tuningResetAll() {
    _tuning.value?.let {
      it.resetAll()
      tuningNotifyObservers()
    }
  }

  /** Resets the `knobIndex` tuning knob to its default. */
  fun tuningKnobReset(knobIndex: Int) {
    tuningKnobSetValue(knobIndex, Tuning.DEFAULT_TUNING_KNOBS[knobIndex])
  }

  /** Sets the `knobIndex` tuning knob to the given control value. */
  fun tuningKnobSetValue(knobIndex: Int, value: Int) {
    val knob = _tuning.value?.get(knobIndex) ?: return
    if (knob.value != value) {
      knob.value = value
      tuningNotifyObservers()
    }
  }

  /** Resets all channels to default settings. */
  fun channelMapResetAll() {
    _channelMap.value?.let {
      it.resetAll()
      channelMapNotifyObservers()
    }
  }

  /** Resets channel `tactorIndex` to its defaults. */
  fun channelMapReset(tactorIndex: Int) {
    val channel = _channelMap.value?.get(tactorIndex) ?: return
    channel.reset()
    channelMapNotifyObservers()
  }

  /** Sets whether channel `tactorIndex` is enabled. */
  fun channelMapSetEnable(tactorIndex: Int, enabled: Boolean) {
    val channel = _channelMap.value?.get(tactorIndex) ?: return
    channel.enabled = enabled
    channelMapNotifyObservers()
  }

  /** Sets the source as a base-0 index for channel `tactorIndex`. */
  fun channelMapSetSource(tactorIndex: Int, source: Int) {
    val channel = _channelMap.value?.get(tactorIndex) ?: return
    channel.source = source
    channelMapNotifyObservers()
  }

  /** Sets the gain for channel `tactorIndex` to the given control value. */
  fun channelMapSetGain(tactorIndex: Int, gain: Int) {
    val channel = _channelMap.value?.get(tactorIndex) ?: return
    channel.gain = gain

    _channelMap.value = _channelMap.value // Notify observers.
    _channelMap.value?.let {
      // Send [ChannelGainUpdateMessage] over BLE to the connected device.
      sendMessageIfConnected(ChannelGainUpdateMessage(it, Pair(0, tactorIndex)))
    }
  }

  /** Tells the device to play a test buzz on tactor `tactorIndex`. */
  fun channelMapTest(tactorIndex: Int) {
    _channelMap.value?.let {
      // Send [ChannelGainUpdateMessage] over BLE to the connected device.
      sendMessageIfConnected(ChannelGainUpdateMessage(it, Pair(tactorIndex, tactorIndex)))
    }
  }

  /** Command to prepare for bootloading, This means to stop all interrupts. */
  fun prepareForBootloading() {
    sendMessageIfConnected(PrepareForBluetoothBootloadingMessage())
  }

  /** Clears `tactilePattern` and sets the selected index to -1 (no selection). */
  fun tactilePatternClear() {
    _tactilePattern.value!!.ops.clear()
    _tactilePatternSelectedIndex.value = -1
  }

  /** Reads `tactilePattern` from an InputStream with a pattern in text format. */
  fun tactilePatternReadFromStream(stream: InputStream) {
    try {
      _tactilePatternSelectedIndex.value = -1
      _tactilePattern.setValue(TactilePattern.parse(String(stream.readBytes())))
    } catch (e: IOException) {
      log("Error reading tactile pattern.")
      e.message?.let { log(it) }
    } catch (e: TactilePattern.ParseException) {
      log("Error parsing tactile pattern.")
      e.message?.let { log(it) }
    }
  }

  /** Writes `tactilePattern` to an OutputStream in text format. */
  fun tactilePatternWriteToStream(stream: OutputStream) {
    try {
      stream.write(_tactilePattern.value!!.toString().toByteArray())
    } catch (e: IOException) {
      log("Error writing tactile pattern.")
      e.message?.let { log(it) }
    }
  }

  /** Sends `tactilePattern` to the device for playback. */
  fun tactilePatternPlay(): Boolean {
    return sendMessageIfConnected(TactileExPatternMessage(_tactilePattern.value!!))
  }

  /** Selects the ith line of `tactilePattern`, or deselects if out of range. */
  fun tactilePatternSelect(i: Int) {
    val newValue = if (i in _tactilePattern.value!!.ops.indices) { i } else { -1 /* Deselect. */ }
    if (_tactilePatternSelectedIndex.value!! != newValue) {
      _tactilePatternSelectedIndex.value = newValue
    }
  }

  /** If `op` is nonnull, inserts the op after the selected index, or at the end if no selection. */
  fun tactilePatternInsertOp(op: TactilePattern.Op?): Boolean {
    if (op == null) { return false }
    val ops = _tactilePattern.value!!.ops
    val selectedIndex = _tactilePatternSelectedIndex.value!!
    val i = if (selectedIndex in ops.indices) { selectedIndex + 1 } else { ops.size }
    ops.add(i, op)
    _tactilePatternSelectedIndex.value = i
    return true
  }

  /** Swaps the ith and jth ops of `tactilePattern`. Returns true on success. */
  fun tactilePatternSwapOps(i: Int, j: Int): Boolean {
    return _tactilePattern.value!!.swapOps(i, j)
  }

  /** Removes the selected op from `tactilePattern`. Returns the removed index on success. */
  fun tactilePatternRemoveOp(): Int? {
    val ops = _tactilePattern.value!!.ops
    val selectedIndex = _tactilePatternSelectedIndex.value!!
    if (selectedIndex in ops.indices) {
      ops.removeAt(selectedIndex)
      return selectedIndex
    } else {
      return null
    }
  }

  /** Notifies `tactilePattern` observers. */
  fun tactilePatternNotifyObservers() {
    _tactilePattern.notifyObservers()
  }

  /** Notifies `tactilePatternSelectIndex` observers. */
  fun tactilePatternSelectedIndexNotifyObservers() {
    _tactilePatternSelectedIndex.notifyObservers()
  }

  @VisibleForTesting
  fun channelMapSet(channelMap: ChannelMap) {
    _channelMap.setValue(channelMap)
  }

  private fun tuningNotifyObservers() {
    _tuning.notifyObservers()
    // Send [TuningMessage] over BLE to update tuning on the connected device.
    _tuning.value?.let { sendMessageIfConnected(TuningMessage(it)) }
  }

  private fun channelMapNotifyObservers() {
    _channelMap.notifyObservers()
    // Send [ChannelMapMessage] over BLE to update tuning on the connected device.
    _channelMap.value?.let { sendMessageIfConnected(ChannelMapMessage(it)) }
  }

  /** Sends and logs `message` if BLE is connected. */
  private fun sendMessageIfConnected(message: Message): Boolean {
    if (isConnected.value == true) {
      bleCom.write(message)
      log("Sent $message")
      return true
    }
    return false
  }

  private companion object {
    const val TAG = "BleViewModel"
  }
}
