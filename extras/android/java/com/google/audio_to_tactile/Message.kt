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

import androidx.annotation.StringRes
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.DateTimeException
import java.time.LocalDate
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/** Types of messages. Type values must be between 0 and 255. */
const val MESSAGE_TYPE_TEMPERATURE = 16
const val MESSAGE_TYPE_TUNING = 18
const val MESSAGE_TYPE_TACTILE_PATTERN = 19
const val MESSAGE_TYPE_GET_TUNING = 20
const val MESSAGE_TYPE_CHANNEL_MAP = 24
const val MESSAGE_TYPE_GET_CHANNEL_MAP = 25
const val MESSAGE_TYPE_STATS_RECORD = 26
const val MESSAGE_TYPE_CHANNEL_GAIN_UPDATE = 27
const val MESSAGE_TYPE_BATTERY_VOLTAGE = 28
const val MESSAGE_TYPE_DEVICE_NAME = 29
const val MESSAGE_TYPE_GET_DEVICE_NAME = 30
const val MESSAGE_TYPE_PREPARE_FOR_BLUETOOTH_BOOTLOADING = 31
const val MESSAGE_TYPE_FLASH_WRITE_STATUS = 32
const val MESSAGE_TYPE_ON_CONNECTION_BATCH = 33
const val MESSAGE_TYPE_GET_ON_CONNECTION_BATCH = 34
const val MESSAGE_TYPE_TACTILE_EX_PATTERN = 36

/**
 * `Message` is a unified representation for sending commands and information between devices. This
 * Kotlin library, which runs in the Audio-to-Tactile Android app, interoperates with the C++
 * library src/cpp/message.h, which runs on the wearable's microcontroller. So these two libraries
 * must be consistent.
 *
 * (Note: In principle, we could reuse the C++ Message implementation through JNI bindings. However,
 * there is a lot of "surface" to the interface to represent each kind of message and their fields,
 * which would all need to be marshalled through the bindings. It is simpler and less error-prone to
 * reimplement the protocol in pure Kotlin, and of course use good unit tests.)
 *
 * Messages are encoded in the following format:
 *
 * - bytes 0-1: two-byte Fletcher-16 checksum
 * - byte 2: message type code
 * - byte 3: payload size
 * - bytes 4-n: payload
 *
 * All data is serialized in little endian order.
 */
sealed class Message {
  // Derived classes implement `type`, `payloadSize`, and `serializePayload`.

  /** Type of the message. */
  abstract val type: Int

  /**
   * Payload size in bytes, between 0 and [MAX_PAYLOAD_SIZE]. This need not be constant, it may be a
   * function of the payload contents.
   */
  abstract val payloadSize: Int

  /**
   * Method to write the payload into `buffer`. The supplied buffer has position() set to the
   * beginning of the payload, has space for exactly `payloadSize` bytes remaining, and has byte
   * order set to little endian.
   */
  abstract fun serializePayload(buffer: ByteBuffer)

  /** Method to print message summary in a human-readable form, useful for logging. */
  override fun toString(): String = this::class.java.simpleName

  /** Converts the Message to a serialized ByteArray. */
  fun serialize(): ByteArray {
    val payloadSize = payloadSize
    require(payloadSize in 0..MAX_PAYLOAD_SIZE) { "Invalid payload size: $payloadSize" }

    val bytes = ByteArray(HEADER_SIZE + payloadSize)
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

    // Write type and payload size fields in the header.
    buffer.position(2)
    buffer.put(type.toByte())
    buffer.put(payloadSize.toByte())
    // Write the payload.
    serializePayload(buffer)

    // Write checksum at the beginning.
    buffer.position(0)
    buffer.putUint16(computeChecksum(bytes))

    return bytes
  }

  companion object {
    /** Number of bytes in the message header: 16-bit checksum, 8-bit type, 8-bit payload size. */
    const val HEADER_SIZE = 4

    /**
     * The max supported payload size. This is constrained on the firmware side to a payload of 124
     * bytes (a 128-byte buffer holds the message plus a 4-byte header).
     */
    const val MAX_PAYLOAD_SIZE = 124

    /** Deserializes a Message from `bytes`. Returns null if the message is invalid. */
    fun deserialize(bytes: ByteArray): Message? {
      if (bytes.size < HEADER_SIZE) { return null } // A valid message is at least HEADER_SIZE long.

      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      // Read the message header fields.
      val checksum = buffer.getUint16()
      val type = buffer.get().toNonnegInt()
      val payloadSize = buffer.get().toNonnegInt()
      // At this point, buffer.position() refers to the beginning of the (possibly empty) payload.
      if (buffer.remaining() != payloadSize || checksum != computeChecksum(bytes)) { return null }

      if (payloadSize == 0) {
        // Shortcut: some message types are header-only and have an empty payload, so we don't need
        // a `deserializePayload` method.
        return when (type) {
          MESSAGE_TYPE_GET_TUNING -> GetTuningMessage()
          MESSAGE_TYPE_GET_CHANNEL_MAP -> GetChannelMapMessage()
          MESSAGE_TYPE_GET_DEVICE_NAME -> GetChannelMapMessage()
          MESSAGE_TYPE_PREPARE_FOR_BLUETOOTH_BOOTLOADING -> GetChannelMapMessage()
          else -> null
        }
      } else {
        // Handle message types with nonempty payloads.
        return when (type) {
          MESSAGE_TYPE_TEMPERATURE -> TemperatureMessage.deserializePayload(buffer)
          MESSAGE_TYPE_BATTERY_VOLTAGE -> BatteryVoltageMessage.deserializePayload(buffer)
          MESSAGE_TYPE_TACTILE_PATTERN -> TactilePatternMessage.deserializePayload(buffer)
          MESSAGE_TYPE_TUNING -> TuningMessage.deserializePayload(buffer)
          MESSAGE_TYPE_CHANNEL_MAP -> ChannelMapMessage.deserializePayload(buffer)
          MESSAGE_TYPE_CHANNEL_GAIN_UPDATE -> ChannelGainUpdateMessage.deserializePayload(buffer)
          MESSAGE_TYPE_STATS_RECORD -> StatsRecordMessage.deserializePayload(buffer)
          MESSAGE_TYPE_DEVICE_NAME -> DeviceNameMessage.deserializePayload(buffer)
          MESSAGE_TYPE_FLASH_WRITE_STATUS -> FlashMemoryStatusMessage.deserializePayload(buffer)
          MESSAGE_TYPE_ON_CONNECTION_BATCH -> OnConnectionBatchMessage.deserializePayload(buffer)
          MESSAGE_TYPE_TACTILE_EX_PATTERN -> TactileExPatternMessage.deserializePayload(buffer)
          else -> null
        }
      }
    }

    /** Computes Fletcher-16 checksum over all of `bytes` except for the first two bytes. */
    fun computeChecksum(bytes: ByteArray): Int {
      val buffer = ByteBuffer.wrap(bytes)
      buffer.position(2)
      return buffer.fletcher16()
    }
  }
}

/** Base class for messages that are header-only with an empty payload. */
sealed class HeaderOnlyMessage(override val type: Int) : Message() {
  final override val payloadSize = 0
  final override fun serializePayload(buffer: ByteBuffer) {}
}

/** Message to request the current tuning knob settings. */
class GetTuningMessage : HeaderOnlyMessage(MESSAGE_TYPE_GET_TUNING)

/** Message to request the ChannelMap configuration. */
class GetChannelMapMessage : HeaderOnlyMessage(MESSAGE_TYPE_GET_CHANNEL_MAP)

/** Message to request the device name. */
class GetDeviceNameMessage : HeaderOnlyMessage(MESSAGE_TYPE_GET_DEVICE_NAME)

/** Message to prepare for bluetooth bootloading. */
class PrepareForBluetoothBootloadingMessage :
  HeaderOnlyMessage(MESSAGE_TYPE_PREPARE_FOR_BLUETOOTH_BOOTLOADING)

/** Message to request an on-connection batch message. */
class GetOnConnectionBatchMessage : HeaderOnlyMessage(MESSAGE_TYPE_GET_ON_CONNECTION_BATCH)

/** Message to send temperature, encoded in Celsius as a 32-bit float. */
class TemperatureMessage(val celsius: Float) : Message() {
  override val type = MESSAGE_TYPE_TEMPERATURE
  override val payloadSize = 4

  override fun serializePayload(buffer: ByteBuffer) {
    buffer.putFloat(celsius)
  }

  /**
   * Converts the TemperatureMessage to a string for logging.
   *
   * NOTE: String interpolation with "$temperature" would show 7 or so digits, which is excessive
   * precision. We use %.1f to round the display to the tenth of a degree.
   */
  override fun toString() = "TemperatureMessage(%.1f C)".format(celsius)

  internal companion object {
    fun deserializePayload(buffer: ByteBuffer): TemperatureMessage? {
      if (buffer.remaining() != 4) { return null }
      return TemperatureMessage(buffer.getFloat())
    }
  }
}

/** Message to send battery voltage, encoded as a 32-bit float. */
class BatteryVoltageMessage(val voltage: Float) : Message() {
  override val type = MESSAGE_TYPE_BATTERY_VOLTAGE
  override val payloadSize = 4

  override fun serializePayload(buffer: ByteBuffer) {
    buffer.putFloat(voltage)
  }

  override fun toString() = "BatteryVoltageMessage(%.2f V)".format(voltage)

  internal companion object {
    fun deserializePayload(buffer: ByteBuffer): BatteryVoltageMessage? {
      if (buffer.remaining() != 4) { return null }
      return BatteryVoltageMessage(buffer.getFloat())
    }
  }
}

/** Message with a tactile pattern. The pattern syntax is as in src/tactile/tactile_pattern.c. */
class TactilePatternMessage(val pattern: ByteArray) : Message() {
  override val type = MESSAGE_TYPE_TACTILE_PATTERN
  override val payloadSize = pattern.size

  override fun serializePayload(buffer: ByteBuffer) {
    buffer.put(pattern)
  }

  override fun toString() = "TactilePatternMessage(\"${String(pattern)}\")"

  internal companion object {
    fun deserializePayload(buffer: ByteBuffer): TactilePatternMessage? {
      val payload = ByteArray(buffer.remaining()).also { buffer.get(it) }
      return TactilePatternMessage(payload)
    }
  }
}

/** Message with tuning knob settings. */
class TuningMessage(val tuning: Tuning) : Message() {
  override val type = MESSAGE_TYPE_TUNING
  override val payloadSize = Tuning.NUM_TUNING_KNOBS
  override fun serializePayload(buffer: ByteBuffer) {
    for (i in tuning.indices) { // Convert `.values` ints to bytes and write them into `buffer`.
      buffer.put(tuning.values[i].toByte())
    }
  }

  internal companion object {
    fun deserializePayload(buffer: ByteBuffer): TuningMessage? {
      if (buffer.remaining() != Tuning.NUM_TUNING_KNOBS) { return null }
      // Convert payload bytes to an array of ints in 0-255.
      val values = IntArray(Tuning.NUM_TUNING_KNOBS) { buffer.get().toNonnegInt() }
      return TuningMessage(Tuning(values))
    }
  }
}

/** Message with all channel map settings. */
class ChannelMapMessage(val channelMap: ChannelMap) : Message() {
  override val type = MESSAGE_TYPE_CHANNEL_MAP
  override val payloadSize = computePayloadSize(channelMap.numOutputChannels)
  override fun serializePayload(buffer: ByteBuffer) {
    // Write number of input and output channels in the first byte.
    buffer.put(Byte.fromNibbles(channelMap.numInputChannels % 16, channelMap.numOutputChannels))

    // Write source mapping, 4 bits per channel.
    for (c in 0 until channelMap.numOutputChannels step 2) {
      buffer.put(Byte.fromNibbles(channelMap[c].source, channelMap[c + 1].source))
    }

    serializeGains(channelMap, buffer)
  }

  internal companion object {
    fun computePayloadSize(numOut: Int) = 1 + 3 * ((numOut + 3) / 4) + (numOut + 1) / 2

    fun deserializePayload(buffer: ByteBuffer): ChannelMapMessage? {
      val channelMap = deserializeNumChannels(buffer, ::computePayloadSize) ?: return null
      deserializeSources(buffer, channelMap)
      deserializeGains(buffer, channelMap)

      return ChannelMapMessage(channelMap)
    }

    /**
     * Deserializes the number of channels and returns null if they mismatch the payload size. This
     * function is also called by ChannelGainUpdateMessage, where the payload has a different size,
     * so the [payloadSizeFun] arg is set according to the kind of message.
     */
    fun deserializeNumChannels(buffer: ByteBuffer, payloadSizeFun: (Int) -> Int): ChannelMap? {
      val payloadSize = buffer.remaining()
      if (payloadSize < 1) { return null }

      val firstByte = buffer.get()
      val numIn = 1 + ((firstByte.lowerNibble() + 15) % 16)
      val numOut = firstByte.upperNibble()
      if (numOut !in 1..12 || payloadSize != payloadSizeFun(numOut)) { return null }

      return ChannelMap(numIn, numOut)
    }

    /** Reads the source mapping, 4 bits per channel. */
    fun deserializeSources(buffer: ByteBuffer, channelMap: ChannelMap) {
      val validSourceRange = 0 until channelMap.numInputChannels
      for (c in 0 until channelMap.numOutputChannels step 2) {
        val byte = buffer.get()
        channelMap[c].source = byte.lowerNibble().coerceIn(validSourceRange)
        if (c + 1 < channelMap.numOutputChannels) {
          channelMap[c + 1].source = byte.upperNibble().coerceIn(validSourceRange)
        }
      }
    }

    /** Reads channel gains. This is used also by ChannelGainUpdateMessage. */
    fun deserializeGains(buffer: ByteBuffer, channelMap: ChannelMap) {
      val numOut = channelMap.numOutputChannels
      for (i in 0 until numOut step 4) { // Read gains, 6 bits per channel.
        var pack24 = buffer.getUint24()
        for (j in 0 until min(numOut - i, 4)) {
          channelMap[i + j].gain = pack24 and 63
          pack24 = pack24 shr 6
        }
      }
    }

    /** Writes channel gains. This is used also by ChannelGainUpdateMessage. */
    fun serializeGains(channelMap: ChannelMap, buffer: ByteBuffer) {
      for (c in 0 until channelMap.numOutputChannels step 4) {
        val pack24 =
            (channelMap[c].gain or // Each gain is a 6-bit value in 0-63.
             (channelMap[c + 1].gain shl 6) or
             (channelMap[c + 2].gain shl 12) or
             (channelMap[c + 3].gain shl 18))
        buffer.putUint24(pack24)
      }
    }
  }
}

/** Message of channel map gains, but not sources, and indices of two test channels. */
class ChannelGainUpdateMessage(val channelMap: ChannelMap, val testChannels: Pair<Int, Int>) :
  Message() {
  override val type = MESSAGE_TYPE_CHANNEL_GAIN_UPDATE
  override val payloadSize = computePayloadSize(channelMap.numOutputChannels)
  override fun serializePayload(buffer: ByteBuffer) {
    // Write number of input and output channels in the first byte.
    buffer.put(Byte.fromNibbles(channelMap.numInputChannels % 16, channelMap.numOutputChannels))

    // Write the two test channel indices, 4 bits for each.
    buffer.put(Byte.fromNibbles(testChannels.first, testChannels.second))

    ChannelMapMessage.serializeGains(channelMap, buffer)
  }

  internal companion object {
    private fun computePayloadSize(numOut: Int) = 1 + 3 * ((numOut + 3) / 4) + 1

    fun deserializePayload(buffer: ByteBuffer): ChannelGainUpdateMessage? {
      val channelMap =
        ChannelMapMessage.deserializeNumChannels(buffer, ::computePayloadSize) ?: return null
      val testChannels = deserializeTestChannels(buffer)
      ChannelMapMessage.deserializeGains(buffer, channelMap)

      return ChannelGainUpdateMessage(channelMap, testChannels)
    }

    /** Reads the two test channel indices, 4 bits each. */
    private fun deserializeTestChannels(buffer: ByteBuffer): Pair<Int, Int> {
      val byte = buffer.get()
      return Pair(byte.lowerNibble(), byte.upperNibble())
    }
  }
}

/**
 * Message sending a sequence of input level measurements. These measurements are collected on the
 * device and displayed in the Android app as a scrolling plot. For details on the encoding, see
 * src/tactile/envelope_tracker.c.
 */
class StatsRecordMessage(val inputLevel: FloatArray) : Message() {
  init {
    require(inputLevel.size == NUM_POINTS) {
      "inputLevel must have size $NUM_POINTS"
    }
  }

  override val type = MESSAGE_TYPE_STATS_RECORD
  override val payloadSize = NUM_BYTES

  override fun serializePayload(buffer: ByteBuffer) {
    var cumulative = encodeEnergy(inputLevel[0])
    buffer.put(cumulative.toByte())

    // Each iteration in i handles 8 measurements, encoding them as 24 bits.
    for (i in 1 until NUM_POINTS step 8) {
      var pack24 = 0
      for (j in 0..7) {
        // Encode the delta between the current measurement and `cumulative`.
        val code = encodeDelta(encodeEnergy(inputLevel[i + j]) - cumulative)
        // To account for error, decode the delta and add it to `cumulative`.
        cumulative += decodeDelta(code)
        // Store the 3-bit code in little endian order.
        pack24 += code shl (3 * j)
      }

      buffer.putUint24(pack24)
    }
  }

  companion object {
    /**
     * Number of points in inputLevel. Note: the current implementation relies on that this number
     * is 1 + a multiple of 8. Otherwise, encoding would involve a partially full final byte.
     */
    const val NUM_POINTS = 33

    /** Message payload size: 1 byte for first point + 3 bits per subsequent point. */
    private const val NUM_BYTES = (1 + (NUM_POINTS - 1) * 3 / 8)

    /** Encodes `energy` as a value between 0 and 255. */
    private fun encodeEnergy(energy: Float): Int {
      val value = energy.pow(1.0f / 6.0f).coerceIn(0.0f, 1.0f)
      return (255.0f * value).roundToInt()
    }

    /** Decodes an energy value, the reverse of EncodeEnergy(). */
    private fun decodeEnergy(value: Int) = (value / 255.0f).pow(6.0f)

    /**
     * Encodes an integer delta as a 3-bit code between 0 and 7. The correspodence between codes and
     * deltas is:
     *
     * ```
     * code  delta      code  delta
     *    0      0         4      0 (redundant with code 0)
     *    1     +1         5     -1
     *    2     +4         6     -4
     *    3    +11         7    -11
     * ```
     */
    private fun encodeDelta(delta: Int): Int {
      var code = ENCODE_DELTA_TABLE[abs(delta).coerceAtMost(8)]
      if (delta < 0) { code += 4 }
      return code
    }

    private val ENCODE_DELTA_TABLE = intArrayOf(0, 1, 1, 2, 2, 2, 2, 2, 3)

    /** Decodes a delta, the reverse of EncodeDelta(). */
    private fun decodeDelta(code: Int) = DECODE_DELTA_TABLE[code]

    private val DECODE_DELTA_TABLE = intArrayOf(0, 1, 4, 11, 0, -1, -4, -11)

    internal fun deserializePayload(buffer: ByteBuffer): StatsRecordMessage? {
      if (buffer.remaining() != NUM_BYTES) { return null }

      var cumulative = buffer.get().toNonnegInt()
      val inputLevel = FloatArray(NUM_POINTS)
      inputLevel[0] = decodeEnergy(cumulative)

      for (i in 1 until NUM_POINTS step 8) {
        var pack24 = buffer.getUint24()
        for (j in 0..7) {
          // Get the next 3-bit delta code, decode it, and add to `cumulative`.
          cumulative += decodeDelta(pack24 and 7)
          inputLevel[i + j] = decodeEnergy(cumulative)
          pack24 = pack24 shr 3
        }
      }

      return StatsRecordMessage(inputLevel)
    }
  }
}

/** Message with the device name. */
class DeviceNameMessage(val name: String) : Message() {
  val nameBytes = name.toByteArray()

  init {
    require(nameBytes.size <= MAX_NAME_BYTES) {
      "name must have at most $MAX_NAME_BYTES bytes, got: ${nameBytes.size}"
    }
  }

  override val type = MESSAGE_TYPE_DEVICE_NAME
  override val payloadSize = nameBytes.size

  override fun serializePayload(buffer: ByteBuffer) {
    buffer.put(nameBytes)
  }

  override fun toString() = "DeviceNameMessage(\"$name\")"

  companion object {
    /** Max supported device name length in bytes. */
    const val MAX_NAME_BYTES = 16

    internal fun deserializePayload(buffer: ByteBuffer): DeviceNameMessage? {
      if (buffer.remaining() > MAX_NAME_BYTES) { return null }
      val payload = ByteArray(buffer.remaining()).also { buffer.get(it) }
      return DeviceNameMessage(String(payload))
    }
  }
}

enum class FlashMemoryStatus(val value: Int, @StringRes val stringRes: Int) {
  /** Catch-all case for when a BLE operation failed, but we don't have a specific explanation. */
  SUCCESS(0, R.string.flash_write_success),
  /**
   * Catch-all case for when a flash write operation failed, but we don't have a specific
   * explanation.
   */
  UNKNOWN_ERROR(1, R.string.flash_write_failed_unknown),
  /** Catch-all case for when a BLE operation failed, but we don't have a specific explanation. */
  FLASH_NOT_FORMATTED_ERROR(2, R.string.flash_write_failed_not_formatted);

  companion object {
    fun fromInt(value: Int) =
      FlashMemoryStatus.values().firstOrNull { it.value == value } ?: UNKNOWN_ERROR
  }
}

/** Message with flash memory write status. */
class FlashMemoryStatusMessage(val status: FlashMemoryStatus) : Message() {
  override val type = MESSAGE_TYPE_FLASH_WRITE_STATUS
  override val payloadSize = 1

  override fun serializePayload(buffer: ByteBuffer) {
    buffer.put(status.value.toByte())
  }

  override fun toString() = "FlashMemoryStatusMessage($status)"

  internal companion object {
    fun deserializePayload(buffer: ByteBuffer): FlashMemoryStatusMessage? {
      if (buffer.remaining() != 1) {
        return null
      }
      return FlashMemoryStatusMessage(FlashMemoryStatus.fromInt(buffer.get().toInt()))
    }
  }
}

/**
 * Message sent from device to app on connection, batching:
 *  - firmware build date
 *  - battery voltage
 *  - temperature
 *  - device name
 *  - tuning settings
 *  - channel map
 */
class OnConnectionBatchMessage(
  val firmwareBuildDate: LocalDate?,
  val batteryVoltage: Float,
  val temperatureCelsius: Float,
  val deviceName: String?,
  val tuning: Tuning?,
  val channelMap: ChannelMap?,
) : Message() {
  override val type = MESSAGE_TYPE_ON_CONNECTION_BATCH
  override val payloadSize =
    FIXED_FIELDS_SIZE +
    1 + (deviceName?.let { deviceName.toByteArray().size } ?: 0) +
    1 + (tuning?.let { Tuning.NUM_TUNING_KNOBS } ?: 0) +
    1 + (channelMap?.let { ChannelMapMessage.computePayloadSize(it.numOutputChannels) } ?: 0)

  override fun serializePayload(buffer: ByteBuffer) {
    serializeDate(buffer, firmwareBuildDate)
    buffer.putFloat(batteryVoltage)
    buffer.putFloat(temperatureCelsius)

    // The following data are formatted in a block structure beginning with a byte
    // indicating the size of the block followed by the block data. This allows
    // the reader to skip individual blocks, in case of version mismatch or other
    // error, rather than failing to read the message entirely.
    for (block in
      arrayOf<Message?>(
        deviceName?.let { DeviceNameMessage(deviceName) },
        tuning?.let { TuningMessage(tuning) },
        channelMap?.let { ChannelMapMessage(channelMap) },
      )) {
      if (block != null) {
        val payloadSize = block.payloadSize
        buffer.put(payloadSize.toByte()) // Write the size of the block.
        block.serializePayload(buffer.slice().apply {
          limit(payloadSize)
          order(ByteOrder.LITTLE_ENDIAN)
        }) // Write block itself.
        buffer.position(buffer.position() + payloadSize)
      } else {
        buffer.put(0) // Write an empty block.
      }
    }
  }

  internal companion object {
    private const val FIXED_FIELDS_SIZE = 12

    fun deserializePayload(buffer: ByteBuffer): Message? {
      if (buffer.remaining() < FIXED_FIELDS_SIZE) { return null }
      return OnConnectionBatchMessage(
        firmwareBuildDate = deserializeDate(buffer),
        batteryVoltage = buffer.getFloat(),
        temperatureCelsius = buffer.getFloat(),
        deviceName = deserializeBlock(buffer) { DeviceNameMessage.deserializePayload(it)?.name },
        tuning = deserializeBlock(buffer) { TuningMessage.deserializePayload(it)?.tuning },
        channelMap =
          deserializeBlock(buffer) { ChannelMapMessage.deserializePayload(it)?.channelMap },
      )
    }

    /** Deserializes one block within the payload, using `reader` to read the block contents. */
    private fun <T> deserializeBlock(buffer: ByteBuffer, reader: (ByteBuffer) -> T?): T? {
      if (buffer.remaining() >= 1) {
        val payloadSize = buffer.get().toInt() // Read the size of the block.
        if (buffer.remaining() >= payloadSize) {
          val result = reader(buffer.slice().apply {
            limit(payloadSize)
            order(ByteOrder.LITTLE_ENDIAN)
          }) // Read block itself.
          buffer.position(buffer.position() + payloadSize)
          return result
        }
      }
      return null
    }

    /** Serializes a date. E.g. 2021-10-13 becomes integer 20211013, which is then serialized. */
    private fun serializeDate(buffer: ByteBuffer, date: LocalDate?) {
      buffer.putInt(date?.let { 10000 * it.year + 100 * it.monthValue + it.dayOfMonth } ?: 0)
    }

    /** Deserializes a date as written by `serializeDate`. */
    private fun deserializeDate(buffer: ByteBuffer): LocalDate? {
      val code = buffer.getInt()
      val year = code / 10000
      val month = (code / 100) % 100
      val dayOfMonth = code % 100
      return try { LocalDate.of(year, month, dayOfMonth) } catch (e: DateTimeException) { null }
    }
  }
}

/** Message with a tactile extended pattern as serialized by TactilePattern.kt. */
class TactileExPatternMessage(val pattern: TactilePattern) : Message() {
  override val type = MESSAGE_TYPE_TACTILE_EX_PATTERN
  private val payload: ByteArray = pattern.serialize()
  override val payloadSize = payload.size

  override fun serializePayload(buffer: ByteBuffer) {
    buffer.put(payload)
  }

  override fun toString() = "TactileExPatternMessage(${pattern.ops.size} ops)"

  internal companion object {
    fun deserializePayload(buffer: ByteBuffer): TactileExPatternMessage? {
      val payload = ByteArray(buffer.remaining()).also { buffer.get(it) }
      val pattern = TactilePattern.deserialize(payload) ?: return null
      return TactileExPatternMessage(pattern)
    }
  }
}
