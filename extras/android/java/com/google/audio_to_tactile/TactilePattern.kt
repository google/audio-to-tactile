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

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Representation of tactile patterns playable by the tactile_pattern C library. Patterns can be
 * converted to and from a human-readable text representation for editing and converted to and from
 * the binary representation understood by the C library.
 *
 * Text representation of an example pattern (# lines are comments):
 *
 *     # Set waveform on channel 2 (channels are indexed starting from 1).
 *     wave 2   chirp
 *     # Play for 100 ms.
 *     play 100
 *     # Set gain on all channels (linear gain in [0, 1]).
 *     gain all 0.5
 *     # Move channel 2 to channel 5.
 *     move 2   5
 *
 * Each line represents one op. The line begins with the op name followed by the op args, separated
 * by spaces. Whitespace is insignificant. Indentation, blank lines, and the exact number of spaces
 * between args does not affect meaning.
 *
 *     Op       Args               Description
 *     ---------------------------------------------------------------------------------------------
 *     play     duration           Tells the synthesizer to generate samples for a duration before
 *                                 the next op. `duration` is in units of ms and between 20 and 640.
 *
 *     wave     channel waveform   Sets the waveform on the specified channel. `channel` is an int
 *                                 1-16 or "all" to set all channels. `waveform` is one of the
 *                                 waveforms named below.
 *
 *     gain     channel gain       Sets the gain on the specified channel. `channel` is an int 1-16
 *                                 or "all" to set all channels. `gain` is a float in [0.0, 1.0].
 *
 *     move     from to            For movement patterns, moves a waveform from channel `from` to
 *                                 channel `to`, where `from` and `to` are ints 1-16.
 *
 *     end      (none)             Marks the ends the pattern. See also the note below.
 *     ---------------------------------------------------------------------------------------------
 *
 *     Waveform                    Description
 *     ---------------------------------------------------------------------------------------------
 *     sin_25_hz                   Pure sinusoidal tones. The tones range from 25 to 350 Hz,
 *     sin_30_hz                   covering the range where both hardware and tactile perception are
 *     sin_35_hz                   responsive, and with a median increment between successive tones
 *     sin_45_hz                   of 20%, comparable to the 20-30% JND for frequency change.
 *     sin_50_hz
 *     sin_60_hz
 *     sin_70_hz
 *     sin_90_hz
 *     sin_100_hz
 *     sin_125_hz
 *     sin_150_hz
 *     sin_175_hz
 *     sin_200_hz
 *     sin_250_hz
 *     sin_300_hz
 *     sin_350_hz
 *
 *     chirp                       A rising exponential chirp.
 *     ---------------------------------------------------------------------------------------------
 *
 * NOTE: The C library requires an explicit "end" op to know where the pattern ends, but this isn't
 * required for patterns in this library---the end is implicit from the size of the container.
 * When communicating patterns to the C library, "end" should be appended where needed.
 *
 * See src/tactile/tactile_pattern.h for details on the binary representation.
 */
class TactilePattern {
  /** The list of ops that define the pattern. */
  var ops = mutableListOf<Op>()
  /** Number of ops in the pattern. */
  val size: Int
    get() = ops.size

  /** Converts pattern to human-readable text representation. */
  override fun toString() = ops.joinToString("\n")

  /** Converts pattern to the binary representation understood by the C library. */
  fun serialize(): ByteArray {
    val bytes = ByteArray(ops.sumOf { it.serializedSize })
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    ops.forEach { it.serialize(buffer) }
    return bytes
  }

  /** Swaps ith and jth ops if i and j are distict and valid indices. Returns true on success. */
  fun swapOps(i: Int, j: Int): Boolean {
    ops.let { ops ->
      if (i != j && i in ops.indices && j in ops.indices) {
        ops[i] = ops[j].also { ops[j] = ops[i] }
        return true
      }
    }
    return false
  }

  /**
   * Checks whether the pattern has at least a `wave` op followed by a `play` op. This is a
   * necessary condition for the pattern to produce nonzero output.
   */
  fun hasPlayAfterWave(): Boolean {
    val i = ops.indexOfFirst { it is SetWaveformOp }
    if (i == -1) { return false }
    return ops.drop(i).find { it is PlayOp } != null
  }

  companion object {
    /**
     * Parses a `TactilePattern` from the text representation described in the top-level comment
     * above. Throws `ParseError` with a diagnostic message on failure.
     */
    fun parse(string: String): TactilePattern {
      val pattern = TactilePattern()

      var lineNumber = 0
      for (untrimmedLine in string.split("\n")) {
        lineNumber++
        val line = untrimmedLine.trim() // Trim any indentation or trailing whitespace.
        if (line.isEmpty() or line.startsWith("#")) { continue } // Skip comments and blank lines.

        val op = try {
          Op.parse(line)
        } catch (e: ParseException) {
          // If Op.parse throws, catch the exception and prepend the line number to the message.
          throw ParseException("Line $lineNumber: ${e.message}", e)
        }

        if (op is EndOp) { break }
        pattern.ops.add(op)
      }

      return pattern
    }

    /** Deserializes a `TactilePattern` from binary representation. Returns null on failure. */
    fun deserialize(bytes: ByteArray): TactilePattern? {
      val pattern = TactilePattern()

      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      while (buffer.remaining() > 0) {
        val op = Op.deserialize(buffer) ?: return null
        if (op is EndOp) { break }
        pattern.ops.add(op)
      }

      return pattern
    }

    /** Max number of channels supported by the representation. */
    const val MAX_CHANNELS = 16

    /** Op codes for binary representation. */
    private const val OP_END = 0x00
    private const val OP_PLAY = 0x80
    private const val OP_SET_WAVEFORM = 0xa0
    private const val OP_SET_GAIN = 0xb0
    private const val OP_SET_ALL_WAVEFORM = 0x01
    private const val OP_SET_ALL_GAIN = 0x02
    private const val OP_MOVE = 0x03

    /** Op names for text representation. */
    private const val OP_NAME_END = "end"
    private const val OP_NAME_PLAY = "play"
    private const val OP_NAME_SET_WAVEFORM = "wave"
    private const val OP_NAME_SET_GAIN = "gain"
    private const val OP_NAME_MOVE = "move"
  }

  /** Base class for all ops. See `PlayOp`, `SetWaveformOp`, `SetGainOp`, `MoveOp` below. */
  sealed class Op {
    /** Size in bytes of the op when serialized. */
    abstract val serializedSize: Int

    /** Serializes the op to `buffer`. */
    abstract fun serialize(buffer: ByteBuffer)

    companion object {
      /** Parses an `Op` give one line of text. Throws ParseException on failure. */
      fun parse(line: String): Op {
        val tokens = line.lowercase().split(" ").filter { it.isNotEmpty() }
        if (tokens.isEmpty()) { throw ParseException("Empty input") }

        val opName = tokens[0]
        val args = tokens.drop(1)
        return when (opName) {
          OP_NAME_END -> EndOp()
          OP_NAME_PLAY -> PlayOp.parse(args)
          OP_NAME_SET_WAVEFORM -> SetWaveformOp.parse(args)
          OP_NAME_SET_GAIN -> SetGainOp.parse(args)
          OP_NAME_MOVE -> MoveOp.parse(args)
          else -> throw ParseException("Invalid op \"$opName\"")
        }
      }

      /** Deserializes an `Op` from `buffer`, starting from its current .position(). */
      fun deserialize(buffer: ByteBuffer): Op? {
        if (buffer.remaining() < 1) { return null }
        val opcode = buffer.get().toNonnegInt()

        // Opcodes >= 0x80 indicate an action with the high bits and a parameter (usually a channel
        // index) with the lower bits.
        return when (if (opcode >= 0x80) { opcode and 0xf0 } else { opcode }) {
          OP_END -> EndOp()
          OP_PLAY, OP_PLAY + 0x10 -> PlayOp.deserialize(opcode)
          OP_SET_WAVEFORM, OP_SET_ALL_WAVEFORM -> SetWaveformOp.deserialize(opcode, buffer)
          OP_SET_GAIN, OP_SET_ALL_GAIN -> SetGainOp.deserialize(opcode, buffer)
          OP_MOVE -> MoveOp.deserialize(buffer)
          else -> null
        }
      }

      /**
       * Converts channel to string. `c` is assumed to be a base-0 index, and is coverted by this
       * function to base-1 index.
       */
      fun channelToString(c: Int) = if (c < 0) { "all" } else { (c + 1).toString().padEnd(3) }

      /** Parses a channel string; the reverse of `channelToString`. */
      fun parseChannel(s: String): Int {
        if (s == "all") { return -1 }
        val channel = try { s.toInt() - 1 } catch (e: NumberFormatException) { null }
        if (channel != null && channel in 0 until MAX_CHANNELS) { return channel }
        throw ParseException("Invalid channel \"$s\"")
      }
    }
  }

  /** "End" op, representing the end of the pattern. */
  class EndOp() : Op() {
    override fun toString(): String = OP_NAME_END

    override val serializedSize = 1

    override fun serialize(buffer: ByteBuffer) {
      buffer.put(OP_END.toByte())
    }
  }

  /** "Play" op: tells the synthesizer to generate samples for a duration before the next op. */
  class PlayOp(durationMsIn: Int) : Op() {
    private var _durationMs: Int = constrainDuration(durationMsIn)
    /** Duration in milliseconds, constrained to the range 20--640 ms. */
    var durationMs: Int
      get() = _durationMs
      set(newDuration) {
        _durationMs = constrainDuration(newDuration)
      }

    override fun toString() = "$OP_NAME_PLAY $durationMs"

    override val serializedSize = 1

    override fun serialize(buffer: ByteBuffer) {
      val opcode = OP_PLAY + encodeDuration(durationMs)
      buffer.put(opcode.toByte())
    }

    internal companion object {
      fun parse(args: List<String>): PlayOp {
        if (args.size != 1) { throw ParseException("Expected 1 arg: `play <duration>`") }
        val durationMs = try {
          args[0].toInt()
        } catch (e: NumberFormatException) {
          throw ParseException("Invalid duration \"${args[0]}\"", e)
        }
        return PlayOp(durationMs)
      }

      fun deserialize(opcode: Int): PlayOp {
        return PlayOp(decodeDuration(opcode and 0x1f /* Extract the lowest 5 bits. */))
      }

      fun encodeDuration(durationMs: Int) = (durationMs.coerceIn(20..640) - 10) / 20

      fun decodeDuration(code: Int) = (code + 1) * 20

      fun constrainDuration(durationMs: Int) = decodeDuration(encodeDuration(durationMs))
    }
  }

  /** The waveforms that can be set with `SetWaveformOp`. */
  enum class Waveform(val value: Int) {
    /** Pure sinusoidal tones. The suffix is the frequency in Hz, e.g. `SIN_60` is 60 Hz. */
    SIN_25_HZ(0),
    SIN_30_HZ(1),
    SIN_35_HZ(2),
    SIN_45_HZ(3),
    SIN_50_HZ(4),
    SIN_60_HZ(5),
    SIN_70_HZ(6),
    SIN_90_HZ(7),
    SIN_100_HZ(8),
    SIN_125_HZ(9),
    SIN_150_HZ(10),
    SIN_175_HZ(11),
    SIN_200_HZ(12),
    SIN_250_HZ(13),
    SIN_300_HZ(14),
    SIN_350_HZ(15),
    /** Rising exponential chirp. */
    CHIRP(16);

    companion object {
      /** Gets the `Waveform` with value `value`. Returns null on failure. */
      fun fromInt(value: Int) = Waveform.values().firstOrNull { it.value == value }

      /** Gets the `Waveform` with name `name`. Throws `ParseException` on failure. */
      fun fromString(name: String) =
        Waveform.values().firstOrNull { it.name.lowercase() == name }
        ?: throw ParseException("Invalid waveform \"$name\"")
    }
  }

  /** Sets the waveform for one channel, or for all channels if `channel < 0`. */
  class SetWaveformOp(var channel: Int, var waveform: Waveform) : Op() {
    override fun toString() =
      "$OP_NAME_SET_WAVEFORM ${Op.channelToString(channel)} ${waveform.name.lowercase()}"

    override val serializedSize = 2

    override fun serialize(buffer: ByteBuffer) {
      val opcode = if (channel < 0) { OP_SET_ALL_WAVEFORM } else { OP_SET_WAVEFORM + channel}
      buffer.put(opcode.toByte())
      buffer.put(waveform.value.toByte())
    }

    internal companion object {
      fun parse(args: List<String>): SetWaveformOp {
        if (args.size != 2) { throw ParseException("Expected 2 args: `wave <channel> <waveform>`") }
        val channel = Op.parseChannel(args[0])
        val waveform = Waveform.fromString(args[1])
        return SetWaveformOp(channel, waveform)
      }

      fun deserialize(opcode: Int, buffer: ByteBuffer): SetWaveformOp? {
        if (buffer.remaining() < 1) { return null }
        val channel = if (opcode == OP_SET_ALL_WAVEFORM) { -1 } else { opcode and 0x0f }
        val waveform = Waveform.fromInt(buffer.get().toNonnegInt()) ?: return null
        return SetWaveformOp(channel, waveform)
      }
    }
  }

  /** Sets the gain for one channel, or for all channels if `channel < 0`. */
  class SetGainOp(var channel: Int, gainIn: Float) : Op() {
    private var _gain: Float = constrainGain(gainIn)
    /** The gain, represented as a linear value in the range [0, 1]. */
    var gain: Float
      get() = _gain
      set(newGain) {
        _gain = constrainGain(newGain)
      }

    override fun toString() = "$OP_NAME_SET_GAIN %s %.3f".format(Op.channelToString(channel), gain)

    override val serializedSize = 2

    override fun serialize(buffer: ByteBuffer) {
      val opcode = if (channel < 0) { OP_SET_ALL_GAIN } else { OP_SET_GAIN + channel}
      buffer.put(opcode.toByte())
      buffer.put(encodeGain(gain).toByte())
    }

    companion object {
      internal fun parse(args: List<String>): SetGainOp {
        if (args.size != 2) { throw ParseException("Expected 2 args: `gain <channel> <gain>`") }
        val channel = Op.parseChannel(args[0])
        val gain = try {
          args[1].toFloat()
        } catch (e: NumberFormatException) {
          throw ParseException("Invalid gain \"${args[1]}\"", e)
        }
        return SetGainOp(channel, gain)
      }

      internal fun deserialize(opcode: Int, buffer: ByteBuffer): SetGainOp? {
        if (buffer.remaining() < 1) { return null }
        val channel = if (opcode == OP_SET_ALL_GAIN) { -1 } else { opcode and 0x0f }
        val gain = decodeGain(buffer.get().toNonnegInt())
        return SetGainOp(channel, gain)
      }

      internal fun encodeGain(gain: Float) = (255.0f * gain.coerceIn(0.0f, 1.0f)).roundToInt()

      internal fun decodeGain(code: Int) = code / 255.0f

      fun constrainGain(gain: Float) = decodeGain(encodeGain(gain))
    }
  }

  /** Moves channel `fromChannel` to `toChannel`. */
  class MoveOp(var fromChannel: Int, var toChannel: Int) : Op() {
    override fun toString() = "$OP_NAME_MOVE %-3d %d".format(fromChannel + 1, toChannel + 1)

    override val serializedSize = 2

    override fun serialize(buffer: ByteBuffer) {
      buffer.put(OP_MOVE.toByte())
      buffer.put(Byte.fromNibbles(toChannel, fromChannel))
    }

    internal companion object {
      fun parse(args: List<String>): MoveOp {
        if (args.size != 2) { throw ParseException("Expected 2 args: `move <from> <to>`") }
        val fromChannel = Op.parseChannel(args[0])
        val toChannel = Op.parseChannel(args[1])
        return MoveOp(fromChannel, toChannel)
      }

      fun deserialize(buffer: ByteBuffer): MoveOp? {
        if (buffer.remaining() < 1) { return null }
        val channels = buffer.get()
        return MoveOp(fromChannel = channels.upperNibble(), toChannel = channels.lowerNibble())
      }
    }
  }

  /** Exception thrown when text parsing fails. */
  class ParseException(message: String, cause: Throwable? = null) : Exception(message, cause)
}
