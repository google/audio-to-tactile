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

/**
 * `ChannelMap` describes remapping and gains on a multichannel signal, corresponding to C library
 * src/dsp/channel_map.h. The mapping is of the form `output[c] = gains[c] * input[sources[c]]`.
 * Each gain is represented with an integer control value in 0-63 to specify a gain of 0 (channel is
 * silent) for control value 0 or a gain in decibels from -18 dB to +0 dB.
 *
 * `ChannelMap` is initialized with a unit gain identity mapping, with `source[c] = c` source
 * mapping and gain +0 dB (control value 63).
 *
 * The number of input channels must be between 1 and 16, and the number of output channels must be
 * between 1 and 12.
 */
class ChannelMap
    private constructor(
      val numInputChannels: Int,
      val numOutputChannels: Int,
      val entries: List<Entry>
    ) {

  constructor(numInputChannels: Int, numOutputChannels: Int) :
      this(
        numInputChannels,
        numOutputChannels,
        List(numOutputChannels) { i -> Entry(numInputChannels, i) }
      )

  init {
    require(numInputChannels in 1..16 && numOutputChannels in 1..12) {
      "Unsupported number of channels: $numInputChannels in, $numOutputChannels out"
    }
  }

  /** Number of output channels, also the number of entries in the channel map. */
  val size = numOutputChannels
  /** Range of indices for iterating the channel map. */
  val indices
    get() = entries.indices

  /** Interface for one tuning knob, accessed through the [ChannelMap.get] operator. */
  class Entry(val numInputChannels: Int, val index: Int) {
    /** Input source channel as a 0-base index. */
    var source = index.coerceIn(0 until numInputChannels)
    /** Whether this channel is enabled. */
    var enabled = true

    /**
     * Gain control value, an integer in 0-63. The control value is mapped to decibels according to
     * [channelGainMapping]. A control value of 0 represents that this channel is disabled.
     */
    var gain: Int
      get() = if (enabled) enabledGain else 0
      set(value) {
        val clamped = value.coerceIn(0, UNIT_GAIN) // Clamp control value to 0-63.
        if (clamped > 0) { // Positive control value => channel is enabled.
          _enabledGain = clamped
          enabled = true
        } else { // Control value 0 => channel is disabled.
          enabled = false
        }
      }

    /**
     * The last gain control value when the channel was enabled. This member is used to restore the
     * former gain when a channel is disabled and then re-enabled.
     */
    private var _enabledGain = UNIT_GAIN
    val enabledGain: Int
      get() = _enabledGain

    /** String representation of the source as a 1-based index, useful for UI. */
    val sourceString
      get() = (source + 1).toString()
    /** String representation of the gain in decibels, useful for UI. */
    val gainString
      get() = gainMapping(gain)

    /** Resets `Entry` to initial settings. */
    fun reset() {
      source = index.coerceIn(0 until numInputChannels)
      enabled = true
      _enabledGain = UNIT_GAIN
    }

    /** Creates a deep copy of the `Entry`. */
    fun copyOf() =
      Entry(numInputChannels, index).also {
        it.source = source
        it.enabled = enabled
        it._enabledGain = enabledGain
      }
  }
  operator fun get(index: Int) = entries[index.coerceIn(indices)]

  /** Resets all channels to initial settings. */
  fun resetAll() {
    entries.forEach { it.reset() }
  }

  /** Creates a deep copy of the `ChannelMap`. */
  fun copyOf() = ChannelMap(numInputChannels, numOutputChannels, entries.map { it.copyOf() })

  /** Compares two `ChannelMap` instances. Returns true if all fields are equal. */
  fun contentEquals(rhs: ChannelMap?): Boolean {
    if (!sameNumChannelsAs(rhs)) { return false }
    return indices.all { i ->
      val a = entries[i]
      val b = rhs!!.entries[i]
      a.source == b.source && a.gain == b.gain }
  }

  /**
   * Compares two `ChannelMap` instances ignoring the gains. Returns true if the number of channels
   * and all sources are equal. Together with [contentEquals], this method is useful to determine in
   * what way two ChannelMaps differ:
   *
   * ```
   * if (channelMap.contentEquals(rhs)) {
   *   // All fields are equal.
   * } else if (channelMap.sameSourcesAs(rhs)) {
   *   // One or more gains differ, all other fields are equal.
   * } else {
   *   // Something besides the gains differs.
   * }
   * ```
   */
  fun sameSourcesAs(rhs: ChannelMap?): Boolean {
    if (!sameNumChannelsAs(rhs)) { return false }
    return indices.all { i -> entries[i].source == rhs!!.entries[i].source }
  }

  /** Returns true if `rhs` has the same number of input and output channels. */
  fun sameNumChannelsAs(rhs: ChannelMap?) =
    rhs != null &&
      numInputChannels == rhs.numInputChannels &&
      numOutputChannels == rhs.numOutputChannels

  companion object {
    /** Control value representing unit gain. */
    const val UNIT_GAIN = 63

    /** Converts gain control value in 0-63 to a string, useful for UI display. */
    fun gainMapping(gain: Int): String {
      // Control value of 0 means the channel is disabled.
      if (gain == 0) {
        return "off"
      }

      // Convert control value in 1-63 to decibels.
      val gainDecibels = (18.0f / 62.0f) * (gain - 63).toFloat()
      return "%.1f dB".format(gainDecibels).replace('-', '\u2212' /* unicode minus */)
    }
  }
}
