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

/** Algorithm tuning knob settings, corresponding to `TuningKnobs` in src/tactile/tuning.h. */
package com.google.audio_to_tactile

class Tuning(initValues: IntArray = TuningNative.defaultTuningKnobs) {
  /** IntArray of all the tuning knob values. */
  val values = initValues.copyOf()
  /** Number of tuning knobs. */
  val size
    get() = values.size
  /** Range indices for iterating the tuning knobs. */
  val indices
    get() = values.indices

  /** Interface for one tuning knob, accessed through the [Tuning.get] operator. */
  class Entry(val values: IntArray, val index: Int) {
    /** Knob value, an integer between 0 and 255. */
    var value: Int
      get() = values[index]
      set(newValue) {
        values[index] = newValue.coerceIn(0, 255) // Clamp value to 0-255.
      }
    /** Brief name string identifying this knob. */
    val name
      get() = TuningNative.name(index)
    /** Longer description string with more details. */
    val description
      get() = TuningNative.description(index)
    /** The knob's current value represented as a string, including units if applicable. */
    val valueString
      get() = mapping(value)

    /** Function for how this knob maps `value` in 0-255 to a string. */
    fun mapping(value: Int): String =
      TuningNative.mapControlValue(index, value)
        // When printing negative values, replace '-' with nicer-looking unicode minus symbol.
        .replace('-', '\u2212')

    /** Resets knob to its default value. */
    fun reset() {
      value = TuningNative.default(index)
    }
  }
  operator fun get(index: Int) = entries[index.coerceAtMost(size - 1)]
  private val entries = List(size, { i -> Entry(values, i) })

  /** Creates a copy of the Tuning. */
  fun copyOf() = Tuning(values)
  /** Tests whether two Tunings are equal. */
  fun contentEquals(rhs: Tuning?) = rhs?.values?.contentEquals(values) ?: false

  /** Resets all knobs to their default values. */
  fun resetAll() {
    for (i in indices) {
      get(i).reset()
    }
  }

  companion object {
    val NUM_TUNING_KNOBS = TuningNative.numTuningKnobs()
    val DEFAULT_TUNING_KNOBS = TuningNative.defaultTuningKnobs
  }
}

/** Tuning JNI bindings. */
private object TuningNative {
  init {
    System.loadLibrary("tuning_jni")
  }

  val defaultTuningKnobs = IntArray(numTuningKnobs()) { i -> default(i) }

  /** Gets the name string for `knob`. */
  external fun name(knob: Int): String
  /** Gets the description string for `knob`. */
  external fun description(knob: Int): String
  /** Gets the default value for `knob`. */
  external fun default(knob: Int): Int
  /** Gets the number of tuning knobs. */
  external fun numTuningKnobs(): Int
  /** Maps a knob control value and formats it as a string, including units if applicable. */
  external fun mapControlValue(knob: Int, value: Int): String
}
