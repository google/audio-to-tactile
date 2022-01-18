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

/** Utilities for serializing and deserializing some small primitives in ByteBuffers. */
package com.google.audio_to_tactile

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Converts Byte to a nonnegative Int in the range 0-255. */
fun Byte.toNonnegInt() = toInt() and 0xff

/** Extracts the lower 4 bits (nibble) from a Byte. */
fun Byte.lowerNibble() = toInt() and 0xf
/** Extracts the upper 4 bits (nibble) from a Byte. */
fun Byte.upperNibble() = (toInt() shr 4) and 0xf
/** Constructs a Byte from two nibbles. */
fun Byte.Companion.fromNibbles(lower: Int, upper: Int) = ((lower and 0xf) or (upper shl 4)).toByte()

/** Deserializes nonnegative 16-bit int value. */
fun ByteBuffer.getUint16() = short.toInt() and 0xffff
/** Serializes nonnegative 16-bit int value. */
fun ByteBuffer.putUint16(value: Int) {
  putShort(value.toShort())
}

/** Deserializes nonnegative 24-bit int value. */
fun ByteBuffer.getUint24(): Int {
  return when (order()) {
    ByteOrder.LITTLE_ENDIAN ->
      (get().toNonnegInt() or
        (get().toNonnegInt() shl 8) or
        (get().toNonnegInt() shl 16))
    else ->
      ((get().toNonnegInt() shl 16) or
        (get().toNonnegInt() shl 8) or
        get().toNonnegInt())
  }
}
/** Serializes nonnegative 24-bit int value. */
fun ByteBuffer.putUint24(value24: Int) {
  when (order()) {
    ByteOrder.LITTLE_ENDIAN ->
      put(value24.toByte())
        .put((value24 shr 8).toByte())
        .put((value24 shr 16).toByte())
    else ->
      put((value24 shr 16).toByte())
        .put((value24 shr 8).toByte())
        .put(value24.toByte())
  }
}

/**
 * Computes the 16-bit Fletcher-16 checksum [https://en.wikipedia.org/wiki/Fletcher%27s_checksum]:
 *
 * ```
 * sum1 = (init1 + D0 + D1 + ...) % 255,
 * sum2 = ((init2 + init1 + D0) + (init2 + init1 + D0 + D1) + ...) % 255,
 * checksum = (sum2 << 8) | sum1,
 * ```
 *
 * where D0, D1, etc. are data bytes and init1 and init2 derive from `init`. The checksum is
 * computed over the `ByteBuffer` starting from `position()` and ending at `limit()`. The `init` arg
 * specifies an initial checksum for starting the two sums. A good starting value is 1. This arg is
 * useful for incrementally computing a checksum over multiple calls.
 */
fun ByteBuffer.fletcher16(init: Int = 1): Int {
  val MAX_BLOCK_LEN = 4102
  val originalPosition = position()
  var sum1 = init and 0xff
  var sum2 = init shr 8

  while (hasRemaining()) {
    // To reduce the number of % ops, a standard optimization is to accumulate as much as possible
    // before sum2 could overflow [due to Nakassis, 1988, "Fletcher's error detection algorithm: how
    // to implement it efficiently and how to avoid the most common pitfalls"]. After n steps:
    //
    //   sum1 <= 254 + 255 n,
    //   sum2 <= 254 + 254 n + 255 (n + 1) n / 2.
    //
    // So sum2 <= 2^31 - 1 for n <= 4102 = `MAX_BLOCK_LEN`.
    val blockLen = MAX_BLOCK_LEN.coerceAtMost(remaining())
    for (i in 0 until blockLen) {
      sum1 += get().toNonnegInt()
      sum2 += sum1
    }

    sum1 %= 255
    sum2 %= 255
  }

  position(originalPosition) // Restore original position.
  return sum1 or (sum2 shl 8)
}
