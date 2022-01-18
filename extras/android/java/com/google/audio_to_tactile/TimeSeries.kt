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

import androidx.annotation.VisibleForTesting
import kotlin.math.max

/**
 * This class represents time series data, useful for a scrolling plot. The time series is a list of
 * `Points`, each Point having a Long-valued timestamp and a Float value. Timestamps are UTC
 * milliseconds from the epoch as returned by `System.currentTimeMillis()`.
 */
class TimeSeries(val samplePeriodMs: Long, val windowDurationMs: Long) {
  /** One point in the time series, having a time in units of milliseconds and a Float value. */
  data class Point(val timeMs: Long, val value: Float)

  private var _points = mutableListOf<Point>()
  /** The time series data, a list of timestamped values. */
  val points: List<Point>
    get() = _points

  private var _currentTimeMs = 0L
  /** The current time, the time of the latest point. */
  val currentTimeMs: Long
    get() = _currentTimeMs

  /**
   * Appends new samples to the time series, where the timestamp of the last sample is `lastTimeMs`,
   * and old points are discarded. If needed, timestamps are increased to ensure they are
   * monotonically increasing and spaced at least `samplePeriod` apart.
   */
  fun add(lastTimeMs: Long, values: FloatArray) {
    _currentTimeMs = max(_currentTimeMs, lastTimeMs - samplePeriodMs * values.size)

    // Get the time corresponding to the left (trailing) edge of the window.
    val leftEdgeTimeMs = _currentTimeMs + samplePeriodMs * values.size - windowDurationMs
    val updatedPoints = discardOldPoints(leftEdgeTimeMs)

    // Append new data.
    for (value in values) {
      _currentTimeMs += samplePeriodMs
      updatedPoints.add(Point(_currentTimeMs, value))
    }

    _points = updatedPoints
  }

  /**
   * Returns a new list of points where old data is dropped: points older than leftEdgeTimeMs are
   * outside the window and won't be plotted. Since the plot draws line segments, one line segment
   * may cross the left edge, so we keep the latest point <= leftEdgeTimeMs.
   */
  @VisibleForTesting
  fun discardOldPoints(leftEdgeTimeMs: Long): MutableList<Point> {
    val i = _points.indexOfLast { it.timeMs <= leftEdgeTimeMs }
    if (i < 1) { return _points }
    return _points.drop(i).toMutableList()
  }

  /**
   * Converts the points to a line coordinates for use with `Canvas.drawLines`. The `dataToScreen`
   * function specifies how to convert a Point in "data space" to (x,y) drawing coordinates in
   * "screen space".
   */
  inline fun getLines(dataToScreen: (Point) -> Pair<Float, Float>) =
    with (points) {
      if (size < 2) { return@with floatArrayOf() }

      val lines = FloatArray(4 * (size - 1))
      var (x0, y0) = dataToScreen(get(0))
      var j = 0

      for (i in 1 until size) {
        val (x1, y1) = dataToScreen(get(i))
        lines[j] = x0
        lines[j + 1] = y0
        lines[j + 2] = x1
        lines[j + 3] = y1
        x0 = x1
        y0 = y1
        j += 4
      }

      return@with lines
    }
}
