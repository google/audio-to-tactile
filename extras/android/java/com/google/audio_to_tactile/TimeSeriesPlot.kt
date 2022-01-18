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

import android.animation.TimeAnimator
import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.drawable.Drawable
import androidx.annotation.ColorInt

/**
 * Animated Drawable that plots a scrolling time series. Timestamps are UTC milliseconds from the
 * epoch as returned by `System.currentTimeMillis()`.
 */
class TimeSeriesPlot(
  val timeSeries: TimeSeries,
  var currentTimeMs: Long,
  @ColorInt lineColor: Int,
  @ColorInt bgColor: Int,
  @ColorInt gridColor: Int
) : Drawable(), TimeAnimator.TimeListener {
  /** Paint object for drawing the plot line, having color `lineColor`. */
  private val linePaint = Paint().apply {
    color = lineColor
    strokeWidth = 5f
    isAntiAlias = true
  }

  /** Paint for filling the plot box background. */
  private val bgPaint = Paint().apply {
    color = bgColor
    style = Paint.Style.FILL
  }

  /** Paint for drawing vertical grid lines. */
  private val gridPaint = Paint().apply {
    color = gridColor
    strokeWidth = 3f
    isAntiAlias = true
  }

  /** Number of grid lines to draw, which are spaced one second apart. */
  private val numGridLines = 1 + timeSeries.windowDurationMs / 1000

  init {
    TimeAnimator().apply {
      setTimeListener(this@TimeSeriesPlot)
      start()
    }
  }

  /** On every frame, update `currentTime` to scroll the plot and call `invalidateSelf()`. */
  override fun onTimeUpdate(animation: TimeAnimator?, totalTimeMs: Long, deltaTimeMs: Long) {
    currentTimeMs += deltaTimeMs
    // The time series data is generated on the wearable, which does not have an accurate clock.
    // timeSeries makes timing adjustments to force the data to be spaced with a presumed sample
    // period, and consequently, there can be clock drift where timeSeries is ahead of real time.
    //
    // We account for this by adding a couple milliseconds per frame as needed to make gradual
    // corrections. ADJUST_MS the adjustment increment, which should be small enough to avoid
    // visually obvious jumps but large enough to make timing corrections. Supposing 60 frames per
    // second, adding 2 ms per frame can correct up to 120 ms per second, which is more than enough.
    val ADJUST_MS = 2L
    if (timeSeries.currentTimeMs - currentTimeMs >= ADJUST_MS) { currentTimeMs += ADJUST_MS }

    invalidateSelf() // Indicate that plot needs to be redrawn.
  }

  /** Draws the plot. */
  override fun draw(canvas: Canvas) {
    val width = bounds.width().toFloat()
    val height = bounds.height().toFloat()
    val timeLeftEdgeMs = currentTimeMs - timeSeries.windowDurationMs
    val xScale = width / timeSeries.windowDurationMs
    val yScale = height
    val lines = timeSeries.getLines { point ->
      val x = xScale * (point.timeMs - timeLeftEdgeMs).toFloat()
      val y = yScale * (1.0f - point.value)
      Pair(x, y)
    }

    // Fill plot background rectangle.
    canvas.drawRect(0f, 0f, width, height, bgPaint)
    // Draw grid lines spaced one second apart.
    val phase = 1000 - timeLeftEdgeMs % 1000
    for (i in 0 until numGridLines) {
      val x = xScale * (phase + i * 1000).toFloat()
      canvas.drawLine(x, 0f, x, height, gridPaint)
    }
    // Plot the time series data.
    canvas.drawLines(lines, linePaint)
  }

  override fun setAlpha(alpha: Int) {}
  override fun setColorFilter(colorFilter: ColorFilter?) {}
  override fun getOpacity(): Int = PixelFormat.OPAQUE
}
