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

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.RecyclerView
import dagger.hilt.android.AndroidEntryPoint

/**
 * Define the Log fragment, accessed from the nav drawer. This is intended as a tool for
 * development. The Log fragment shows timestamped log messages about BLE and other events.
 */
@AndroidEntryPoint class LogFragment : Fragment() {
  private val bleViewModel: BleViewModel by activityViewModels()
  private var prevNumLines = 0
  private val handler = Handler(Looper.getMainLooper())
  private var scrollToBottomOnNewMessage = true

  private class LogItemViewHolder(val view: View) :
    RecyclerView.ViewHolder(view) {
    val logTimestamp: TextView = view.findViewById(R.id.log_item_timestamp)
    val logMessage: TextView = view.findViewById(R.id.log_item_message)
  }

  /** Adapter to display a list of all the tuning knob names and values. */
  private val logItemAdapter =
    object : RecyclerView.Adapter<LogItemViewHolder>() {

      override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LogItemViewHolder {
        val view: View =
          LayoutInflater.from(parent.context).inflate(R.layout.log_list_item, parent, false)
        return LogItemViewHolder(view)
      }

      override fun getItemCount(): Int = bleViewModel.logLines.value?.size ?: 0

      override fun onBindViewHolder(holder: LogItemViewHolder, position: Int) {
        bleViewModel.logLines.value?.let { logLines ->
          val logLine = logLines[position]
          holder.logTimestamp.text = logLine.timestamp
          holder.logMessage.text = logLine.message
        }
      }
    }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_log, container, false)

    val recyclerView = root.findViewById<RecyclerView>(R.id.log_recycler_view).apply {
      adapter = logItemAdapter
      setHasFixedSize(false)

      // Add a listener such that `scrollToBottomOnNewMessage` is disabled for a few seconds when
      // the user scrolls the log.
      addOnScrollListener(
        object : RecyclerView.OnScrollListener() {
          override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
            if (newState == RecyclerView.SCROLL_STATE_DRAGGING) {
              scrollToBottomOnNewMessage = false // Disable scrolling temporarily.

              handler.removeCallbacksAndMessages(null) // If a callback is pending, cancel it.
              handler.postDelayed({ // After a delay, re-enable scrolling and scroll to bottom.
                scrollToBottomOnNewMessage = true
                recyclerView.scrollToPosition(logItemAdapter.getItemCount() - 1)
              }, 3000L /* milliseconds */)
            }
          }
        }
      )
    }

    // Observe [BleViewModel.logLines], so UI updates on change.
    bleViewModel.logLines.observe(viewLifecycleOwner) {
      it?.let { logLines ->
        logItemAdapter.notifyItemRangeChanged(prevNumLines, logLines.size - prevNumLines)
        if (scrollToBottomOnNewMessage) {
          recyclerView.scrollToPosition(logItemAdapter.getItemCount() - 1)
        }
        prevNumLines = logLines.size
      }
    }

    return root
  }
}
