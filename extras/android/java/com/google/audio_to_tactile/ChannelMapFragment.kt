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
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.annotation.IdRes
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.fragment.NavHostFragment
import androidx.recyclerview.widget.RecyclerView
import dagger.hilt.android.AndroidEntryPoint

/**
 * Define the ChannelMap fragment, accessed from the nav drawer. This fragment enables the user to
 * configure the channel gains and mapping of sources to tactors.
 */
@AndroidEntryPoint class ChannelMapFragment : Fragment() {
  private val bleViewModel: BleViewModel by activityViewModels()

  private class ChannelItemViewHolder(val view: View, onClick: (Int) -> Unit) :
    RecyclerView.ViewHolder(view) {
    val tactorIndex: TextView = view.findViewById(R.id.channel_map_item_tactor_index)
    val sourceIndex: TextView = view.findViewById(R.id.channel_map_item_source_index)
    val gain: TextView = view.findViewById(R.id.channel_map_item_gain)
    val testButton: Button = view.findViewById(R.id.channel_map_item_test_button)

    init {
      view.setOnClickListener { onClick(absoluteAdapterPosition) }
    }
  }

  /** Adapter to display a list of all the channels. */
  private val channelItemAdapter =
    object : RecyclerView.Adapter<ChannelItemViewHolder>() {

      override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ChannelItemViewHolder {
        val view: View =
          LayoutInflater.from(parent.context).inflate(R.layout.channel_map_list_item, parent, false)
        return ChannelItemViewHolder(view) { channelIndex ->
          @IdRes val actionId = R.id.action_nav_channel_map_to_channel_map_dialog
          with(NavHostFragment.findNavController(this@ChannelMapFragment)) {
            // Edge case: This onClick listener may be called twice if the user taps the list with
            // two fingers or double taps. The first call navigates to [ChannelMapDialogFragment] as
            // intended. However, navigation in the second call would fail and crash the app, since
            // that would attempt to go from the dialog to itself. To avoid that, we check that
            // `currentDestination?.getAction(actionId)` is nonnull before navigating.
            if (currentDestination?.getAction(actionId) != null) {
              navigate(actionId, ChannelMapDialogFragment.createArgs(channelIndex))
            }
          }
        }
      }

      override fun getItemCount(): Int = bleViewModel.channelMap.value!!.numOutputChannels

      override fun onBindViewHolder(holder: ChannelItemViewHolder, position: Int) {
        holder.tactorIndex.text = (position + 1).toString() // Convert to base-1 index for UI.

        // Observe channel `position` in [BleViewModel.channelMap], so UI updates on change.
        bleViewModel.channelMap.observe(viewLifecycleOwner) {
          it?.let { channelMap ->
            val channel = channelMap[position]
            holder.sourceIndex.text = channel.sourceString
            holder.gain.text = channel.gainString
          }
        }

        holder.testButton.setOnClickListener { bleViewModel.channelMapTest(position) }
      }
    }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_channel_map, container, false)

    val resetAllButton: Button = root.findViewById(R.id.channel_map_reset_all)
    resetAllButton.setOnClickListener { bleViewModel.channelMapResetAll() }

    root.findViewById<RecyclerView>(R.id.channel_map_recycler_view).apply {
      adapter = channelItemAdapter
      setHasFixedSize(true)
    }

    return root
  }
}
