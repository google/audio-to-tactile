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
 * Define the Sound Events fragment, accessed from the nav drawer. This fragment will enable the
 * user to view and map sound events to different tactile patterns.
 */
@AndroidEntryPoint class SoundEventsFragment : Fragment() {
  private val bleViewModel: BleViewModel by activityViewModels()

  private class SoundEventItemViewHolder(val view: View, onClick: (Int) -> Unit) :
    RecyclerView.ViewHolder(view) {
    val soundEventIndex: TextView = view.findViewById(R.id.sound_event_index)
    val TactilePatternDescriptionButton: Button =
      view.findViewById(R.id.select_tactile_pattern_button)

    init {
      view.setOnClickListener { onClick(absoluteAdapterPosition) }
    }
  }

  /** Adapter to display a list of all the sound events. */
  private val soundEventItemAdapter =
    object : RecyclerView.Adapter<SoundEventItemViewHolder>() {

      override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SoundEventItemViewHolder {
        val view: View =
          LayoutInflater.from(parent.context)
            .inflate(R.layout.sound_events_list_item, parent, false)
        return SoundEventItemViewHolder(view) { soundIndex ->
          @IdRes val actionId = R.id.action_nav_sound_event_to_sound_event_dialog
          with(NavHostFragment.findNavController(this@SoundEventsFragment)) {
            if (currentDestination?.getAction(actionId) != null) {
              navigate(actionId, SoundEventsDialogFragment.createArgs(soundIndex))
            }
          }
        }
      }

      override fun getItemCount(): Int = SoundEvents.SOUND_EVENTS.size

      override fun onBindViewHolder(holder: SoundEventItemViewHolder, position: Int) {
        holder.soundEventIndex.text = (SoundEvents.SOUND_EVENTS[position])
        holder.TactilePatternDescriptionButton.text = (SoundEvents.TACTILE_PATTERNS[position])

        // Click to play the test pattern. Right now it is a placeholder that plays channelMapTest.
        holder.TactilePatternDescriptionButton.setOnClickListener {
          bleViewModel.channelMapTest(position)
        }
      }
    }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_sound_events, container, false)

    root.findViewById<RecyclerView>(R.id.sound_events_recycler_view).apply {
      adapter = soundEventItemAdapter
      setHasFixedSize(true)
    }

    return root
  }
}
