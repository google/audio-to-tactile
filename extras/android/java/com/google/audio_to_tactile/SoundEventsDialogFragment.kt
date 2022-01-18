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
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.Button
import android.widget.TextView
import androidx.core.os.bundleOf
import androidx.fragment.app.activityViewModels
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.slider.Slider
import com.google.android.material.switchmaterial.SwitchMaterial
import com.google.android.material.textfield.TextInputLayout
import dagger.hilt.android.AndroidEntryPoint
import kotlin.math.roundToInt

/**
 * Define the Sound Events Dialog fragment accessed from Sound Events fragment. This fragment allow
 * tuning of specific sound event.
 */
@AndroidEntryPoint class SoundEventsDialogFragment : BottomSheetDialogFragment() {
  private val bleViewModel: BleViewModel by activityViewModels()

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    return inflater.inflate(R.layout.fragment_sound_event_dialog, container, false)
  }

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val soundEventIndex = arguments?.getInt(SOUND_INDEX_KEY) ?: return

    view.findViewById<TextView>(R.id.sound_event_dialog_title).apply {
      text = SoundEvents.SOUND_EVENTS[soundEventIndex]
    }

    // Reset button.
    view.findViewById<Button>(R.id.sound_event_reset).apply {
      setOnClickListener {
        // TODO: Add action for clicking on reset button.
      }
    }

    // Source dropdown menu.
    val soundEventSourceField: TextInputLayout = view.findViewById(R.id.sound_events_source_field)

    val soundEventSourceDropdown: AutoCompleteTextView =
      view.findViewById(R.id.sound_events_source_dropdown)

    soundEventSourceDropdown.apply {
      setAdapter(
        ArrayAdapter(
          this@SoundEventsDialogFragment.requireContext(),
          R.layout.sound_event_configure_menu_item,
          SoundEvents.TACTILE_PATTERNS
        )
      )
      setOnItemClickListener { _, _, position, _ ->
        // TODO: Add action for selecting a tactile pattern.
      }
    }

    // Gain value text.
    val soundEventGainValue: TextView = view.findViewById(R.id.sound_events_gain_value)

    // Gain slider.
    val soundEventGainSlider: Slider = view.findViewById(R.id.sound_event_gain_slider)
    soundEventGainSlider.apply {
      setLabelFormatter { value -> ChannelMap.gainMapping(value.roundToInt()) }
      addOnSliderTouchListener(
        object : Slider.OnSliderTouchListener {
          override fun onStartTrackingTouch(slider: Slider) {}

          override fun onStopTrackingTouch(slider: Slider) {
            // TODO: Add action for gain slider.
          }
        }
      )
    }

    // Enable switch.
    val soundEventEnableSwitch: SwitchMaterial = view.findViewById(R.id.sound_event_enable_switch)
    soundEventEnableSwitch.setOnCheckedChangeListener { _, enabled ->
      // TODO: Add action for enable/disable sound events toggle switch.
    }
  }

  companion object {
    private const val SOUND_INDEX_KEY = "channel_index_key"

    /** Creates args Bundle for navigating to this fragment. */
    fun createArgs(soundEventIndex: Int): Bundle = bundleOf(SOUND_INDEX_KEY to soundEventIndex)
  }
}
