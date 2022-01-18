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

@AndroidEntryPoint class ChannelMapDialogFragment : BottomSheetDialogFragment() {
  private val bleViewModel: BleViewModel by activityViewModels()

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    return inflater.inflate(R.layout.fragment_channel_map_dialog, container, false)
  }

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val channelIndex = arguments?.getInt(CHANNEL_INDEX_KEY) ?: return

    view.findViewById<TextView>(R.id.channel_map_dialog_title).apply {
      text = "Tactor " + (channelIndex + 1).toString() // Convert to base-1 index for UI.
    }

    // Reset button.
    view.findViewById<Button>(R.id.channel_map_reset).apply {
      setOnClickListener { bleViewModel.channelMapReset(channelIndex) }
    }

    // Source dropdown menu.
    val channelMapSourceField: TextInputLayout = view.findViewById(R.id.channel_map_source_field)
    val channelMapSourceDropdown: AutoCompleteTextView =
      view.findViewById(R.id.channel_map_source_dropdown)
    bleViewModel.channelMap.value?.let { channelMap ->
      channelMapSourceDropdown.apply {
        val items = (1..channelMap.numInputChannels).map { it.toString() }
        setAdapter(
          ArrayAdapter(
            this@ChannelMapDialogFragment.requireContext(),
            R.layout.channel_map_source_menu_item,
            items
          )
        )
        setText(items[channelIndex], false)
        setOnItemClickListener { _, _, position, _ ->
          bleViewModel.channelMapSetSource(channelIndex, position)
        }
      }
    }

    // Gain value text.
    val channelMapGainValue: TextView = view.findViewById(R.id.channel_map_gain_value)

    // Gain slider.
    val channelMapGainSlider: Slider = view.findViewById(R.id.channel_map_gain_slider)
    channelMapGainSlider.apply {
      setLabelFormatter { value -> ChannelMap.gainMapping(value.roundToInt()) }
      addOnSliderTouchListener(
        object : Slider.OnSliderTouchListener {
          override fun onStartTrackingTouch(slider: Slider) {}

          override fun onStopTrackingTouch(slider: Slider) {
            bleViewModel.channelMapSetGain(channelIndex, slider.value.roundToInt())
          }
        }
      )
    }

    // Enable switch.
    val channelMapEnableSwitch: SwitchMaterial = view.findViewById(R.id.channel_map_enable_switch)
    channelMapEnableSwitch.setOnCheckedChangeListener { _, enabled ->
      bleViewModel.channelMapSetEnable(channelIndex, enabled)
    }

    // Source dropdown, gain text, and gain slider observe [BleViewModel.channelMap].
    bleViewModel.channelMap.observe(viewLifecycleOwner) { channelMap ->
      channelMap?.get(channelIndex)?.let {
        channelMapSourceDropdown.setText(it.sourceString, false)
        channelMapGainValue.text = it.gainString
        channelMapGainSlider.value = it.enabledGain.toFloat()

        channelMapEnableSwitch.isChecked = it.enabled
        channelMapSourceField.isEnabled = it.enabled
        channelMapGainSlider.isEnabled = it.enabled
      }
    }
  }

  companion object {
    private const val CHANNEL_INDEX_KEY = "channel_index_key"

    /** Creates args Bundle for navigating to this fragment. */
    fun createArgs(channelIndex: Int): Bundle = bundleOf(CHANNEL_INDEX_KEY to channelIndex)
  }
}
