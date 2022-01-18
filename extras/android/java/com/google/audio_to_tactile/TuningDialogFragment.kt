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
import androidx.core.os.bundleOf
import androidx.fragment.app.activityViewModels
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.slider.Slider
import dagger.hilt.android.AndroidEntryPoint
import kotlin.math.roundToInt

@AndroidEntryPoint class TuningDialogFragment : BottomSheetDialogFragment() {
  private val bleViewModel: BleViewModel by activityViewModels()

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    return inflater.inflate(R.layout.fragment_tuning_dialog, container, false)
  }

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    val tuningKnobIndex = arguments?.getInt(TUNING_KNOB_INDEX_KEY) ?: return
    val knob = bleViewModel.tuning.getValue()?.get(tuningKnobIndex) ?: return

    view.findViewById<TextView>(R.id.tuning_knob_name).apply {
      text = knob.name
    }
    view.findViewById<TextView>(R.id.tuning_knob_description).apply {
      text = knob.description
    }

    // Reset button.
    view.findViewById<Button>(R.id.tuning_knob_reset).apply {
      setOnClickListener {
        bleViewModel.tuningKnobReset(tuningKnobIndex)
      }
    }

    // Knob value text.
    val tuningKnobValue: TextView = view.findViewById(R.id.tuning_knob_value)

    // Knob slider.
    val tuningKnobSlider: Slider = view.findViewById(R.id.tuning_knob_slider)
    tuningKnobSlider.apply {
      setLabelFormatter { value -> knob.mapping(value.roundToInt()) }
      addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
        override fun onStartTrackingTouch(slider: Slider) {}

        override fun onStopTrackingTouch(slider: Slider) {
          bleViewModel.tuningKnobSetValue(tuningKnobIndex, slider.value.roundToInt())
        }
      })
    }

    // The knob value text and slider observe [BleViewModel.tuning].
    bleViewModel.tuning.observe(viewLifecycleOwner) { tuning ->
      tuning?.get(tuningKnobIndex)?.let {
        tuningKnobValue.text = it.valueString
        tuningKnobSlider.value = it.value.toFloat()
      }
    }
  }

  companion object {
    private const val TUNING_KNOB_INDEX_KEY = "tuning_knob_index_key"

    /** Creates args Bundle for navigating to this fragment. */
    fun createArgs(tuningKnobIndex: Int): Bundle =
      bundleOf(TUNING_KNOB_INDEX_KEY to tuningKnobIndex)
  }
}
