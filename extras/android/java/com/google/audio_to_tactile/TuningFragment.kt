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
 * Define the Tuning fragment, accessed from the nav drawer. In this fragment, algorithm tuning
 * knobs like input gain and denoising strengths can be modified.
 */
@AndroidEntryPoint class TuningFragment : Fragment() {
  private val bleViewModel: BleViewModel by activityViewModels()

  private class TuningItemViewHolder(val view: View, onClick: (Int) -> Unit) :
    RecyclerView.ViewHolder(view) {
    val knobName: TextView = view.findViewById(R.id.tuning_item_name)
    val knobValue: TextView = view.findViewById(R.id.tuning_item_value)

    init {
      view.setOnClickListener { onClick(absoluteAdapterPosition) }
    }
  }

  /** Adapter to display a list of all the tuning knob names and values. */
  private val tuningItemAdapter =
    object : RecyclerView.Adapter<TuningItemViewHolder>() {

      override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TuningItemViewHolder {
        val view: View =
          LayoutInflater.from(parent.context).inflate(R.layout.tuning_list_item, parent, false)
        return TuningItemViewHolder(view) { tuningKnobIndex ->
          @IdRes val actionId = R.id.action_nav_tuning_to_tuning_dialog
          with(NavHostFragment.findNavController(this@TuningFragment)) {
            // Edge case: This onClick listener may be called twice if the user taps the list with
            // two fingers or double taps. The first call navigates to [TuningDialogFragment] as
            // intended. However, navigation in the second call would fail and crash the app, since
            // that would attempt to go from the dialog to itself. To avoid that, we check that
            // `currentDestination?.getAction(actionId)` is nonnull before navigating.
            if (currentDestination?.getAction(actionId) != null) {
              navigate(actionId, TuningDialogFragment.createArgs(tuningKnobIndex))
            }
          }
        }
      }

      override fun getItemCount(): Int = Tuning.NUM_TUNING_KNOBS

      override fun onBindViewHolder(holder: TuningItemViewHolder, position: Int) {
        // Observe knob with index `position` in [BleViewModel.tuning], so UI updates on change.
        bleViewModel.tuning.observe(viewLifecycleOwner) {
          it?.let { tuning ->
            val knob = tuning[position]
            holder.knobName.text = knob.name
            holder.knobValue.text = knob.valueString
          }
        }
      }
    }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_tuning, container, false)

    val resetAllButton: Button = root.findViewById(R.id.tuning_reset_all)
    resetAllButton.setOnClickListener { bleViewModel.tuningResetAll() }

    root.findViewById<RecyclerView>(R.id.tuning_recycler_view).apply {
      adapter = tuningItemAdapter
      setHasFixedSize(true)
    }

    return root
  }
}
