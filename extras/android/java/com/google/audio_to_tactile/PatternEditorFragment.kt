/* Copyright 2021-2022 Google LLC
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

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.drawable.InsetDrawable
import android.net.Uri
import android.os.Bundle
import android.util.TypedValue
import android.view.LayoutInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.Button
import android.widget.Filter
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.AttrRes
import androidx.annotation.ColorInt
import androidx.appcompat.view.menu.MenuBuilder
import androidx.appcompat.widget.PopupMenu
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.slider.Slider
import dagger.hilt.android.AndroidEntryPoint
import kotlin.math.roundToInt

/**
 * Define the PatternEditor fragment, accessed from the nav drawer. This fragment enables the user
 * to design tactile patterns.
 *
 * TODO: Add a menu of a few hardcorded examples to showcase different functionalities.
 */
@AndroidEntryPoint class PatternEditorFragment : Fragment(),
    PopupMenu.OnMenuItemClickListener {
  private val bleViewModel: BleViewModel by activityViewModels()
  private lateinit var recyclerView: RecyclerView
  private var bgSelectedColor = 0

  /** Tactile pattern ops. */
  private val pattern
    get() = bleViewModel.tactilePattern.value!!
  /** Line index of the currently selected op. */
  private val selectedIndex
    get() = bleViewModel.tactilePatternSelectedIndex.value!!
  /** The currently selected op. */
  private val selectedOp
    get() = bleViewModel.tactilePatternSelectedOp

  private val openPatternLauncher =
    registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
      if (uri != null) {
        requireContext().contentResolver.openInputStream(uri)?.let {
          patternOpItemAdapter.notifyItemRangeRemoved(0, pattern.size)
          bleViewModel.tactilePatternReadFromStream(it)
          patternOpItemAdapter.notifyItemRangeChanged(0, pattern.size)
        }
      }
    }

  private val savePatternLauncher =
    registerForActivityResult(ActivityResultContracts.CreateDocument()) { uri: Uri? ->
      if (uri != null) {
        requireContext().contentResolver.openOutputStream(uri)
          ?.let { bleViewModel.tactilePatternWriteToStream(it) }
      }
    }

  private class PatternOpItemViewHolder(val view: View) :
    RecyclerView.ViewHolder(view) {
    val line: TextView = view.findViewById(R.id.pattern_item_text)
  }

  private val patternOpItemAdapter =
    object : RecyclerView.Adapter<PatternOpItemViewHolder>() {

      override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PatternOpItemViewHolder {
        val view: View =
          LayoutInflater.from(parent.context).inflate(R.layout.pattern_list_item, parent, false)
        return PatternOpItemViewHolder(view)
      }

      override fun getItemCount(): Int = pattern.size

      override fun onBindViewHolder(holder: PatternOpItemViewHolder, position: Int) {
        val op = pattern.ops[position]
        // Observe `tactilePattern` to update display when op parameters are modified.
        bleViewModel.tactilePattern.observe(viewLifecycleOwner) {
          holder.line.text = op.toString()
        }
        // Observe `tactilePatternSelectedIndex` to update highlighting.
        bleViewModel.tactilePatternSelectedIndex.observe(viewLifecycleOwner) {
          holder.view.setBackgroundColor(if (selectedOp === op) { bgSelectedColor } else { 0 })
        }
        // Clicking the op selects it.
        holder.view.setOnClickListener {
          bleViewModel.tactilePatternSelect(pattern.ops.indexOf(op))
        }
      }
    }

  /**
   * Base class for a parameter slider, used for play duration and gain. The current value is shown
   * in both the TextView and Slider. Moving the Slider calls `onSliderChange()`.
   */
  abstract class ParamSlider(private val textView: TextView, private val slider: Slider) {
    /** Override to define how to display the value. */
    abstract fun valueString(value: Float): String

    /** Override to respond to when the slider changes. */
    abstract fun onSliderChange(value: Float)

    /** Sets the value of the Slider and updates the TextView. */
    fun setValue(newValue: Float) {
      textView.text = valueString(newValue)
      slider.value = newValue
    }

    init {
      slider.setLabelFormatter { valueString(it) }
      slider.addOnSliderTouchListener(
        object : Slider.OnSliderTouchListener {
          override fun onStartTrackingTouch(slider: Slider) {}

          override fun onStopTrackingTouch(slider: Slider) {
            textView.text = valueString(slider.value)
            onSliderChange(slider.value)
          }
        }
      )
    }
  }

  /** Base class for a dropdown menu, used for selecting channels and waveforms. */
  abstract class ParamDropdown(
    context: Context,
    private val dropdown: AutoCompleteTextView,
    private val items: List<String>
  ) {
    /** Override to define how to convert `value` to a base-0 items list index. */
    abstract fun valueToPosition(value: Int): Int

    /** Override to define the inverse of `valueToPosition()`. */
    abstract fun positionToValue(position: Int): Int

    /** Override to respond to when the dropdown selection changes. */
    abstract fun onChange(newValue: Int)

    /** Sets the selected value. */
    fun setValue(newValue: Int) {
      dropdown.setText(items[valueToPosition(newValue)], false)
    }

    init {
      dropdown.setAdapter(NoFilterArrayAdapter(context, R.layout.dropdown_menu_item, items))
      dropdown.setText(items[0], false)
      dropdown.setOnItemClickListener { _, _, position, _ -> onChange(positionToValue(position)) }
    }

    /**
     * An ArrayAdapter that doesn't filter, so that all items are always shown. Without this custom
     * adapter, the dropdowns will filter (despite the `false` arg in setText calls) when switching
     * between fragments.
     */
    class NoFilterArrayAdapter(context: Context, resource: Int, items: List<String>) :
      ArrayAdapter<String>(context, resource, items) {
      class NoFilter(private val items: List<String>) : Filter() {
        override fun performFiltering(constraint: CharSequence) =
          FilterResults().apply {
            values = items
            count = items.size
          }

        override fun publishResults(constraint: CharSequence, results: FilterResults) {}
      }

      private val filter = NoFilter(items)
      override fun getFilter() = filter
    }
  }

  /** Gets the color associated with a resource, with its alpha component set to `alpha`. */
  @ColorInt
  private fun getColor(@AttrRes colorRes: Int, alpha: Int): Int {
    val value = TypedValue()
    requireContext().theme.resolveAttribute(colorRes, value, true)
    val color = (value.data and 0xffffff) or (alpha shl 24)
    return color
  }

  /**
   * Enables icons in popup menu. There is no public API to add icons, but there is a workaround.
   * Source: https://material.io/components/menus/android#dropdown-menus
   */
  @SuppressLint("RestrictedApi")
  private fun enableMenuIcons(popup: PopupMenu) {
    (popup.menu as? MenuBuilder)?.apply {
      setOptionalIconsVisible(true)
      val iconMarginPx =
        TypedValue.applyDimension(
          TypedValue.COMPLEX_UNIT_DIP, 16.0f, resources.displayMetrics
        ).toInt()
      visibleItems.map {
        if (it.icon != null) {
          it.icon = InsetDrawable(it.icon, iconMarginPx, 0, iconMarginPx, 0)
        }
      }
    }
  }

  /** Swaps the selected op with `newSelectedIndex` and updates the RecyclerView. */
  private fun swapSelectedWith(newSelectedIndex: Int) {
    val oldIndex = selectedIndex
    if (bleViewModel.tactilePatternSwapOps(oldIndex, newSelectedIndex)) {
      bleViewModel.tactilePatternSelect(newSelectedIndex)
      patternOpItemAdapter.notifyItemMoved(oldIndex, newSelectedIndex)
      patternOpItemAdapter.notifyItemChanged(oldIndex)
      patternOpItemAdapter.notifyItemChanged(newSelectedIndex)
    }
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? {
    val root: View = inflater.inflate(R.layout.fragment_pattern_editor, container, false)
    val context = requireContext()
    // Get a suitable background color for highlighting the selected op.
    bgSelectedColor = getColor(android.R.attr.colorSecondary, alpha = 120)

    recyclerView = root.findViewById<RecyclerView>(R.id.pattern_recycler_view).apply {
      adapter = patternOpItemAdapter
      setHasFixedSize(false)
    }

    // File button.
    root.findViewById<Button>(R.id.pattern_editor_file).let { button ->
      button.setOnClickListener {
        PopupMenu(requireContext(), button).apply {
          setOnMenuItemClickListener(this@PatternEditorFragment)
          inflate(R.menu.pattern_editor_file)
          show()
        }
      }
    }

    // Play button.
    root.findViewById<Button>(R.id.pattern_editor_play).setOnClickListener {
      (if (pattern.ops.isEmpty()) {
        R.string.toast_pattern_empty
      } else if (!pattern.hasPlayAfterWave()) {
        R.string.toast_pattern_does_nothing
      } else if (!bleViewModel.tactilePatternPlay()) {
        R.string.toast_no_connection
      } else {
        null
      })?.let { messageId ->
        Toast.makeText(requireContext(), getString(messageId), Toast.LENGTH_SHORT).show()
      }
    }

    // Pattern add op ("+") button.
    root.findViewById<Button>(R.id.pattern_add_op).let { button ->
      button.setOnClickListener {
        PopupMenu(requireContext(), button).apply {
          enableMenuIcons(this)
          setOnMenuItemClickListener(this@PatternEditorFragment)
          inflate(R.menu.pattern_editor_add_op)
          show()
        }
      }
    }
    // Move op up button.
    root.findViewById<Button>(R.id.pattern_move_op_up).setOnClickListener {
      swapSelectedWith(selectedIndex - 1)
    }
    // Move op down button.
    root.findViewById<Button>(R.id.pattern_move_op_down).setOnClickListener {
      swapSelectedWith(selectedIndex + 1)
    }
    // Remove op button.
    root.findViewById<Button>(R.id.pattern_remove_op).setOnClickListener {
      bleViewModel.tactilePatternRemoveOp()?.let { removedIndex ->
        patternOpItemAdapter.notifyItemRemoved(removedIndex)
        // Notify so that selection highlight is redrawn.
        bleViewModel.tactilePatternSelectedIndexNotifyObservers()
      }
    }

    // Lists of strings used as dropdown menu items below.
    // The channels ["1", "2", ...].
    val channels = (1..bleViewModel.channelMap.value!!.numInputChannels).map { it.toString() }
    // The channels or all ["all", "1", "2", ...].
    val channelsOrAll = listOf("all") + channels
    // The waveforms ["sin_25_hz", ... "chirp"].
    val waveforms = TactilePattern.Waveform.values().map { it.name.lowercase() }

    // Play op settings.
    val patternPlayOp: ConstraintLayout = root.findViewById(R.id.pattern_play_op)
    val playOpDuration = object : PatternEditorFragment.ParamSlider(
      patternPlayOp.findViewById(R.id.pattern_play_op_duration_value),
      patternPlayOp.findViewById(R.id.pattern_play_op_duration_slider)
    ) {
      override fun valueString(value: Float) = "${value.roundToInt()} ms"

      override fun onSliderChange(value: Float) {
        (selectedOp as? TactilePattern.PlayOp)?.let { op ->
          op.durationMs = value.roundToInt()
          bleViewModel.tactilePatternNotifyObservers()
        }
      }
    }

    // Wave op settings.
    val patternWaveOp: ConstraintLayout = root.findViewById(R.id.pattern_wave_op)
    val waveOpChannel = object : PatternEditorFragment.ParamDropdown(
      context, patternWaveOp.findViewById(R.id.pattern_wave_op_channel_dropdown), channelsOrAll
    ) {
      override fun valueToPosition(value: Int) = value + 1
      override fun positionToValue(position: Int) = position - 1

      override fun onChange(newValue: Int) {
        (selectedOp as? TactilePattern.SetWaveformOp)?.let { op ->
          op.channel = newValue
          bleViewModel.tactilePatternNotifyObservers()
        }
      }
    }
    val waveOpWaveform = object : PatternEditorFragment.ParamDropdown(
      context, patternWaveOp.findViewById(R.id.pattern_wave_op_waveform_dropdown), waveforms
    ) {
      override fun valueToPosition(value: Int) = value
      override fun positionToValue(position: Int) = position

      override fun onChange(newValue: Int) {
        (selectedOp as? TactilePattern.SetWaveformOp)?.let { op ->
          TactilePattern.Waveform.fromInt(newValue)?.let {
            op.waveform = it
            bleViewModel.tactilePatternNotifyObservers()
          }
        }
      }
    }

    // Gain op settings.
    val patternGainOp: ConstraintLayout = root.findViewById(R.id.pattern_gain_op)
    val gainOpChannel = object : PatternEditorFragment.ParamDropdown(
      context, patternGainOp.findViewById(R.id.pattern_gain_op_channel_dropdown), channelsOrAll
    ) {
      override fun valueToPosition(value: Int) = value + 1
      override fun positionToValue(position: Int) = position - 1

      override fun onChange(newValue: Int) {
        (selectedOp as? TactilePattern.SetGainOp)?.let { op ->
          op.channel = newValue
          bleViewModel.tactilePatternNotifyObservers()
        }
      }
    }
    val gainOpGain = object : PatternEditorFragment.ParamSlider(
      patternGainOp.findViewById(R.id.pattern_gain_op_gain_value),
      patternGainOp.findViewById(R.id.pattern_gain_op_gain_slider)
    ) {
      override fun valueString(value: Float) =
        "%.3f".format(TactilePattern.SetGainOp.constrainGain(value))

      override fun onSliderChange(value: Float) {
        (selectedOp as? TactilePattern.SetGainOp)?.let { op ->
          op.gain = value
          bleViewModel.tactilePatternNotifyObservers()
        }
      }
    }

    // Move op settings.
    val patternMoveOp: ConstraintLayout = root.findViewById(R.id.pattern_move_op)
    val moveOpFrom = object : PatternEditorFragment.ParamDropdown(
      context, patternMoveOp.findViewById(R.id.pattern_move_op_from_dropdown), channels
    ) {
      override fun valueToPosition(value: Int) = value
      override fun positionToValue(position: Int) = position

      override fun onChange(newValue: Int) {
        (selectedOp as? TactilePattern.MoveOp)?.let { op ->
          op.fromChannel = newValue
          bleViewModel.tactilePatternNotifyObservers()
        }
      }
    }
    val moveOpTo = object : PatternEditorFragment.ParamDropdown(
      context, patternMoveOp.findViewById(R.id.pattern_move_op_to_dropdown), channels
    ) {
      override fun valueToPosition(value: Int) = value
      override fun positionToValue(position: Int) = position

      override fun onChange(newValue: Int) {
        (selectedOp as? TactilePattern.MoveOp)?.let { op ->
          op.toChannel = newValue
          bleViewModel.tactilePatternNotifyObservers()
        }
      }
    }

    bleViewModel.tactilePatternSelectedIndex.observe(viewLifecycleOwner) { i ->
      // Set all op settings UI to invisible.
      patternPlayOp.visibility = View.INVISIBLE
      patternWaveOp.visibility = View.INVISIBLE
      patternGainOp.visibility = View.INVISIBLE
      patternMoveOp.visibility = View.INVISIBLE

      if (i != null && i in pattern.ops.indices) {
        val op = pattern.ops[i]
        // Based on the op's type, show the corresponding UI and update it with the op's parameters.
        when (op) {
          is TactilePattern.PlayOp -> {
            patternPlayOp.visibility = View.VISIBLE
            playOpDuration.setValue(op.durationMs.toFloat())
          }
          is TactilePattern.SetWaveformOp -> {
            patternWaveOp.visibility = View.VISIBLE
            waveOpChannel.setValue(op.channel)
            waveOpWaveform.setValue(op.waveform.value)
          }
          is TactilePattern.SetGainOp -> {
            patternGainOp.visibility = View.VISIBLE
            gainOpChannel.setValue(op.channel)
            gainOpGain.setValue(op.gain)
          }
          is TactilePattern.MoveOp -> {
            patternMoveOp.visibility = View.VISIBLE
            moveOpFrom.setValue(op.fromChannel)
            moveOpTo.setValue(op.toChannel)
          }
          else -> {}
        }
      }
    }

    if (pattern.ops.isNotEmpty()) {
      patternOpItemAdapter.notifyItemRangeChanged(0, pattern.size)
    }

    return root
  }

  /** This callback is called when a file or op menu item is selected. */
  override fun onMenuItemClick(item: MenuItem): Boolean {
    when (item.itemId) {
      R.id.pattern_editor_file_new -> {
        patternOpItemAdapter.notifyItemRangeRemoved(0, pattern.size)
        bleViewModel.tactilePatternClear()
        return true
      }
      R.id.pattern_editor_file_open -> {
        openPatternLauncher.launch(arrayOf("text/plain"))
        return true
      }
      R.id.pattern_editor_file_save -> {
        savePatternLauncher.launch("pattern.txt")
        return true
      }
      else -> {
        // Insert the selected op with some default parameters.
        if (bleViewModel.tactilePatternInsertOp(
            when (item.itemId) {
              R.id.pattern_op_play_item -> TactilePattern.PlayOp(100)
              R.id.pattern_op_wave_item ->
                TactilePattern.SetWaveformOp(-1, TactilePattern.Waveform.SIN_100_HZ)
              R.id.pattern_op_gain_item -> TactilePattern.SetGainOp(-1, 0.15f)
              R.id.pattern_op_move_item -> TactilePattern.MoveOp(0, 1)
              else -> null
            }
          )
        ) {
          // Tell the RecyclerView to draw the inserted item.
          patternOpItemAdapter.notifyItemInserted(selectedIndex)
          // Scroll to the inserted item, in case the pattern is long enough that it is off screen.
          recyclerView.scrollToPosition(selectedIndex)
          return true
        }
      }
    }

    return false
  }
}
