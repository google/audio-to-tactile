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

/**
 * This class contains the information about sound events and corresponding tactile patterns. Right
 * now this class is just placeholder for UI.
 */
class SoundEvents {

  companion object {
    val SOUND_EVENTS =
      arrayOf<String>(
        "Bird sounds",
        "Music",
        "Speech",
        "No Speech",
        "Dog sounds",
        "Knock",
        "Siren",
        "Alarm",
        "Baby Cry",
        "Cat",
        "Wind",
        "Water",
        "Doorbell",
      )
    val TACTILE_PATTERNS =
      arrayOf<String>(
        "left short ",
        "right short",
        "top short",
        "bottom short",
        "left long",
        "right long",
        "top long",
        "bottom long",
        "Movement left ",
        "Movement right",
        "Intensity increase",
        "Intensity decrease",
        "Pulsing pattern",
      )
  }
}
