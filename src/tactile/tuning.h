/* Copyright 2020-2021 Google LLC
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
 *
 *
 * Functions for tuning TactileProcessor's noise robustness vs. sensitivity.
 *
 * The idea is the tuning functions below may be called in response to button
 * presses on the puck or BLE web app to adjust noise robustness vs. sensitivity
 * while it is running. To simplify the calling code, each parameter is
 * represented with an integer control value between 0 and 255, which is then
 * mapped in a way that makes sense for that parameter.
 *
 * Example use:
 *   TuningKnobs tuning_knobs = kDefaultTuningKnobs;
 *   // Make changes to the knobs...
 *   tuning_knobs.values[kKnobOutputGain] = 100;
 *   // Apply the new settings.
 *   TactileProcessorApplyTuning(tactile_processor, &tuning_knobs);
 */

#ifndef THIRD_PARTY_AUDIO_TO_TACTILE_SRC_TACTILE_TUNING_H_
#define THIRD_PARTY_AUDIO_TO_TACTILE_SRC_TACTILE_TUNING_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Constants for the meaning of each array entry in `TuningKnobs::values`. For
 * instance, the AGC strength control value is `values[kKnobAgcStrength]`.
 */
enum {
  /* Input gain applied before any processing. The range is about -40 dB to
   * +40 dB with control value = 127 corresponding to 0 dB (unit gain).
   *
   * NOTE: Unlike other tuning knobs, TactileProcessor itself does not use the
   * input gain. Instead, the mapped input gain should be read with
   *
   *   float gain = TuningGetInputGain(&tuning);
   *
   * and applied to the input audio before calling TactileProcessor. This way,
   * this gain affects all consumers of the input audio, and the gain can be
   * absorbed into the scale factor in int16->float conversion.
   */
  kKnobInputGain,

  /* The `output_gain` parameter of all Enveloper channels based on a control
   * value between 0 and 255. The control is such that value = 0 corresponds to
   * -18 dB gain, value = 191 to +0 dB, and value = 255 to +6 dB.
   */
  kKnobOutputGain,

  /* The `denoise_thresh_factor` parameter, separately for each Enveloper
   * channel. It logarithmically maps control values to the range [1.0, 200.0].
   */
  kKnobDenoisingBaseband,
  kKnobDenoisingVowel,
  kKnobDenoisingShFricative,
  kKnobDenoisingFricative,

  /* Time constant for diffusing energy envelopes across channels. It
   * logarithmically maps control values to the range [0.04, 4.0], with value =
   * 51 corresponding to tau = 0.1 s.
   */
  kKnobCrossChannelTau,

  /* The `agc_strength` parameter of all Enveloper channels. It linearly maps
   * control values to the range [0.1, 0.9], with value = 191 corresponding to
   * strength = 0.7.
   */
  kKnobAgcStrength,

  /* The `noise_tau_s` parameter of all Enveloper channels. It logarithmically
   * maps control values to the range [0.04, 4.0], with value = 127
   * corresponding to tau = 0.4 s.
   */
  kKnobNoiseTau,

  /* The `gain_tau_release_s` parameter of all Enveloper channels. It
   * logarithmically maps control values to the range [0.04, 4.0], with value =
   * 73 corresponding to tau = 0.15 s.
   */
  kKnobGainTauRelease,

  /* The `compressor_exponent` parameter of all Enveloper channels. It linearly
   * maps control values to exponents in the range [0.1, 0.5], with value = 96
   * corresponding to exponent = 0.25.
   */
  kKnobCompressor,

  /* Sentinel, must always be the last enum entry. */
  kNumTuningKnobs,
};

/* A struct of TactileProcessor tuning settings.
 * NOTE: This struct is just an array of bytes, so it can be trivially
 * serialized/deserialized by memcpy'ing the `values` array.
 */
typedef struct {
  uint8_t values[kNumTuningKnobs];
} TuningKnobs;
extern const TuningKnobs kDefaultTuningKnobs;

typedef enum {
  kTuningMapLinearly,
  kTuningMapLogarithmically,
} TuningMapMethod;

/* Struct of info about a tuning knob. */
typedef struct {
  /* A brief name to identify the knob, useful for UI display. */
  const char* name;
  /* Format string for displaying the mapped knob value, including units. */
  const char* format;
  /* Description string, describing the algorithm meaning or effect of the knob.
   * To fit in UI displays, this may be up to about 100 characters.
   */
  const char* description;
  /* Method for mapping the control value. */
  TuningMapMethod map_method;
  /* Knob's min mapped value, corresponding to control value 0. */
  float min_value;
  /* Knob's max mapped value, corresponding to control value 255. */
  float max_value;
} TuningKnobInfo;
extern const TuningKnobInfo kTuningKnobInfo[kNumTuningKnobs];

/* Maps control values to float parameters in the same way that TuningApply()
 * does. The `knob` arg is a tuning knob index from 0 to kNumTuningKnobs - 1,
 * and `value` is a control value between 0 and 255.
 *
 * The meaning of the returned float value depends on the knob:
 *
 *   `knob`                return value meaning
 *   kKnobInputGain        Gain with units of dB.
 *   kKnobOutputGain       Gain with units of dB.
 *   kKnobDenosing*        Scale factor.
 *   kKnobCrossChannelTau  Time constant with units of seconds.
 *   kKnobAgcStrength      Exponent, 0 => bypass, 1 => full normalization.
 *   kKnobNoiseTau         Time constant with units of seconds.
 *   kKnobGainTauRelease   Time constant with units of seconds.
 *   kKnobCompressor       Exponent, smaller implies stronger compression.
 */
float TuningMapControlValue(int knob, int value);

/* Gets the input gain as a linear amplitude ratio. */
float TuningGetInputGain(const TuningKnobs* tuning);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif  /* THIRD_PARTY_AUDIO_TO_TACTILE_SRC_TACTILE_TUNING_H_ */
