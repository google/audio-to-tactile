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

#include "tactile/tuning.h"

#include "dsp/fast_fun.h"

const TuningKnobs kDefaultTuningKnobs = {{
  /* kKnobInputGain            */ 127,
  /* kKnobOutputGain           */ 191,
  /* kKnobDenoisingBaseband    */ 155,
  /* kKnobDenoisingVowel       */  67,
  /* kKnobDenoisingShFricative */  67,
  /* kKnobDenoisingFricative   */  67,
  /* kKnobCrossChannelTau      */  51,
  /* kKnobAgcStrength          */ 191,
  /* kKnobNoiseTau             */ 178,
  /* kKnobGainTauRelease       */  73,
  /* kKnobCompressor           */  96,
}};

const TuningKnobInfo kTuningKnobInfo[kNumTuningKnobs] = {
    { /* kKnobInputGain */
        "Input gain",
        "%.1f dB",
        "Gain applied to the input audio before any processing.",
        kTuningMapLinearly,
        -40.0f, /* In units of dB. */
        40.315f,
    },
    { /* kKnobOutputGain */
        "Output gain",
        "%.1f dB",
        "The `output_gain` parameter of all Enveloper channels.",
        kTuningMapLinearly,
        -18.0f, /* In units of dB. */
        6.0f,
    },
    { /* kKnobDenoisingBaseband */
        "Denoising baseband",
        "%.1f",
        "The `denoise_thresh_factor` parameter for the baseband (0) "
        "Enveloper channel. Larger value implies stronger denoising.",
        kTuningMapLogarithmically,
        1.0f,
        200.0f,
    },
    { /* kKnobDenoisingVowel */
        "Denoising vowel",
        "%.1f",
        "The `denoise_thresh_factor` parameter for the vowel (1) "
        "Enveloper channel. Larger value implies stronger denoising.",
        kTuningMapLogarithmically,
        1.0f,
        200.0f,
    },
    { /* kKnobDenoisingShFricative */
        "Denoising sh fricative",
        "%.1f",
        "The `denoise_thresh_factor` parameter for the sh fricative (2) "
        "Enveloper channel. Larger value implies stronger denoising.",
        kTuningMapLogarithmically,
        1.0f,
        200.0f,
    },
    { /* kKnobDenoisingFricative */
        "Denoising fricative",
        "%.1f",
        "The `denoise_thresh_factor` parameter for the fricative (3) "
        "Enveloper channel. Larger value implies stronger denoising.",
        kTuningMapLogarithmically,
        1.0f,
        200.0f,
    },
    { /* kKnobCrossChannelTau */
        "Cross-channel time constant",
        "%.2f s",
        "The `cross_channel_tau_s` parameter. Smaller value implies faster "
        "diffusion and stronger cross-channel inhibition.",
        kTuningMapLogarithmically,
        0.04f, /* In units of seconds. */
        4.0f,
    },
    { /* kKnobAgcStrength */
        "AGC strength",
        "%.2f",
        "Auto gain control (AGC) strength for all Enveloper channels. "
        "Larger value implies stronger normalization.",
        kTuningMapLinearly,
        0.1f,
        0.9f,
    },
    { /* kKnobNoiseTau */
        "Noise time constant",
        "%.2f s",
        "The `noise_tau_s` parameter for all Enveloper channels. Specifies "
        "how quickly the estimate of the noise level adapts.",
        kTuningMapLogarithmically,
        0.04f, /* In units of seconds. */
        4.0f,
    },
    { /* kKnobGainTauRelease */
        "Gain release time constant",
        "%.2f s",
        "The `gain_tau_release_s` parameter for all Enveloper channels.",
        kTuningMapLogarithmically,
        0.04f, /* In units of seconds. */
        4.0f,
    },
    { /* kKnobCompressor */
        "Compressor",
        "%.2f",
        "The `compressor_exponent` parameter for all Enveloper channels. "
        "Smaller value implies stronger compression.",
        kTuningMapLinearly,
        0.1f,
        0.5f,
    },
};

/* Maps `value_in` in [0, 255] linearly in [min_out, max_out]. */
static float MapLinearly(int value_in, float min_out, float max_out) {
  return min_out + ((max_out - min_out) / 255.0f) * value_in;
}

/* Maps `value_in` in [0, 255] logarithmically in [min_out, max_out]. */
static float MapLogarithmically(int value_in, float min_out, float max_out) {
  return FastExp2(MapLinearly(value_in, FastLog2(min_out), FastLog2(max_out)));
}

float TuningMapControlValue(int knob, int value) {
  if (!(0 <= knob && knob < kNumTuningKnobs)) { return 0.0f; }
  const TuningKnobInfo* info = &kTuningKnobInfo[knob];

  /* Since MapLogarthimically uses FastLog2 and FastExp2, the mapping has about
   * 0.3% numerical error. This is not a problem for algorithm behavior, but
   * noticeable in UI display at the endpoints. So we handle the endpoints
   * separately to ensure they map exactly to `min_value` and `max_value`.
   */
  if (value <= 0) {
    return info->min_value;
  } else if (value >= 255) {
    return info->max_value;
  }

  /* Map a control value in [1, 254]. */
  switch (info->map_method) {
    case kTuningMapLinearly:
      return MapLinearly(value, info->min_value, info->max_value);

    case kTuningMapLogarithmically:
      return MapLogarithmically(value, info->min_value, info->max_value);
  }

  /* While the above switch is exhaustive, this return suppresses a "control
   * reaches end of non-void function" warning with GCC.
   */
  return 0;
}

float TuningGetInputGain(const TuningKnobs* tuning) {
  const float input_gain_db =
      TuningMapControlValue(kKnobInputGain, tuning->values[kKnobInputGain]);
  /* Convert dB to linear amplitude ratio. */
  return FastExp2((float)(M_LN10 / (20.0 * M_LN2)) * input_gain_db);
}
