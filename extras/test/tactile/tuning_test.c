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
 */

#include "src/tactile/tuning.h"

#include "src/dsp/logging.h"
#include "src/tactile/energy_envelope.h"
#include "src/tactile/tactile_processor.h"

/* Checks that `actual` is within 0.3% of `expected`. */
static int IsClose(float expected, float actual) {
  if (fabs(expected - actual) > 0.003f * fabs(expected)) {
    fprintf(stderr,
            "Error: Expected actual within 0.3%% of expected, got\n"
            "  actual = %g\n  expected = %g\n",
            expected, actual);
    return 0;
  }
  return 1;
}

static void TestTuningMapControlValue(int knob, int value, float expected) {
  printf("TestTuningMapControlValue(%d, %d)\n", knob, value);
  float mapped_value = TuningMapControlValue(knob, value);
  CHECK(IsClose(expected, mapped_value));
}

static void TestTactileProcessorApplyTuning() {
  puts("TestTactileProcessorApplyTuning");
  int trial;
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  TactileProcessor* processor = CHECK_NOTNULL(TactileProcessorMake(&params));

  TuningKnobs tuning_knobs;
  tuning_knobs = kDefaultTuningKnobs; /* First trial tests the default knobs. */

  for (trial = 0; trial < 3; ++trial) {
    float mapped[kNumTuningKnobs];
    int knob;
    for (knob = 0; knob < kNumTuningKnobs; ++knob) {
      mapped[knob] = TuningMapControlValue(knob, tuning_knobs.values[knob]);
    }

    TactileProcessorApplyTuning(processor, &tuning_knobs);

    int i;
    for (i = 0; i < 4; ++i) {
      CHECK(IsClose(pow(10.0f, mapped[kKnobOutputGain] / 20.0f),
                     processor->channel_states[i].output_gain));
      CHECK(mapped[kKnobDenoising0 + i] ==
            processor->channel_states[i].denoise_thresh_factor);
      CHECK(-mapped[kKnobAgcStrength] ==
            processor->channel_states[i].agc_exponent);
      CHECK(EnergyEnvelopeSmootherCoeff(&processor->channel_states[i],
                                        mapped[kKnobNoiseTau]) ==
            processor->channel_states[i].noise_smoother_coeff);
      CHECK(EnergyEnvelopeSmootherCoeff(&processor->channel_states[i],
                                        mapped[kKnobGainTauRelease]) ==
            processor->channel_states[i].gain_smoother_coeffs[1]);
      CHECK(mapped[kKnobCompressor] ==
            processor->channel_states[i].compressor_exponent);
    }

    /* Subsequent trials test random knob values. */
    for (knob = 0; knob < kNumTuningKnobs; ++knob) {
      tuning_knobs.values[knob] = rand() % 256;
    }
  }

  TactileProcessorFree(processor);
}

static void TestTuningGetInputGain() {
  puts("TestTuningGetInputGain");
  TuningKnobs tuning_knobs = kDefaultTuningKnobs;
  CHECK(IsClose(1.0f, TuningGetInputGain(&tuning_knobs)));

  tuning_knobs.values[kKnobInputGain] = 63;
  CHECK(IsClose(0.098f, TuningGetInputGain(&tuning_knobs)));
}

int main(int argc, char** argv) {
  srand(0);
  TestTuningMapControlValue(kKnobInputGain, 0, -40.0f);
  TestTuningMapControlValue(kKnobInputGain, 255, 40.315f);
  TestTuningMapControlValue(kKnobOutputGain, 0, -18.0f);
  TestTuningMapControlValue(kKnobOutputGain, 191, -0.0235f);
  TestTuningMapControlValue(kKnobOutputGain, 255, 6.0f);
  TestTuningMapControlValue(kKnobDenoising0, 0, 2.0f);
  TestTuningMapControlValue(kKnobDenoising0, 89, 10.0f);
  TestTuningMapControlValue(kKnobDenoising0, 255, 200.0f);
  TestTuningMapControlValue(kKnobAgcStrength, 0, 0.1f);
  TestTuningMapControlValue(kKnobAgcStrength, 191, 0.7f);
  TestTuningMapControlValue(kKnobAgcStrength, 255, 0.9f);
  TestTuningMapControlValue(kKnobNoiseTau, 0, 0.04f);
  TestTuningMapControlValue(kKnobNoiseTau, 127, 0.3964f);
  TestTuningMapControlValue(kKnobNoiseTau, 255, 4.0f);
  TestTuningMapControlValue(kKnobGainTauRelease, 0, 0.04f);
  TestTuningMapControlValue(kKnobGainTauRelease, 73, 0.1495f);
  TestTuningMapControlValue(kKnobGainTauRelease, 255, 4.0f);
  TestTuningMapControlValue(kKnobCompressor, 0, 0.1f);
  TestTuningMapControlValue(kKnobCompressor, 96, 0.2506f);
  TestTuningMapControlValue(kKnobCompressor, 255, 0.5f);

  TestTactileProcessorApplyTuning();
  TestTuningGetInputGain();

  puts("PASS");
  return EXIT_SUCCESS;
}
