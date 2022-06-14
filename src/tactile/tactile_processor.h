/* Copyright 2019, 2021-2022 Google LLC
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
 * Main API for our audio-to-tactile signal processing.
 *
 * `TactileProcessor` performs audio-to-tactile processing, taking a (mono)
 * audio stream as input and producing a 10-channel tactile signal as output, to
 * be presented on a 10-tactor array. There are four groups of output signals,
 * corresponding to the four `Enveloper` channels. Tactile signals are computed
 * as smoothed energy envelopes of the following bands:
 *
 *   Baseband:     80-500 Hz sensitivity to get pitch pulses, percussion, drums.
 *   Vowel:        500-3500 Hz sensitivity to get speech formants.
 *   Sh fricative: 2500-3500 Hz sensitivity for postalveolar sounds like "sh".
 *   Fricative:    4000-6000 Hz sensitivity for alveolar and other fricatives
 *                 closer to the front of the mouth like "s".
 *
 * TactileProcessor additionally runs the vowel embedding network and presents
 * its output on a hex cluster of 7 tactors. The vowel coordinate determines
 * spatial position in the cluster while the fine-time signal is determined by
 * the 500-3500 Hz energy envelope.
 *
 * The output channels are ordered as
 *
 *   0: baseband  5: eh                        (6)-(5)   sh fricative
 *   1: aa        6: ae                        /     \      (8)
 *   2: uw        7: uh               (0)    (1) (7) (4)
 *   3: ih        8: sh fricative   baseband   \     /      (9)
 *   4: iy        9: fricative                 (2)-(3)   fricative
 *                                          vowel cluster
 *
 * TactileProcessor hooks together the CARL+PCEN frontend, vowel embedding,
 * and the tactor energy envelope design.
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PROCESSOR_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PROCESSOR_H_

#include "frontend/carl_frontend.h"
#include "phonetics/embed_vowel.h"
#include "tactile/enveloper.h"
#include "tactile/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of tactors / output channels. */
extern const int kTactileProcessorNumTactors;

/* TactileProcessor parameters. */
typedef struct {
  EnveloperParams enveloper_params;
  /* Decimation factor after computing the energy envelope. */
  int decimation_factor;
  /* Parameters for the `CarlFrontend` used for vowel embedding. Particularly,
   * `frontend_params.input_sample_rate_hz` must be set to the input sample
   * rate and `frontend_params.block_size` to the desired block size.
   */
  CarlFrontendParams frontend_params;
} TactileProcessorParams;

/* Set `params` to default values. */
void TactileProcessorSetDefaultParams(TactileProcessorParams* params);

/* Get TactileProcessor's output sample rate in Hz. */
float TactileProcessorOutputSampleRateHz(const TactileProcessorParams* params);

typedef struct {
  /* Energy envelopes. */
  Enveloper enveloper;
  /* Decimation factor after computing energy envelopes. */
  int decimation_factor;
  /* Vowel embedding frontend. */
  CarlFrontend* frontend;
  /* Workspace buffer with space for `block_size` floats. */
  float* workspace;
  /* PCEN frame buffer. */
  float* frame;
  /* 2D vowel embedding coordinate. */
  float vowel_coord[2];
  /* Interpolation weights for the hexagonal vowel cluster. */
  float vowel_hex_weights[7];
} TactileProcessor;

/* Makes a `TactileProcessor`. The caller should free it when done with
 * `TactileProcessorFree`. The input sample rate is read from
 * `params.frontend_params.input_sample_rate_hz`. Returns NULL on failure.
 */
TactileProcessor* TactileProcessorMake(TactileProcessorParams* params);

/* Frees a `TactileProcessor`. */
void TactileProcessorFree(TactileProcessor* processor);

/* Resets to initial state. */
void TactileProcessorReset(TactileProcessor* processor);

/* Runs the `TactileProcessor` in a streaming manner. `input` is an array of
 * `block_size` elements and `output` is an array of
 * `kTactileProcessorNumTactors * block_size / decimation_factor` elements. The
 * input and output both have sample rate
 * `params.frontend_params.input_sample_rate_hz`.
 */
void TactileProcessorProcessSamples(TactileProcessor* processor,
    const float* input, float* output);

/* Applies tuning specified by `knobs`. May be called at any time. */
void TactileProcessorApplyTuning(TactileProcessor* processor,
                                 const TuningKnobs* tuning_knobs);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_PROCESSOR_H_ */
