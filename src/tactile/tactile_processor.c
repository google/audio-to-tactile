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
 */

#include "tactile/tactile_processor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/decibels.h"
#include "phonetics/hexagon_interpolation.h"

const int kTactileProcessorNumTactors = 10;

void TactileProcessorSetDefaultParams(TactileProcessorParams* params) {
  if (params) {
    params->enveloper_params = kDefaultEnveloperParams;
    params->decimation_factor = 1;
    params->frontend_params = kCarlFrontendDefaultParams;
  }
}

float TactileProcessorOutputSampleRateHz(const TactileProcessorParams* params) {
  return params->frontend_params.input_sample_rate_hz
      / params->decimation_factor;
}

TactileProcessor* TactileProcessorMake(TactileProcessorParams* params) {
  if (params == NULL) { return NULL; }

  TactileProcessor* processor = (TactileProcessor*)malloc(
      sizeof(TactileProcessor));
  if (processor == NULL) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    goto fail;
  }
  processor->frontend = NULL;
  processor->workspace = NULL;
  processor->frame = NULL;
  int i;
  for (i = 0; i < 7; ++i) {
    processor->vowel_hex_weights[i] = 0.0f;
  }

  const int sample_rate_hz = params->frontend_params.input_sample_rate_hz;
  processor->decimation_factor = params->decimation_factor;

  /* Create Enveloper. */
  if (!EnveloperInit(&processor->enveloper, &params->enveloper_params,
                     sample_rate_hz, params->decimation_factor)) {
    fprintf(stderr, "Error: EnveloperInit failed.\n");
    goto fail;
  }

  /* Create CarlFrontend. */
  const int block_size = params->frontend_params.block_size;
  const int decimated_block_size = block_size / params->decimation_factor;
  if (block_size % params->decimation_factor != 0) {
    fprintf(stderr, "Error: block_size must be an "
            "integer multiple of decimation_factor.\n");
    goto fail;
  }
  processor->frontend = CarlFrontendMake(&params->frontend_params);
  if (processor->frontend == NULL) {
    fprintf(stderr, "Error: CarlFrontendMake failed.\n");
    goto fail;
  }
  int workspace_size = kEnveloperNumChannels * decimated_block_size;
  if (workspace_size < block_size) { workspace_size = block_size; }
  processor->workspace = (float*)malloc(workspace_size * sizeof(float));
  processor->frame = (float*)malloc(
      sizeof(float) * CarlFrontendNumChannels(processor->frontend));
  if (processor->workspace == NULL || processor->frame == NULL) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    goto fail;
  }

  return processor;

fail:
  TactileProcessorFree(processor);
  return NULL;
}

void TactileProcessorFree(TactileProcessor* processor) {
  if (processor) {
    free(processor->frame);
    free(processor->workspace);
    CarlFrontendFree(processor->frontend);
    free(processor);
  }
}

void TactileProcessorReset(TactileProcessor* processor) {
  EnveloperReset(&processor->enveloper);
  CarlFrontendReset(processor->frontend);
  int i;
  for (i = 0; i < 7; ++i) {
    processor->vowel_hex_weights[i] = 0.0f;
  }
}

void TactileProcessorProcessSamples(TactileProcessor* processor,
    const float* input, float* output) {
  const int block_size = CarlFrontendBlockSize(processor->frontend);
  const int decimation_factor = processor->decimation_factor;
  const int decimated_block_size = block_size / decimation_factor;
  /* Run the CARL frontend. */
  float* workspace = processor->workspace;
  memcpy(workspace, input, sizeof(float) * block_size);
  CarlFrontendProcessSamples(processor->frontend, workspace, processor->frame);
  /* Get 2-D vowel space coordinate. */
  float vowel_coord[2];
  EmbedVowel(processor->frame, vowel_coord);

  /* Compute energy envelopes, writing into `workspace`. */
  EnveloperProcessSamples(&processor->enveloper, input, block_size, workspace);

  const float* src = workspace;
  float* dest = output;
  int i;
  for (i = 0; i < decimated_block_size; ++i) {
    dest[0] = src[0]; /* Map baseband envelope to output channel 0. */
    dest[8] = src[2]; /* Map sh fricative envelope to output channel 8. */
    dest[9] = src[3]; /* Map fricative envelope to output channel 9. */
    src += kEnveloperNumChannels;
    dest += kTactileProcessorNumTactors;
  }

  float* vowel_hex_weights = processor->vowel_hex_weights;
  float next_vowel_hex_weights[7];
  /* Get the next hexagonal interpolation weights based on `vowel_coord`. The
   * fine-time signal is modulated by the hex weights.
   */
  GetHexagonInterpolationWeights(vowel_coord[0], vowel_coord[1],
      next_vowel_hex_weights);
  /* We will blend linearly from `vowel_hex_weights` to `next_vowel_hex_weights`
   * over the block.
   */
  float weights_diff[7];
  int c;
  for (c = 0; c < 7; ++c) {
    weights_diff[c] = next_vowel_hex_weights[c] - vowel_hex_weights[c];
  }

  /* Map to the vowel hex cluster. */
  const float blend_step = 1.0f / decimated_block_size;
  float blend = 0.0f;
  src = workspace + 1; /* Get the vowel channel energy envelope. */
  dest = output + 1;
  for (i = 0; i < decimated_block_size; ++i) {
    blend += blend_step;
    const float sample = *src; /* Get the next fine-time sample. */
    int c;
    for (c = 0; c < 7; ++c) {  /* Fill the vowel channels. */
      dest[c] = (vowel_hex_weights[c] + blend * weights_diff[c]) * sample;
    }
    src += kEnveloperNumChannels;
    dest += kTactileProcessorNumTactors;
  }

  memcpy(vowel_hex_weights, next_vowel_hex_weights,
         sizeof(next_vowel_hex_weights));
}

void TactileProcessorApplyTuning(TactileProcessor* processor,
                                 const TuningKnobs* knobs) {
  const float output_gain_db = TuningGet(knobs, kKnobOutputGain);
  /* Convert dB to linear amplitude ratio. */
  const float output_gain = FastDecibelsToAmplitudeRatio(output_gain_db);
  const float energy_tau_s = TuningGet(knobs, kKnobEnergyTau);
  const float noise_db_s = TuningGet(knobs, kKnobNoiseAdaptation);
  const float denoising_strength = TuningGet(knobs, kKnobDenoisingStrength);
  const float denoising_transition_db =
      TuningGet(knobs, kKnobDenoisingTransition);
  const float agc_strength = TuningGet(knobs, kKnobAgcStrength);
  const float compressor_exponent = TuningGet(knobs, kKnobCompressor);

  Enveloper* enveloper = &processor->enveloper;
  enveloper->energy_smoother_coeff =
      EnveloperSmootherCoeff(enveloper, energy_tau_s);
  enveloper->noise_coeffs[1] =
      EnveloperGrowthCoeff(enveloper, noise_db_s);
  enveloper->gate_thresh_factor = denoising_strength;
  enveloper->gate_transition_factor =
      DecibelsToPowerRatio(denoising_transition_db);
  enveloper->agc_exponent = -agc_strength;
  enveloper->compressor_exponent = compressor_exponent;

  int c;
  for (c = 0; c < kEnveloperNumChannels; ++c) {
    EnveloperChannel* enveloper_c = &enveloper->channels[c];
    enveloper_c->output_gain = output_gain;
  }

  EnveloperUpdatePrecomputedParams(enveloper);
}
