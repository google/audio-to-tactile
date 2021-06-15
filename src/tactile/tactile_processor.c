/* Copyright 2019, 2021 Google LLC
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

#include "dsp/fast_fun.h"
#include "phonetics/hexagon_interpolation.h"

const int kTactileProcessorNumTactors = 10;

void TactileProcessorSetDefaultParams(TactileProcessorParams* params) {
  if (params) {
    params->baseband_channel_params = kEnergyEnvelopeBasebandParams;
    params->vowel_channel_params = kEnergyEnvelopeVowelParams;
    params->sh_fricative_channel_params = kEnergyEnvelopeShFricativeParams;
    params->fricative_channel_params = kEnergyEnvelopeFricativeParams;
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

  /* Create EnergyEnvelopes. */
  if (!EnergyEnvelopeInit(&processor->channel_states[0],
                          &params->baseband_channel_params,
                          sample_rate_hz, params->decimation_factor) ||
      !EnergyEnvelopeInit(&processor->channel_states[1],
                          &params->vowel_channel_params,
                          sample_rate_hz, params->decimation_factor) ||
      !EnergyEnvelopeInit(&processor->channel_states[2],
                          &params->sh_fricative_channel_params,
                          sample_rate_hz, params->decimation_factor) ||
      !EnergyEnvelopeInit(&processor->channel_states[3],
                          &params->fricative_channel_params,
                          sample_rate_hz, params->decimation_factor)) {
    fprintf(stderr, "Error: EnergyEnvelopeInit failed.\n");
    goto fail;
  }

  /* Create CarlFrontend. */
  const int block_size = params->frontend_params.block_size;
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
  processor->workspace = (float*)malloc(sizeof(float) * block_size);
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
  int i;
  for (i = 0; i < 4; ++i) {
    EnergyEnvelopeReset(&processor->channel_states[i]);
  }
  CarlFrontendReset(processor->frontend);
  for (i = 0; i < 7; ++i) {
    processor->vowel_hex_weights[i] = 0.0f;
  }
}

void TactileProcessorProcessSamples(TactileProcessor* processor,
    const float* input, float* output) {
  const int block_size = CarlFrontendBlockSize(processor->frontend);
  const int decimation_factor = processor->decimation_factor;
  /* Run the CARL frontend. */
  float* workspace = processor->workspace;
  memcpy(workspace, input, sizeof(float) * block_size);
  CarlFrontendProcessSamples(processor->frontend, workspace, processor->frame);
  /* Get 2-D vowel space coordinate. */
  float vowel_coord[2];
  EmbedVowel(processor->frame, vowel_coord);

  /* Run vowel channel to get the fine-time signal. */
  EnergyEnvelopeProcessSamples(&processor->channel_states[1], input, block_size,
                               workspace, 1);

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
  const int decimated_block_size = block_size / decimation_factor;
  const float blend_step = 1.0f / decimated_block_size;
  float blend = 0.0f;
  float* dest = output + 1;
  int i;
  for (i = 0; i < decimated_block_size; ++i) {
    blend += blend_step;
    const float sample = workspace[i];  /* Get the next fine-time sample. */
    int c;
    for (c = 0; c < 7; ++c) {  /* Fill the vowel channels. */
      dest[c] = (vowel_hex_weights[c] + blend * weights_diff[c]) * sample;
    }
    dest += kTactileProcessorNumTactors;
  }

  /* Run baseband and fricative channels. */
  EnergyEnvelopeProcessSamples(&processor->channel_states[0], input, block_size,
                               output, kTactileProcessorNumTactors);
  EnergyEnvelopeProcessSamples(&processor->channel_states[2], input, block_size,
                               output + 8, kTactileProcessorNumTactors);
  EnergyEnvelopeProcessSamples(&processor->channel_states[3], input, block_size,
                               output + 9, kTactileProcessorNumTactors);

  memcpy(vowel_hex_weights, next_vowel_hex_weights,
         sizeof(next_vowel_hex_weights));
}

void TactileProcessorApplyTuning(TactileProcessor* processor,
                                 const TuningKnobs* knobs) {
  const float output_gain_db =
      TuningMapControlValue(kKnobOutputGain, knobs->values[kKnobOutputGain]);
  /* Convert dB to linear amplitude ratio. */
  const float output_gain = FastExp2((float)(M_LN10 / (20.0 * M_LN2)) *
                                     output_gain_db);
  const float agc_strength =
      TuningMapControlValue(kKnobAgcStrength, knobs->values[kKnobAgcStrength]);
  const float noise_tau =
      TuningMapControlValue(kKnobNoiseTau, knobs->values[kKnobNoiseTau]);
  const float gain_tau_release = TuningMapControlValue(
      kKnobGainTauRelease, knobs->values[kKnobGainTauRelease]);
  const float compressor_exponent =
      TuningMapControlValue(kKnobCompressor, knobs->values[kKnobCompressor]);

  int i;
  for (i = 0; i < 4; ++i) {  /* Update each EnergyEvelope instance. */
    const float denoise_thresh_factor = TuningMapControlValue(
        kKnobDenoising0 + i, knobs->values[kKnobDenoising0 + i]);

    processor->channel_states[i].output_gain = output_gain;
    processor->channel_states[i].denoise_thresh_factor = denoise_thresh_factor;
    processor->channel_states[i].agc_exponent = -agc_strength;
    processor->channel_states[i].noise_smoother_coeff =
        EnergyEnvelopeSmootherCoeff(&processor->channel_states[i], noise_tau);
    processor->channel_states[i].gain_smoother_coeffs[1] =
        EnergyEnvelopeSmootherCoeff(&processor->channel_states[i],
                                    gain_tau_release);
    processor->channel_states[i].compressor_exponent = compressor_exponent;

    EnergyEnvelopeUpdatePrecomputedParams(&processor->channel_states[i]);
  }
}
