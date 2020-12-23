/* Copyright 2019 Google LLC
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

#include "frontend/carl_frontend.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/fast_fun.h"
#include "dsp/math_constants.h"
#include "frontend/carl_frontend_design.h"

const CarlFrontendParams kCarlFrontendDefaultParams = {
  /*input_sample_rate_hz=*/16000.0,
  /*block_size=*/64,
  /*highest_pole_frequency_hz=*/7000.0,
  /*min_pole_frequency_hz=*/100.0,
  /*step_erbs=*/0.5,
  /*envelope_cutoff_hz=*/20.0f,
  /*pcen_time_constant_s=*/0.3f,
  /*pcen_cross_channel_diffusivity=*/100.0f,
  /*pcen_init_value=*/1e-7f,
  /*pcen_alpha=*/0.7f,
  /*pcen_beta=*/0.2f,
  /*pcen_gamma=*/1e-12f,
  /*pcen_delta=*/0.001f,
};

/* Computes Z-plane pole location for a Gamma filter with specified cutoff. */
static double ComputeGammaFilterZPole(int order,
                                      float cutoff_frequency_hz,
                                      float sample_rate_hz) {
  const double single_stage_power = pow(0.5, 1.0 / order);
  const double theta = cutoff_frequency_hz * 2.0 * M_PI / sample_rate_hz;
  const double q = (1.0 - single_stage_power * cos(theta))
      / (1.0 - single_stage_power);
  return q - sqrt(q * q - 1.0);
}

/* Counts how many channels will be generated. */
static int CountNumChannels(const CarlFrontendParams* params) {
  int num_channels = 0;
  double pole;
  for (pole = params->highest_pole_frequency_hz;
       pole > params->min_pole_frequency_hz;
       pole = CarlFrontendNextAuditoryFrequency(pole, params->step_erbs)) {
    ++num_channels;
  }
  return num_channels;
}

CarlFrontend* CarlFrontendMake(const CarlFrontendParams* params) {
  /* Check that parameters are valid. */
  if (params == NULL ||
      !(params->input_sample_rate_hz > 0.0f) ||
      !(params->min_pole_frequency_hz < params->highest_pole_frequency_hz) ||
      !(params->highest_pole_frequency_hz < params->input_sample_rate_hz / 2) ||
      !(params->step_erbs > 0.01f) ||
      !(params->envelope_cutoff_hz > 0.0f) ||
      !(params->pcen_time_constant_s > 0.0f) ||
      !(params->pcen_cross_channel_diffusivity >= 0.0f) ||
      !(params->pcen_init_value >= 0.0f) ||
      !(0.0f <= params->pcen_alpha && params->pcen_alpha <= 1.0f) ||
      !(params->pcen_beta > 0.0f) ||
      !(params->pcen_gamma > 0.0f) ||
      !(params->pcen_delta > 0.0f)) {
    fprintf(stderr, "CarlFrontendMake: Invalid CarlFrontendParams.\n");
    return NULL;
  } else if (!(params->block_size >= 1) ||
             !((params->block_size & (params->block_size - 1)) == 0)) {
    fprintf(stderr, "CarlFrontendMake: block_size must be a power of 2.\n");
    return NULL;
  }

  const double output_sample_rate_hz =
      params->input_sample_rate_hz / params->block_size;
  const int num_channels = CountNumChannels(params);

  if (params->envelope_cutoff_hz >= output_sample_rate_hz / 2) {
    fprintf(stderr, "CarlFrontendMake: envelope_cutoff_hz=%g "
            "too large for output sample rate %gHz.\n",
            params->envelope_cutoff_hz, output_sample_rate_hz);
    return NULL;
  } else if (params->pcen_cross_channel_diffusivity >=
             output_sample_rate_hz / 2) {
    fprintf(stderr, "CarlFrontendMake: pcen_cross_channel_diffusivity=%g "
            "too large for output sample rate %gHz.\n",
            params->pcen_cross_channel_diffusivity, output_sample_rate_hz);
    return NULL;
  } else if (!(num_channels >= 2)) {
    fprintf(stderr, "CarlFrontendMake: Must have at least 2 channels.\n");
    return NULL;
  }

  /* Allocate CarlFrontend struct. */
  CarlFrontend* frontend = (CarlFrontend*)malloc(sizeof(CarlFrontend));
  if (frontend == NULL) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return NULL;
  }
  frontend->channel_data = NULL;
  frontend->channel_state = NULL;
  /* Allocate per-channel structures. */
  frontend->channel_data = (CarlFrontendChannelData*)malloc(
      sizeof(CarlFrontendChannelData) * num_channels);
  frontend->channel_state = (CarlFrontendChannelState*)malloc(
      sizeof(CarlFrontendChannelState) * num_channels);
  if (frontend->channel_data == NULL || frontend->channel_state == NULL) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    CarlFrontendFree(frontend);
    return NULL;
  }

  frontend->num_channels = num_channels;
  frontend->block_size = params->block_size;

  /* Compute pcen_smoother_coeff from time constant. */
  frontend->pcen_smoother_coeff = (float)(1.0 - exp(
    -1.0 / (params->pcen_time_constant_s * output_sample_rate_hz)));
  /* Compute pcen_cross_channel_smoother_coeff from diffusivity param. */
  frontend->pcen_cross_channel_smoother_coeff =
      params->pcen_cross_channel_diffusivity / output_sample_rate_hz;

  frontend->pcen_init_value = params->pcen_init_value;
  frontend->pcen_alpha = params->pcen_alpha;
  frontend->pcen_beta = params->pcen_beta;
  frontend->pcen_gamma = params->pcen_gamma;
  frontend->pcen_delta = params->pcen_delta;
  frontend->pcen_offset = FastPow(params->pcen_delta, params->pcen_beta);

  double pole = params->highest_pole_frequency_hz;
  double sample_rate_hz = params->input_sample_rate_hz;
  int c = 0;

  /* Iterate channels, starting with the highest frequency and going down. */
  for (c = 0; c < num_channels; ++c) {
    const double kMaxSamplesPerCycle = 12.0;
    /* Decimate by factor 2 if possible before the next filter. */
    if (pole < params->highest_pole_frequency_hz &&
        pole * kMaxSamplesPerCycle < sample_rate_hz &&
        sample_rate_hz >= 2 * output_sample_rate_hz) {
      sample_rate_hz /= 2.0;
      frontend->channel_data[c].should_decimate = 1;
    } else {
      frontend->channel_data[c].should_decimate = 0;
    }

    /* Design asymmetric resonator biquad filter. */
    CarlFrontendDesignBiquad(pole, sample_rate_hz, &frontend->channel_data[c]);
    /* Normalize channel output to have unit peak gain. */
    double peak_frequency_hz = pole;
    const double peak_gain = CarlFrontendFindPeakGain(
        frontend->channel_data, c, params->input_sample_rate_hz,
        &peak_frequency_hz);
    frontend->channel_data[c].biquad_coeffs.b0 /= peak_gain;
    frontend->channel_data[c].biquad_coeffs.b1 /= peak_gain;
    frontend->channel_data[c].biquad_coeffs.b2 /= peak_gain;

    frontend->channel_data[c].envelope_smoother_coeff =
      (float)(1.0 - ComputeGammaFilterZPole(
            2, params->envelope_cutoff_hz, sample_rate_hz));

    /* Get pole frequency for the next channel. */
    pole = CarlFrontendNextAuditoryFrequency(pole, params->step_erbs);
  }

  CarlFrontendReset(frontend);
  return frontend;
}

void CarlFrontendFree(CarlFrontend* frontend) {
  if (frontend != NULL) {
    free(frontend->channel_data);
    free(frontend->channel_state);
    free(frontend);
  }
}

void CarlFrontendReset(CarlFrontend* frontend) {
  int c;
  for (c = 0; c < frontend->num_channels; ++c) {
    CarlFrontendChannelState* channel_state = &frontend->channel_state[c];
    BiquadFilterInitZero(&channel_state->biquad_state);
    channel_state->diff_state = 0.0f;
    channel_state->energy_envelope_stage1 = 0.0f;
    channel_state->energy_envelope = 0.0f;
    /* Reset to small positive value, not zero, since it is a denominator. */
    channel_state->pcen_denom = frontend->pcen_init_value;
  }
}

int CarlFrontendNumChannels(const CarlFrontend* frontend) {
  return frontend->num_channels;
}

int CarlFrontendBlockSize(const CarlFrontend* frontend) {
  return frontend->block_size;
}

/* The PCEN compression formula. */
static float PcenCompression(const CarlFrontend* frontend,
    float energy, float smoothed_energy) {
  return FastPow(energy * FastPow(frontend->pcen_gamma + smoothed_energy,
                                  -frontend->pcen_alpha)
      + frontend->pcen_delta, frontend->pcen_beta) - frontend->pcen_offset;
}

/* Smooth pcen_denom in-place across channels with a 3-tap filter:
 *
 *   new_pcen_denom[c] = pcen_denom[c] + coeff * (
 *       pcen_denom[c+1] - 2 * pcen_denom[c] + pcen_denom[c-1]).
 *
 * This is 1-D heat equation with a forward Euler finite difference scheme.
 * Boundaries are handled reflecting; no flow across boundaries.
 */
static void PcenDenomCrossChannelSmoothing(CarlFrontend* frontend) {
  CarlFrontendChannelState* channel_state = frontend->channel_state;
  const int num_channels = frontend->num_channels;
  const float coeff = frontend->pcen_cross_channel_smoother_coeff;

  float left_flux;
  float right_flux = channel_state[1].pcen_denom - channel_state[0].pcen_denom;
  channel_state[0].pcen_denom += coeff * right_flux;

  /* It is convenient to define flux[c] = (pcen_denom[c+1] - pcen_denom[c]) and
   * express the time step "in flux form" as
   *
   *   new_pcen_denom[c] = pcen_denom[c] + coeff * (flux[c] - flux[c-1]),
   *
   * or in-place as `pcen_denom[c] += coeff * (flux[c] - flux[c-1])`.
   */
  int c;
  for (c = 1; c < num_channels - 1; ++c) {
    left_flux = right_flux;
    right_flux = channel_state[c + 1].pcen_denom - channel_state[c].pcen_denom;
    channel_state[c].pcen_denom += coeff * (right_flux - left_flux);
  }

  channel_state[c].pcen_denom -= coeff * right_flux;
}

void CarlFrontendProcessSamples(CarlFrontend* frontend,
                                float* input,
                                float* output) {
  const int num_channels = frontend->num_channels;
  int c;
  int stride = 1;

  for (c = 0; c < num_channels; ++c) {
    const CarlFrontendChannelData channel_data = frontend->channel_data[c];
    CarlFrontendChannelState channel_state = frontend->channel_state[c];

    if (channel_data.should_decimate) {
      stride *= 2;  /* Decimate by factor 2. */
    }

    int i;
    for (i = 0; i < frontend->block_size; i += stride) {
      /* Apply asymmetric resonator biquad filter. */
      const float biquad_output = BiquadFilterProcessOneSample(
          &channel_data.biquad_coeffs, &channel_state.biquad_state, input[i]);

      /* Overwrite `input` with the output so that the next biquad is cascaded
       * with this one.
       */
      input[i] = biquad_output;

      /* Apply difference filter. This computes CARL's output. */
      const float carl_output = biquad_output - channel_state.diff_state;
      channel_state.diff_state = biquad_output;

      /* Half-wave rectification and square to get energy. */
      const float energy = (carl_output > 0.0f)
          ? carl_output * carl_output : 0.0f;

      /* Apply 2nd-order Gamma filter to get anti-aliased energy envelope. */
      channel_state.energy_envelope_stage1 +=
          channel_data.envelope_smoother_coeff * (
              energy - channel_state.energy_envelope_stage1);
      channel_state.energy_envelope +=
          channel_data.envelope_smoother_coeff * (
              channel_state.energy_envelope_stage1
              - channel_state.energy_envelope);
    }

    frontend->channel_state[c] = channel_state;
  }

  /* Second pass of lowpass filtering for PCEN denominator, done here outside
   * the loop on the downsampled envelope.
   */
  const float pcen_smoother_coeff = frontend->pcen_smoother_coeff;
  for (c = 0; c < num_channels; ++c) {
    frontend->channel_state[c].pcen_denom += pcen_smoother_coeff * (
        frontend->channel_state[c].energy_envelope
        - frontend->channel_state[c].pcen_denom);
  }

  /* Smooth pcen_denom across channels. */
  PcenDenomCrossChannelSmoothing(frontend);
  /* Compute PCEN-normalized energy. */
  for (c = 0; c < num_channels; ++c) {
    output[c] = PcenCompression(frontend,
                                frontend->channel_state[c].energy_envelope,
                                frontend->channel_state[c].pcen_denom);
  }
}
