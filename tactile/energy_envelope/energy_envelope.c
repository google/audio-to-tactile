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

#include "tactile/energy_envelope/energy_envelope.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp/butterworth.h"
#include "dsp/fast_fun.h"
#include "dsp/math_constants.h"

/* NOTE: These pages have a good description of acoustic phonetics and
 * describe spectrogram characteristics of different categories of phones:
 * http://ec-concord.ied.edu.hk/phonetics_and_phonology/wordpress/learning_website/chapter_2_vowels_new.htm
 * http://ec-concord.ied.edu.hk/phonetics_and_phonology/wordpress/learning_website/chapter_3_consonants_new.htm
 *
 *  - Nasals "n","m","ng" are in 200-450 Hz plus an F3 component around 2.5 kHz.
 *  - For "HK speaker 2", vowels have F1 frequency in about 250-900 Hz, and F2
 *    in 850-2400 Hz, and F3 frequency is roughly around 3000 Hz, for a band of
 *    about 250-3500 Hz for vowels. To reduce overlap on the low end with
 *    nasals, so restricting to 500-3500 Hz makes sense.
 *  - For alveolar sibilants like "s","z" most energy is 3.5-10 kHz.
 *  - Posalveloar sibilants like "sh" are lower, with most energy in 2-10 kHz.
 */

/* Parameters for baseband channel, sensitive to 80-500 Hz.
 * This channel should respond to nasals "n", "m", "ng".
 */
const EnergyEnvelopeParams kEnergyEnvelopeBasebandParams = {
  /*bpf_low_edge_hz=*/80.0f,
  /*bpf_high_edge_hz=*/500.0f,
  /*energy_smoother_cutoff_hz=*/500.0f,
  /*pcen_time_constant_s=*/0.3f,
  /*pcen_alpha=*/0.5f,
  /*pcen_beta=*/0.25f,
  /*pcen_gamma=*/1e-8f,
  /*pcen_delta=*/5e-3f,
  /*output_gain=*/1.0f,
};

/* Parameters for vowel channel, sensitive to 500-3500 Hz. */
const EnergyEnvelopeParams kEnergyEnvelopeVowelParams = {
  /*bpf_low_edge_hz=*/500.0f,
  /*bpf_high_edge_hz=*/3500.0f,
  /*energy_smoother_cutoff_hz=*/500.0f,
  /*pcen_time_constant_s=*/0.3f,
  /*pcen_alpha=*/0.5f,
  /*pcen_beta=*/0.25f,
  /*pcen_gamma=*/1e-8f,
  /*pcen_delta=*/1.5e-4f,
  /*output_gain=*/1.0f,
};

/* Parameters for "sh" fricative channel, sensitive to 2500-3500 Hz.
 * This channel should respond especially to "sh", "ch" and other postalveolar
 * consonants. From experiments, the band for this channel shouldn't go much
 * lower/higher than 2500-3500, otherwise it overlaps too much with vowels and
 * "s" fricatives. Plus, our BPFs are deliberately low order, so there is
 * nontrivial overlap regardless.
 */
const EnergyEnvelopeParams kEnergyEnvelopeShFricativeParams = {
  /*bpf_low_edge_hz=*/2500.0f,
  /*bpf_high_edge_hz=*/3500.0f,
  /*energy_smoother_cutoff_hz=*/500.0f,
  /*pcen_time_constant_s=*/0.3f,
  /*pcen_alpha=*/0.5f,
  /*pcen_beta=*/0.5f,
  /*pcen_gamma=*/1e-8f,
  /*pcen_delta=*/1.5e-4f,
  /*output_gain=*/1.0f,
};

/* Parameters for fricative channel, sensitive to 4000-6000 Hz.
 * This channel should respond especially to "s" and "z" alveolar sounds, and
 * other fricatives like "v" and "th" produced closer to the front of the mouth.
 */
const EnergyEnvelopeParams kEnergyEnvelopeFricativeParams = {
  /*bpf_low_edge_hz=*/4000.0f,
  /*bpf_high_edge_hz=*/6000.0f,
  /*energy_smoother_cutoff_hz=*/500.0f,
  /*pcen_time_constant_s=*/0.3f,
  /*pcen_alpha=*/0.5f,
  /*pcen_beta=*/0.5f,
  /*pcen_gamma=*/1e-8f,
  /*pcen_delta=*/1.5e-4f,
  /*output_gain=*/1.0f,
};

int EnergyEnvelopeInit(EnergyEnvelope* state,
                       const EnergyEnvelopeParams* params,
                       float input_sample_rate_hz,
                       int decimation_factor) {
  if (state == NULL || params == NULL) {
    fprintf(stderr, "EnergyEnvelopeInit: Null argument.\n");
    return 0;
  } else if (!(input_sample_rate_hz > 0.0f) ||
          !(params->pcen_time_constant_s > 0.0f) ||
          !(0.0f <= params->pcen_alpha && params->pcen_alpha <= 1.0f) ||
          !(params->pcen_beta > 0.0f) ||
          !(params->pcen_gamma > 0.0f) ||
          !(params->pcen_delta > 0.0f) ||
          !(params->output_gain >= 0.0f) ||
          !(decimation_factor > 0)) {
    fprintf(stderr, "EnergyEnvelopeInit: Invalid EnergyEnvelopeParams.\n");
    return 0;
  } else if (!DesignButterworthOrder2Bandpass(
      params->bpf_low_edge_hz, params->bpf_high_edge_hz,
      input_sample_rate_hz, state->bpf_biquad_coeffs)) {
    fprintf(stderr, "EnergyEnvelopeInit: Failed to design bandpass filter.\n");
    return 0;
  } else if (!DesignButterworthOrder2Lowpass(
      params->energy_smoother_cutoff_hz,
      input_sample_rate_hz, &state->energy_biquad_coeffs)) {
    fprintf(stderr, "EnergyEnvelopeInit: Failed to design energy smoother.\n");
    return 0;
  }

  state->decimation_factor = decimation_factor;
  state->pcen_smoother_coeff = (float)(1.0 - exp(-decimation_factor
      / (params->pcen_time_constant_s * input_sample_rate_hz)));
  state->pcen_alpha = params->pcen_alpha;
  state->pcen_beta = params->pcen_beta;
  state->pcen_gamma = params->pcen_gamma;
  state->pcen_delta = params->pcen_delta;
  state->pcen_offset = FastPow(params->pcen_delta, params->pcen_beta);
  state->output_gain = params->output_gain;

  EnergyEnvelopeReset(state);
  return 1;
}

void EnergyEnvelopeReset(EnergyEnvelope* state) {
  BiquadFilterInitZero(&state->bpf_biquad_state[0]);
  BiquadFilterInitZero(&state->bpf_biquad_state[1]);
  BiquadFilterInitZero(&state->energy_biquad_state);
  state->pcen_denom = 10 * state->pcen_gamma;
}

void EnergyEnvelopeProcessSamples(EnergyEnvelope* state,
                                  const float* input,
                                  int num_samples,
                                  float* output,
                                  int output_stride) {
  const int decimation_factor = state->decimation_factor;
  const float pcen_smoother_coeff = state->pcen_smoother_coeff;
  float energy = 0.0f;
  float* dest = output;
  int i;
  for (i = decimation_factor - 1; i < num_samples; i += decimation_factor) {
    int j;
    for (j = 0; j < decimation_factor; ++j) {
      /* Bandpass filter. */
      float sample = BiquadFilterProcessOneSample(
          &state->bpf_biquad_coeffs[0], &state->bpf_biquad_state[0], input[j]);
      sample = BiquadFilterProcessOneSample(
          &state->bpf_biquad_coeffs[1], &state->bpf_biquad_state[1], sample);

      /* Half-wave rectification and squaring. */
      energy = (sample > 0.0f) ? sample * sample : 0.0f;

      /* Lowpass filter to smooth the energy envelope. */
      energy = BiquadFilterProcessOneSample(
          &state->energy_biquad_coeffs, &state->energy_biquad_state, energy);
    }

    state->pcen_denom += pcen_smoother_coeff * (energy - state->pcen_denom);

    /* Modified PCEN formula,
     *   agc_out = energy / (gamma + smoothed_energy)^alpha,
     *   out = (agc_out^2 + delta)^beta - delta^beta.
     */
    const float agc_out = energy *
        FastPow(state->pcen_gamma + state->pcen_denom, -state->pcen_alpha);
    *dest = (FastPow(agc_out * agc_out + state->pcen_delta,
          state->pcen_beta) - state->pcen_offset) * state->output_gain;

    dest += output_stride;
    input += decimation_factor;
  }
}

