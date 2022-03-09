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

#include "tactile/enveloper.h"

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

const EnveloperParams kDefaultEnveloperParams = {
    {
        /* Baseband channel, sensitive to 80-500 Hz.
         * This channel should respond to nasals "n", "m", "ng".
         */
        {
            /*bpf_low_edge_hz=*/80.0f,
            /*bpf_high_edge_hz=*/500.0f,
            /*output_gain=*/1.0f,
        },
        /* Vowel channel, sensitive to 500-3500 Hz. */
        {
            /*bpf_low_edge_hz=*/500.0f,
            /*bpf_high_edge_hz=*/3500.0f,
            /*output_gain=*/1.0f,
        },
        /* "sh" fricative channel, sensitive to 2500-3500 Hz.
         * This channel should respond especially to "sh", "ch" and other
         * postalveolar consonants. From experiments, the band for this channel
         * shouldn't go much lower/higher than 2500-3500, otherwise it overlaps
         * too much with vowels and "s" fricatives. Plus, our BPFs are
         * deliberately low order, so there is nontrivial overlap regardless.
         */
        {
            /* Sh fricative channel. */
            /*bpf_low_edge_hz=*/2500.0f,
            /*bpf_high_edge_hz=*/3500.0f,
            /*output_gain=*/1.0f,
        },
        /* Fricative channel, sensitive to 4000-6000 Hz.
         * This channel should respond especially to "s" and "z" alveolar
         * sounds, and other fricatives like "v" and "th" produced closer to the
         * front of the mouth.
         */
        {
            /* Fricative channel. */
            /*bpf_low_edge_hz=*/4000.0f,
            /*bpf_high_edge_hz=*/6000.0f,
            /*output_gain=*/1.0f,
        },
    },
    /*energy_cutoff_hz=*/500.0f,
    /*energy_tau_s=*/0.01f,
    /*noise_db_s=*/2.0f,
    /*agc_strength=*/0.7f,
    /*compressor_exponent=*/0.25f,
    /*compressor_delta=*/0.01f,
};

int EnveloperInit(Enveloper* state,
                  const EnveloperParams* params,
                  float input_sample_rate_hz,
                  int decimation_factor) {
  if (state == NULL || params == NULL) {
    fprintf(stderr, "EnveloperInit: Null argument.\n");
    return 0;
  } else if (!(input_sample_rate_hz > 0.0f) ||
             !(params->energy_tau_s >= 0.0f) ||
             !(0.0f <= params->agc_strength && params->agc_strength <= 1.0f) ||
             !(params->compressor_exponent > 0.0f) ||
             !(params->compressor_delta > 0.0f) ||
             !(decimation_factor > 0)) {
    fprintf(stderr, "EnveloperInit: Invalid EnveloperParams.\n");
    return 0;
  } else if (!DesignButterworthOrder2Lowpass(params->energy_cutoff_hz,
                                             input_sample_rate_hz,
                                             &state->energy_biquad_coeffs)) {
    fprintf(stderr, "EnveloperInit: Failed to design energy smoother.\n");
    return 0;
  }

  int c;
  for (c = 0; c < kEnveloperNumChannels; ++c) {
    const EnveloperChannelParams* params_c = &params->channel_params[c];
    EnveloperChannel* state_c = &state->channels[c];
    state_c->output_gain = params_c->output_gain;

    if (!DesignButterworthOrder2Bandpass(
            params_c->bpf_low_edge_hz, params_c->bpf_high_edge_hz,
            input_sample_rate_hz, state_c->bpf_biquad_coeffs)) {
      fprintf(stderr, "EnveloperInit: Failed to design bandpass filter %d.\n",
              c);
      return 0;
    }
  }

  state->input_sample_rate_hz = input_sample_rate_hz;
  state->decimation_factor = decimation_factor;
  state->agc_strength = params->agc_strength;
  state->compressor_exponent = params->compressor_exponent;
  state->compressor_delta = params->compressor_delta;

  state->energy_smoother_coeff =
      EnveloperSmootherCoeff(state, params->energy_tau_s);
  state->noise_coeffs[1] =
      EnveloperGrowthCoeff(state, params->noise_db_s);

  EnveloperUpdatePrecomputedParams(state);
  EnveloperReset(state);
  return 1;
}

void EnveloperReset(Enveloper* state) {
  int c;
  for (c = 0; c < kEnveloperNumChannels; ++c) {
    EnveloperChannel* state_c = &state->channels[c];
    BiquadFilterInitZero(&state_c->bpf_biquad_state[0]);
    BiquadFilterInitZero(&state_c->bpf_biquad_state[1]);
    BiquadFilterInitZero(&state_c->energy_biquad_state);
    state_c->smoothed_energy = 1e-6f;
    state_c->noise = 1e-7f;
  }
}

void EnveloperUpdatePrecomputedParams(Enveloper* state) {
  /* Precompute noise estimation decay coefficient. */
  state->noise_coeffs[0] = 1.0f / state->noise_coeffs[1];
  /* Precompute `delta^exponent`. */
  state->compressor_offset =
      FastPow(state->compressor_delta, state->compressor_exponent);
}

void EnveloperProcessSamples(Enveloper* state,
                             const float* input,
                             int num_samples,
                             float* output) {
  const float energy_smoother_coeff = state->energy_smoother_coeff;
  const float agc_strength = state->agc_strength;
  const float compressor_exponent = state->compressor_exponent;
  const float compressor_delta = state->compressor_delta;
  const float compressor_offset = state->compressor_offset;
  const int decimation_factor = state->decimation_factor;
  float energy = 0.0f;
  int i;

  for (i = decimation_factor - 1; i < num_samples; i += decimation_factor) {
    int c;
    for (c = 0; c < kEnveloperNumChannels; ++c) {
      EnveloperChannel* state_c = &state->channels[c];

      int j;
      for (j = 0; j < decimation_factor; ++j) {
        /* Apply bandpass filter. */
        float sample = BiquadFilterProcessOneSample(
            &state_c->bpf_biquad_coeffs[0],
            &state_c->bpf_biquad_state[0], input[j]);
        sample = BiquadFilterProcessOneSample(
            &state_c->bpf_biquad_coeffs[1],
            &state_c->bpf_biquad_state[1], sample);

        /* Half-wave rectification and squaring. */
        const float rectified = (sample > 0.0f) ? sample * sample : 0.0f;

        /* Lowpass filter the energy envelope. */
        energy = BiquadFilterProcessOneSample(
            &state->energy_biquad_coeffs,
            &state_c->energy_biquad_state, rectified);
      }

      if (energy < 0.0f) { energy = 0.0f; }

      float smoothed_energy = state_c->smoothed_energy;
      float noise = state_c->noise;

      /* Update PCEN denominator. */
      smoothed_energy += energy_smoother_coeff * (energy - smoothed_energy);

      /* Update noise level estimate. */
      noise *= state->noise_coeffs[smoothed_energy > noise];
      if (noise < 1e-9f) { noise = 1e-9f; }

      state_c->smoothed_energy = smoothed_energy;
      state_c->noise = noise;

      /* Apply auto gain control. */
      const float ratio = smoothed_energy /
          (smoothed_energy * smoothed_energy + noise);
      const float agc_gain = FastPow(1e-12f + ratio, agc_strength);
      output[c] = state_c->output_gain *
                  (FastPow(agc_gain * energy + compressor_delta,
                           compressor_exponent) - compressor_offset);
    }

    output += kEnveloperNumChannels;
    input += decimation_factor;
  }
}
