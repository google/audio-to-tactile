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
            /*denoise_thresh_factor=*/25.0f,
            /*output_gain=*/1.0f,
        },
        /* Vowel channel, sensitive to 500-3500 Hz. */
        {
            /*bpf_low_edge_hz=*/500.0f,
            /*bpf_high_edge_hz=*/3500.0f,
            /*denoise_thresh_factor=*/4.0f,
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
            /*denoise_thresh_factor=*/4.0f,
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
            /*denoise_thresh_factor=*/4.0f,
            /*output_gain=*/1.0f,
        },
    },
    /*energy_cutoff_hz=*/500.0f,
    /*energy_tau_s=*/0.01f,
    /*cross_channel_tau_s=*/0.1f,
    /*noise_tau_s=*/1.0f,
    /*agc_strength=*/0.7f,
    /*gain_tau_attack_s=*/0.002f,
    /*gain_tau_release_s=*/0.15f,
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
             !(params->cross_channel_tau_s >= 0.0f) ||
             !(params->noise_tau_s >= 0.0f) ||
             !(0.0f <= params->agc_strength && params->agc_strength <= 1.0f) ||
             !(params->gain_tau_attack_s >= 0.0f) ||
             !(params->gain_tau_release_s >= 0.0f) ||
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

    if (!(params_c->denoise_thresh_factor >= 0.0f) ||
        !(params_c->output_gain >= 0.0f)) {
      fprintf(stderr,
              "EnveloperInit: Invalid EnveloperParams for channel %d.\n", c);
      return 0;
    } else if (!DesignButterworthOrder2Bandpass(
            params_c->bpf_low_edge_hz, params_c->bpf_high_edge_hz,
            input_sample_rate_hz, state_c->bpf_biquad_coeffs)) {
      fprintf(stderr, "EnveloperInit: Failed to design bandpass filter %d.\n",
              c);
      return 0;
    }
    state_c->denoise_thresh_factor = params_c->denoise_thresh_factor;
    state_c->output_gain = params_c->output_gain;
  }

  state->input_sample_rate_hz = input_sample_rate_hz;
  state->decimation_factor = decimation_factor;

  state->energy_smoother_coeff =
      EnveloperSmootherCoeff(state, params->energy_tau_s);
  state->cross_channel_diffusion_coeff = EnveloperCrossChannelDiffusionCoeff(
      state, params->cross_channel_tau_s);
  state->noise_smoother_coeff =
      EnveloperSmootherCoeff(state, params->noise_tau_s);
  state->gain_smoother_coeffs[0] =
      EnveloperSmootherCoeff(state, params->gain_tau_attack_s);
  state->gain_smoother_coeffs[1] =
      EnveloperSmootherCoeff(state, params->gain_tau_release_s);
  state->agc_exponent = -params->agc_strength;

  state->compressor_exponent = params->compressor_exponent;
  state->compressor_delta = params->compressor_delta;

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
    state_c->log2_noise = -19.932f; /* = log2(1e-6). */
    state_c->smoothed_gain = 0.0f;
  }
}

void EnveloperUpdatePrecomputedParams(Enveloper* state) {
  /* Precompute `delta^exponent`. */
  state->compressor_offset =
      FastPow(state->compressor_delta, state->compressor_exponent);

  int c;
  for (c = 0; c < kEnveloperNumChannels; ++c) {
    EnveloperChannel* state_c = &state->channels[c];
    /* Precompute `(1/output_gain + delta^exponent)^(1/exponent) - delta`. */
    state_c->agc_max_output =
        FastPow(1.0f / state_c->output_gain + state->compressor_offset,
                1.0f / state->compressor_exponent) -
        state->compressor_delta;
  }
}

/* Computes the AGC gain with noise gating, in which `x` is the smoothed energy:
 *
 *   gain = 0                  if x <= thresh,
 *   gain ~= x^agc_exponent    if x >> thresh,
 *
 * with a smooth transition from zero to x^agc_exponent for x near thresh.
 */
static float ComputeGain(float x, float agc_exponent, float thresh) {
  const float y = x - thresh;
  if (y <= 1e-8f) {
    return 0.0f; /* Gain of zero if x <= thresh. */
  }
  /* We create a smooth transition with this function (for some constant c > 0):
   *
   *   y^2 / (y^2 + c),
   *
   * which is zero at y = 0 and asymptotically one as y -> infinity. We set c
   * to 3 * thresh^2 so that the inflection is at x = 2 thresh or y = thresh. So
   * the gain is
   *
   *   gain = x^agc_exponent * (y^2 / (y^2 + c))
   *        = x^agc_exponent / (1 + 3 * (thresh / y)^2).
   */
  const float ratio = thresh / y;
  return FastPow(1e-12f + x, agc_exponent) / (1.0f + 3.0f * ratio * ratio);
}

/* Diffuse `smoothed_energy` in-place across channels with a 3-tap filter:
 *
 *   new_smoothed_energy[c] = smoothed_energy[c] + coeff * (
 *       smoothed_energy[c+1] - 2 * smoothed_energy[c] + smoothed_energy[c-1])
 *
 * with no flow across the boundaries.
 */
static void CrossChannelDiffusion(Enveloper* state) {
  const float coeff = state->cross_channel_diffusion_coeff;
  EnveloperChannel* channels = state->channels;

  /* It is convenient to define flux[c] = (smoothed_energy[c+1] -
   * smoothed_energy[c]) and express the time step "in flux form" as
   *
   *   new_smoothed_energy[c] =
   *       smoothed_energy[c] + coeff * (flux[c] - flux[c-1]),
   *
   * or in-place as `smoothed_energy[c] += coeff * (flux[c] - flux[c-1])`.
   */
  float left;
  float right = /* right = coeff * flux[0]. */
      coeff * (channels[1].smoothed_energy - channels[0].smoothed_energy);
  channels[0].smoothed_energy += right; /* Left endpoint: flux[-1] is zero. */
  int c;
  for (c = 1; c < kEnveloperNumChannels - 1; ++c) {
    left = right; /* left = coeff * flux[c-1]. */
    right = /* right = coeff * flux[c]. */
        coeff * (channels[c + 1].smoothed_energy - channels[c].smoothed_energy);
    channels[c].smoothed_energy += right - left;
  }
  channels[c].smoothed_energy -= right; /* Right endpoint. */
}

void EnveloperProcessSamples(Enveloper* state,
                             const float* input,
                             int num_samples,
                             float* output) {
  const int decimation_factor = state->decimation_factor;
  float energy = 0.0f;
  int i;

  for (i = decimation_factor - 1; i < num_samples; i += decimation_factor) {
    int c;
    for (c = 0; c < kEnveloperNumChannels; ++c) {
      EnveloperChannel* state_c = &state->channels[c];

      int j;
      for (j = 0; j < decimation_factor; ++j) {
        /* Bandpass energy computation. ***************************************/
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

      /* Ensure energy > 0 to avoid problems with log-space filtering below. */
      if (energy < 1e-8f) { energy = 1e-8f; }

      /* Automatic gain control with noise gating. ****************************/
      float smoothed_energy = state_c->smoothed_energy;
      float log2_noise = state_c->log2_noise;
      float smoothed_gain = state_c->smoothed_gain;

      /* Update noise envelope estimate. */
      smoothed_energy +=
          state->energy_smoother_coeff * (energy - smoothed_energy);
      log2_noise +=
          state->noise_smoother_coeff * (FastLog2(smoothed_energy) - log2_noise);

      /* Set noise gate threshold as a multiple of the estimate noise. */
      const float noise_gate_threshold =
          state_c->denoise_thresh_factor * FastExp2(log2_noise);
      /* Compute (unsmoothed) AGC gain from smoothed energy and threshold. */
      const float gain = ComputeGain(
          smoothed_energy, state->agc_exponent, noise_gate_threshold);
      /* Update smoothed AGC gain with asymmetric smoother. */
      smoothed_gain += state->gain_smoother_coeffs[gain < smoothed_gain] *
                       (gain - smoothed_gain);
      float normalized_energy = smoothed_gain * energy;
      /* If needed, reduce smoothed_gain so that final output <= 1:
       *
       *   normalized_energy
       *      <= (1/output_gain + delta^exponent)^(1/exponent) - delta
       *      =  agc_max_output.
       */
      if (normalized_energy > state_c->agc_max_output) {
        smoothed_gain = state_c->agc_max_output / energy;
        normalized_energy = state_c->agc_max_output;
      }

      state_c->smoothed_energy = smoothed_energy;
      state_c->log2_noise = log2_noise;
      state_c->smoothed_gain = smoothed_gain;

      /* Power law compression. ***********************************************/
      /* Compress with a power law and multiply by `output_gain` as
       *
       * output_gain * ((normalized_energy + delta)^exponent - delta^exponent).
       */
      output[c] = state_c->output_gain *
                (FastPow(normalized_energy + state->compressor_delta,
                         state->compressor_exponent) -
                 state->compressor_offset);
    }

    /* Diffuse `smoothed_energy` across channels. */
    CrossChannelDiffusion(state);

    output += kEnveloperNumChannels;
    input += decimation_factor;
  }
}
