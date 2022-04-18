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
    /*denoising_strength=*/1.0f,
    /*denoising_transition_db=*/10.0f,
    /*agc_strength=*/0.7f,
    /*compressor_exponent=*/0.25f,
    /*compressor_delta=*/0.01f,
};

static float ComputeFilteredPeak(
    const BiquadFilterCoeffs* filter, const EnveloperChannelParams* params_c,
    float input_sample_rate_hz) {
  /* Coefficients of the first few Fourier series terms for a half-waved
   * rectified and squared signal:
   *
   *   max(0, cos(x))^2 = sum_k kSeriesCoeffs[k] cos(k x).
   */
  static const float kSeriesCoeffs[4] =
      {0.25f, (float)(4 / (3 * M_PI)), 0.25f, (float)(4 / (15 * M_PI))};
  const float freq_hz =
      (float)sqrt(params_c->bpf_low_edge_hz * params_c->bpf_high_edge_hz);

  /* Apply the filter to the Fourier series and evaluate at the signal peak. */
  float peak = kSeriesCoeffs[0];
  int k;
  for (k = 1; k <= 3; ++k) {
    peak += (float)ComplexDoubleAbs(BiquadFilterFrequencyResponse(
        filter, k * freq_hz / input_sample_rate_hz)) * kSeriesCoeffs[k];
  }
  return peak;
}

int EnveloperInit(Enveloper* state,
                  const EnveloperParams* params,
                  float input_sample_rate_hz,
                  int decimation_factor) {
  if (state == NULL || params == NULL) {
    fprintf(stderr, "EnveloperInit: Null argument.\n");
    return 0;
  } else if (!(input_sample_rate_hz > 0.0f) ||
             !(params->energy_tau_s >= 0.0f) ||
             !(params->denoising_strength > 0.0f) ||
             !(params->denoising_transition_db > 0.0f) ||
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
    state_c->peak = ComputeFilteredPeak(
        &state->energy_biquad_coeffs, params_c, input_sample_rate_hz);
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
  state->agc_exponent = -params->agc_strength;
  state->compressor_exponent = params->compressor_exponent;
  state->compressor_delta = params->compressor_delta;

  state->energy_smoother_coeff =
      EnveloperSmootherCoeff(state, params->energy_tau_s);
  state->noise_coeffs[1] =
      EnveloperGrowthCoeff(state, params->noise_db_s);
  state->gate_thresh_factor = params->denoising_strength;
  state->gate_transition_factor =
      DecibelsToPowerRatio(params->denoising_transition_db);

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
    state_c->smoothed_energy = 1e-5f * state_c->equalization;
    state_c->noise = state_c->smoothed_energy;
  }
}

void EnveloperUpdatePrecomputedParams(Enveloper* state) {
  /* Precompute noise estimation decay coefficient. */
  state->noise_coeffs[0] = 1.0f / state->noise_coeffs[1];
  /* Precompute `delta^exponent`. */
  state->compressor_offset =
      FastPow(state->compressor_delta, state->compressor_exponent);

  /* Enveloper's process for envelope extraction and compression inherently
   * amplifies lower frequencies somewhat more than higher frequencies.
   *
   * After half-wave rectification, roughly half of the signal is zero, and the
   * other half are semiperiodic pulses. The signal is then squared and lowpass
   * filtered with cutoff `energy_cutoff_hz`. For the baseband channel, the
   * pulse rate is well below the cutoff and the pulse peak height is mostly
   * maintained. But for higher channels, the pulse rate is above the cutoff and
   * pulse peak height is substantially reduced. PCEN's AGC amplifies these
   * differences because (by design) the AGC emphasizes signal onsets.
   *
   * To compensate for this effect, we compute and use an `equalization`
   * factor for each channel as follows:
   *
   * 1. Suppose the input is a unit-amplitude sinusoid with frequency at the
   *    geometric midpoint of the channel band. ComputeFilteredPeak() computes
   *    the peak signal height after half-wave rectification, squaring, and
   *    lowpass filtering. This value is stored in `state_c->peak`.
   *
   * 2. In the loop below, `pcen_peak` computes the signal peak value after
   *    PCEN. The `energy_tau_s` parameter is assumed to be large enough that
   *    the denominator has smoothed to the signal average, which is 1/4. We
   *    compute `equalization` such that scaling the denominator by this factor
   *    would make PCEN's output approximately equal to kTargetOutput.
   *
   * 3. In EnveloperProcessSamples(), we scale the `smoothed_energy` signal by
   *    `equalization`.
   */
  int c;
  for (c = 0; c < kEnveloperNumChannels; ++c) {
    EnveloperChannel* state_c = &state->channels[c];

    const float kTargetOutput = 0.8f;
    const float pcen_peak =
        FastPow(FastExp2(-2 * state->agc_exponent) * state_c->peak +
            state->compressor_delta, state->compressor_exponent)
        - state->compressor_offset;
    state_c->equalization = FastPow(kTargetOutput / pcen_peak,
        1.0f / (state->agc_exponent * state->compressor_exponent));
  }
}

/* Smooth gate function `x^2 / (x^2 + halfway_point^2)`. The function behaves
 * like `x^2 / halfway_point^2` for x near zero, is equal to 1/2 at
 * x = halfway_point, and is asymptotically 1 as x -> infinity.
 */
static float SoftGate(float x, float halfway_point) {
  const float x_sqr = x * x;
  return x_sqr / (x_sqr + halfway_point * halfway_point);
}

void EnveloperProcessSamples(Enveloper* state,
                             const float* input,
                             int num_samples,
                             float* output) {
  const float energy_smoother_coeff = state->energy_smoother_coeff;
  const float gate_thresh_factor = state->gate_thresh_factor;
  const float gate_transition_factor = state->gate_transition_factor;
  const float agc_exponent = state->agc_exponent;
  const float compressor_exponent = state->compressor_exponent;
  const float compressor_delta = state->compressor_delta;
  const float compressor_offset = state->compressor_offset;
  const int decimation_factor = state->decimation_factor;
  int i;

  for (i = decimation_factor - 1; i < num_samples; i += decimation_factor) {
    float energy = 0.0f;
    float prev_smoothed_energy = 0.0f;
    int c;

    for (c = kEnveloperNumChannels - 1; c >= 0; --c) {
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
      smoothed_energy += energy_smoother_coeff * (
          state_c->equalization * energy - smoothed_energy);

      if (prev_smoothed_energy > smoothed_energy) {
        smoothed_energy = prev_smoothed_energy;
      }
      prev_smoothed_energy = smoothed_energy;

      /* Update noise level estimate. */
      noise *= state->noise_coeffs[smoothed_energy > noise];
      if (noise < 1e-9f) { noise = 1e-9f; }

      state_c->smoothed_energy = smoothed_energy;
      state_c->noise = noise;

      const float thresh = gate_thresh_factor * noise;
      const float diff = smoothed_energy - thresh;
      float agc_output;
      if (diff <= 1e-9f) {
        agc_output = 0.0f;  /* Gain of zero if smoothed_energy <= thresh. */
      } else {
        /* Apply soft noise gate and AGC gain. */
        agc_output = SoftGate(diff, gate_transition_factor * thresh) *
            FastPow(smoothed_energy, agc_exponent) * energy;
      }

      /* Apply power law compression and output gain. */
      output[c] = state_c->output_gain *
                  (FastPow(agc_output + compressor_delta, compressor_exponent)
                   - compressor_offset);
    }

    output += kEnveloperNumChannels;
    input += decimation_factor;
  }
}
