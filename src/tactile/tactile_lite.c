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

#include "tactile/tactile_lite.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp/butterworth.h"
#include "dsp/fast_fun.h"
#include "dsp/iir_design.h"

const SingleBandEnvelopeParams kDefaultSingleBandEnvelopeParams = {
    /*feedback_hpf_cutoff_hz=*/180.0f,
    /*feedback_hpf_stopband_ripple_db=*/40.0f,
    /* Broad vowel channel, sensitive to 300-3500 Hz. */
    /*bpf_low_edge_hz=*/300.0f,
    /*bpf_high_edge_hz=*/3500.0f,
    /*denoising_strength=*/4.0f,
    /*energy_cutoff_hz=*/250.0f,
    /*energy_tau_s=*/0.01f,
    /*noise_db_s=*/2.0f,
    /*denoising_transition_db=*/5.0f,
    /*agc_strength=*/0.7f,
    /*gain_tau_attack_s=*/0.005f,
    /*gain_tau_release_s=*/0.15f,
    /*compressor_exponent=*/0.25f,
};

const SparsePeakPickerParams kDefaultSparsePeakPickerParams = {
  /*smoothed_tau_attack_s=*/0.005f,
  /*smoothed_tau_release_s=*/0.03f,
  /*thresh_tau_attack_s=*/0.03f,
  /*thresh_tau_release_s=*/0.1f,
  /*min_peak_spacing_s=*/0.05f,
};

static const float kCompressorStabilization = 0.125f;

int SingleBandEnvelopeInit(SingleBandEnvelope* state,
                    const SingleBandEnvelopeParams* params,
                    float input_sample_rate_hz,
                    int decimation_factor) {
  if (state == NULL || params == NULL) {
    fprintf(stderr, "SingleBandEnvelopeInit: Null argument.\n");
    return 0;
  } else if (!(input_sample_rate_hz > 0.0f) ||
             !(params->energy_tau_s >= 0.0f) ||
             !(params->denoising_transition_db > 0.0f) ||
             !(0.0f <= params->agc_strength && params->agc_strength <= 1.0f) ||
             !(params->compressor_exponent > 0.0f) ||
             !(decimation_factor > 0)) {
    fprintf(stderr, "SingleBandEnvelopeInit: Invalid params.\n");
    return 0;
  } else if (!DesignButterworthOrder2Lowpass(params->energy_cutoff_hz,
                                             input_sample_rate_hz,
                                             &state->energy_biquad_coeffs)) {
    fprintf(stderr,
            "SingleBandEnvelopeInit: Failed to design energy smoother.\n");
    return 0;
  } else if (!DesignButterworthOrder2Bandpass(
          params->bpf_low_edge_hz, params->bpf_high_edge_hz,
          input_sample_rate_hz, state->bpf_biquad_coeffs)) {
    fprintf(stderr,
            "SingleBandEnvelopeInit: Failed to design bandpass filter.\n");
    return 0;
  }

  if (!(params->feedback_hpf_cutoff_hz > 0.0f)) {
    /* Bypass the feedback highpass filter. */
    state->feedback_hpf_biquad_coeffs[0] = kBiquadFilterIdentityCoeffs;
    state->feedback_hpf_biquad_coeffs[1] = kBiquadFilterIdentityCoeffs;
  } else if (DesignChebyshev2Highpass(
                 4, params->feedback_hpf_stopband_ripple_db,
                 params->feedback_hpf_cutoff_hz, input_sample_rate_hz,
                 state->feedback_hpf_biquad_coeffs, 2) != 2) {
    fprintf(stderr,
            "SingleBandEnvelopeInit: "
            "Failed to design feedback highpass filter.\n");
    return 0;
  }

  state->input_sample_rate_hz = input_sample_rate_hz;
  state->decimation_factor = decimation_factor;
  state->gate_thresh_factor = params->denoising_strength;
  state->agc_exponent = -params->agc_strength;
  state->compressor_exponent = params->compressor_exponent;

  state->energy_smoother_coeff =
      SingleBandEnvelopeSmootherCoeff(state, params->energy_tau_s);
  state->noise_coeffs[1] =
      SingleBandEnvelopeGrowthCoeff(state, params->noise_db_s);
  state->gate_transition_factor =
      DecibelsToPowerRatio(params->denoising_transition_db);
  state->gain_smoother_coeffs[0] =
      SingleBandEnvelopeSmootherCoeff(state, params->gain_tau_attack_s);
  state->gain_smoother_coeffs[1] =
      SingleBandEnvelopeSmootherCoeff(state, params->gain_tau_release_s);

  /* Warm up duration is 500 ms. */
  state->num_warm_up_samples =
      (int)(0.5f * input_sample_rate_hz / decimation_factor + 0.5f);

  SingleBandEnvelopeUpdatePrecomputedParams(state);
  SingleBandEnvelopeReset(state);
  return 1;
}

void SingleBandEnvelopeReset(SingleBandEnvelope* state) {
  BiquadFilterInitZero(&state->feedback_hpf_biquad_state[0]);
  BiquadFilterInitZero(&state->feedback_hpf_biquad_state[1]);
  BiquadFilterInitZero(&state->bpf_biquad_state[0]);
  BiquadFilterInitZero(&state->bpf_biquad_state[1]);
  BiquadFilterInitZero(&state->energy_biquad_state);
  state->smoothed_energy = 0.0f;
  state->noise = 0.0f;
  state->smoothed_gain = 0.0f;
  state->warm_up_counter = state->num_warm_up_samples;
}

void SingleBandEnvelopeUpdatePrecomputedParams(SingleBandEnvelope* state) {
  /* Precompute noise estimation decay coefficient. */
  state->noise_coeffs[0] = 1.0f / state->noise_coeffs[1];
  /* Precompute compressor delta = kCompressorStabilization^(1/exponent). */
  state->compressor_delta = (float) pow(kCompressorStabilization,
                                        1.0f / state->compressor_exponent);
}

/* Smooth gate function `x^2 / (x^2 + halfway_point^2)`. The function behaves
 * like `x^2 / halfway_point^2` for x near zero, is equal to 1/2 at
 * x = halfway_point, and is asymptotically 1 as x -> infinity.
 */
static float SoftGate(float x, float halfway_point) {
  const float x_sqr = x * x;
  return x_sqr / (x_sqr + halfway_point * halfway_point);
}

void SingleBandEnvelopeProcessSamples(SingleBandEnvelope* state,
                                      const float* input,
                                      int num_samples,
                                      float* output) {
  const float energy_smoother_coeff = state->energy_smoother_coeff;
  const float gate_transition_factor = state->gate_transition_factor;
  const float agc_exponent = state->agc_exponent;
  const float compressor_exponent = state->compressor_exponent;
  const float compressor_delta = state->compressor_delta;
  const int decimation_factor = state->decimation_factor;
  float smoothed_energy = state->smoothed_energy;
  float noise = state->noise;
  float smoothed_gain = state->smoothed_gain;
  int warm_up_counter = state->warm_up_counter;
  int i;

  for (i = decimation_factor - 1; i < num_samples; i += decimation_factor) {
    float energy = 0.0f;

    int j;
    for (j = 0; j < decimation_factor; ++j) {
      /* Apply feedback highpass filter. */
      float sample = BiquadFilterProcessOneSample(
          &state->feedback_hpf_biquad_coeffs[0],
          &state->feedback_hpf_biquad_state[0], input[j]);
      sample = BiquadFilterProcessOneSample(
          &state->feedback_hpf_biquad_coeffs[1],
          &state->feedback_hpf_biquad_state[1], sample);

      /* Apply bandpass filter. */
      sample = BiquadFilterProcessOneSample(
          &state->bpf_biquad_coeffs[0],
          &state->bpf_biquad_state[0], sample);
      sample = BiquadFilterProcessOneSample(
          &state->bpf_biquad_coeffs[1],
          &state->bpf_biquad_state[1], sample);

      /* Half-wave rectification and squaring. */
      const float rectified = (sample > 0.0f) ? sample * sample : 0.0f;

      /* Lowpass filter the energy envelope. */
      energy = BiquadFilterProcessOneSample(
          &state->energy_biquad_coeffs,
          &state->energy_biquad_state, rectified);
    }

    if (energy < 0.0f) { energy = 0.0f; }

    /* Update PCEN denominator. */
    smoothed_energy += energy_smoother_coeff * (energy - smoothed_energy);

    if (warm_up_counter) {  /* While warming up. */
      /* When processing first starts up, we don't yet have a good estimate of
        * the noise. During this "warm up" period, we compute `noise` to be the
        * average of the `energy` samples seen so far, and multiplied by 2 to
        * err on the side that the actual noise level might be somewhat higher.
        */
      noise += 2.0f * energy;  /* Sum up `energy`. */

      /* Divide to get the average. */
      const float average =
          noise / (state->num_warm_up_samples - warm_up_counter + 1);
      /* Store the average on the last warm up sample. Otherwise, store the
        * unnormalized energy sum.
        */
      state->noise = (warm_up_counter == 1) ? average : noise;
      noise = average;  /* Work with the average in the processing below. */
    } else {  /* After warm up is done. */
      /* Update noise level estimate. */
      noise *= state->noise_coeffs[smoothed_energy > noise];
      state->noise = noise;
    }

    if (noise < 1e-9f) { noise = 1e-9f; }

    const float thresh = state->gate_thresh_factor * noise;
    const float diff = smoothed_energy - thresh;
    float gain;
    if (diff <= 1e-9f) {
      gain = 0.0f;  /* Gain of zero if smoothed_energy <= thresh. */
    } else {
      /* Apply soft noise gate and AGC gain. */
      gain = SoftGate(diff, gate_transition_factor * thresh) *
          FastPow(smoothed_energy, agc_exponent);
    }

    /* Update smoothed AGC gain with asymmetric smoother. */
    smoothed_gain += state->gain_smoother_coeffs[gain < smoothed_gain] *
                      (gain - smoothed_gain);


    /* Apply power law compression and output gain. */
    *output = (FastPow(smoothed_gain * energy + compressor_delta,
                       compressor_exponent) -
               kCompressorStabilization);

    if (warm_up_counter) { --warm_up_counter; }

    ++output;
    input += decimation_factor;
  }

  state->smoothed_energy = smoothed_energy;
  state->smoothed_gain = smoothed_gain;
  state->warm_up_counter = warm_up_counter;
}

static float SmootherTimeConstant(float tau_s, float sample_rate_hz) {
  return 1.0f - exp(-1.0f / (tau_s * sample_rate_hz));
}

int /*bool*/ SparsePeakPickerInit(SparsePeakPicker* state,
                                  const SparsePeakPickerParams* params,
                                  float sample_rate_hz) {
  if (state == NULL || params == NULL) {
    fprintf(stderr, "SparsePeakPickerInit: Null argument.\n");
    return 0;
  } else if (!(sample_rate_hz > 0.0f) ||
             !(params->smoothed_tau_attack_s > 0.0f) ||
             !(params->smoothed_tau_release_s > 0.0f) ||
             !(params->thresh_tau_attack_s > 0.0f) ||
             !(params->thresh_tau_release_s > 0.0f) ||
             !(params->min_peak_spacing_s > 0.0f)) {
    fprintf(stderr, "SparsePeakPickerInit: Invalid params.\n");
    return 0;
  }

  state->smoothed_coeffs[0] =
      SmootherTimeConstant(params->smoothed_tau_attack_s, sample_rate_hz);
  state->smoothed_coeffs[1] =
      SmootherTimeConstant(params->smoothed_tau_release_s, sample_rate_hz);
  state->thresh_coeffs[0] =
      SmootherTimeConstant(params->thresh_tau_attack_s, sample_rate_hz);
  state->thresh_coeffs[1] =
      SmootherTimeConstant(params->thresh_tau_release_s, sample_rate_hz);
  state->min_peak_spacing_samples =
      (int)(params->min_peak_spacing_s * sample_rate_hz + 0.5f);

  SparsePeakPickerReset(state);
  return 1;
}

void SparsePeakPickerReset(SparsePeakPicker* state) {
  state->smoothed[0] = 0.0f;
  state->smoothed[1] = 0.0f;
  state->thresh = 0.0f;
  state->peak = 0.0f;
  state->spacing_counter = state->min_peak_spacing_samples;
}

float SparsePeakPickerProcessSamples(SparsePeakPicker* state,
                                     const float* samples, int num_samples) {
  float last_picked_peak = 0.0f;

  int i;
  for (i = 0; i < num_samples; ++i) {
    const float sample = samples[i];
    /* Apply asymmetric smoother to the input signal. */
    state->smoothed[0] += state->smoothed_coeffs[state->smoothed[0] > sample] *
                          (sample - state->smoothed[0]);
    /* The asymmetric smoother tends to leave cusps in the signal, so run a
     * another one-pole smoothing using the attack smoothing coefficient.
     */
    state->smoothed[1] += state->smoothed_coeffs[0] *
                          (state->smoothed[0] - state->smoothed[1]);
    /* Apply another asymmetric smoother with longer time constants to get a
     * threshold signal.
     */
    state->thresh += state->thresh_coeffs[state->thresh > state->smoothed[1]] *
                     (state->smoothed[1] - state->thresh);

    if (state->spacing_counter) {
      /* Decrement counter if a peak was recently picked. */
      state->spacing_counter--;
    } else {
      /* When smoothed signal > threshold signal, find the peak value. */
      const float new_peak = state->smoothed[1];
      if (state->smoothed[1] >= state->thresh && new_peak >= state->peak) {
        state->peak = new_peak;
      } else if (state->peak > 0.05f) {
        /* Pick the peak. */
        last_picked_peak = state->peak;
        state->peak = 0.0f;
        state->spacing_counter = state->min_peak_spacing_samples;
      }
    }
  }

  return last_picked_peak;
}
