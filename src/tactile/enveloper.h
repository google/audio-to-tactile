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
 * Tactor energy envelope, a simple audio-to-tactile representation.
 *
 * This library computes energy envelopes over 4 bandpass channels of an input
 * mono audio stream:
 *
 *   Channel 0: Baseband channel with 80-500 Hz sensitivity to capture low pitch
 *              harmonics, percussion, drums.
 *
 *   Channel 1: Vowel channel, bandpassed to 500-3500 Hz to get speech formants,
 *              followed by an envelope with 500 Hz cutoff to get pitch rate.
 *
 *   Channel 2: Sh fricative channel, bandpassed to 2500-3500 Hz to pick out
 *              postalveolar sounds like "sh" vs. other fricatives.
 *
 *   Channel 3: Fricative channel, bandpassed to 4000-6000 Hz followed by
 *              an envelope, to capture frication noise.
 *
 * For compression and onset emphasis, we apply a modified version of PCEN
 * [per-channel energy normalization, go/pcen] to each channel.
 *
 * For each channel, audio is processed this way:
 *
 *   1. Bandpass energy computation.
 *     a. A bandpass filter is applied, implemented as two biquads.
 *     b. The signal is half-wave rectified and squared to obtain energy.
 *     c. A second order lowpass filter is applied to the energy.
 *     d. The lowpassed energy is decimated.
 *
 *   2. Soft noise gating. We want to normalize speech and salient sounds toward
 *      0 dB, yet we don't want to amplify noise. Our assumption is that the
 *      noise envelope changes slowly and that salient sounds are bursty and
 *      louder than the noise.
 *
 *     a. A smoothed energy is computed using a one-pole smoother.
 *
 *     b. The noise energy is estimated by smoothing the smoothed energy in log
 *        space. Filtering in log space puts more weight on small values and is
 *        less influenced by short bursts of energy in the signal.
 *
 *     c. The noise gate soft threshold is set at `denoising_strength * noise`
 *        and with a transition width of `denoising_transition_db` dB.
 *
 *   3. A (partially) normalized energy is computed as
 *      agc_output = soft_gate_weight * smoothed_energy^-agc_strength * energy.
 *
 *   4. The normalized energy is compressed with a power law as
 *      `(agc_output + delta)^exponent - delta^exponent`.
 *
 *   5. Final output is multiplied by a constant output gain factor.
 *
 * Example use:
 *   // Initialize a vowel energy envelope.
 *   Enveloper enveloper;
 *   EnveloperInit(&enveloper, &kDefaultEnveloperParams,
 *                 input_sample_rate_hz, decimation_factor);
 *
 *   // Processing loop.
 *   const int kNumSamples = 64;
 *   const int kOutputFrames = kNumSamples / decimation_factor;
 *   while (...) {
 *     float input[kNumSamples] = ...
 *     float output[kOutputFrames * kEnveloperNumChannels];
 *     EnveloperProcessSamples(&state, input, kNumSamples, output);
 *     ...
 *   }
 */
#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_ENERGY_ENVELOPE_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_ENERGY_ENVELOPE_H_

#include "dsp/biquad_filter.h"
#include "dsp/decibels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of bandpass channels. */
#define kEnveloperNumChannels 4

/* Parameters for one bandpass channel. */
typedef struct {
  /* Low and high cutoff edges in Hz for the 2nd order Butterworth bandpass
   * filter. This bandpass filter is the first processing that the input signal
   * passes through.
   */
  float bpf_low_edge_hz;
  float bpf_high_edge_hz;
  /* The final output is multiplied by this gain. */
  float output_gain;
} EnveloperChannelParams;

typedef struct {
  EnveloperChannelParams channel_params[kEnveloperNumChannels];

  /* Cutoff in Hz for the energy lowpass filter. Usually 500 Hz. */
  float energy_cutoff_hz;

  /* Time constant in seconds for computing a smoothed energy, used for noise
   * estimation and PCEN's AGC.
   */
  float energy_tau_s;

  /* Noise estimate adaptation rate in dB per second. */
  float noise_db_s;
  /* Denoising strength, a positive value where larger means stronger denoising.
   * This parameter scales the soft noise gate threshold.
   */
  float denoising_strength;
  /* Soft noise gate transition width in dB. A large value makes the gate
   * more gradual, which helps avoid a "breathing" artifact from noise
   * fluctuations crossing above the threshold.
   */
  float denoising_transition_db;

  /* Parameters for per-channel energy normalization (PCEN). */
  /* Normalization strength (0 => bypass, 1 => full normalization). */
  float agc_strength;
  /* Compression exponent in a memoryless nonlinearity, between 0.0 and 1.0. */
  float compressor_exponent;
  /* Delta added to stabilize the compression. */
  float compressor_delta;
} EnveloperParams;
extern const EnveloperParams kDefaultEnveloperParams;

typedef struct {
  /* Bandpass filter coefficients, represented as two second-order sections. */
  BiquadFilterCoeffs bpf_biquad_coeffs[2];
  float peak;
  float equalization;
  float output_gain;

  BiquadFilterState bpf_biquad_state[2];
  BiquadFilterState energy_biquad_state;
  float smoothed_energy;
  float noise;
} EnveloperChannel;

/* Enveloper data and state variables. */
typedef struct {
  EnveloperChannel channels[kEnveloperNumChannels];

  /* Energy envelope smoothing coefficients. */
  BiquadFilterCoeffs energy_biquad_coeffs;
  /* Input sample rate in Hz. */
  float input_sample_rate_hz;
  /* Decimation factor after computing the energy envelope. */
  int decimation_factor;

  float energy_smoother_coeff;
  float noise_coeffs[2];
  float gate_thresh_factor;
  float gate_transition_factor;
  float agc_exponent;
  float compressor_exponent;
  float compressor_delta;
  float compressor_offset;
} Enveloper;

/* Initialize state with the specified parameters. The output sample rate is
 * input_sample_rate_hz / decimation_factor. Returns 1 on success, 0 on failure.
 */
int EnveloperInit(Enveloper* state,
                  const EnveloperParams* params,
                  float input_sample_rate_hz,
                  int decimation_factor);

/* Resets to initial state. */
void EnveloperReset(Enveloper* state);

/* Process audio in a streaming manner. The `input` pointer should point to a
 * contiguous array of `num_samples` samples, where `num_samples` is a multiple
 * of `decimation_factor`. The output has `num_samples / decimation_factor` and
 * `kEnveloperNumChannels` channels, written in interleaved order. In-place
 * processing output == input is not allowed.
 *
 * NOTE: It is assumed that `num_samples` is a multiple of `decimation_factor`.
 * Otherwise, the last `num_samples % decimation_factor` samples are ignored.
 */
void EnveloperProcessSamples(Enveloper* state,
                             const float* input,
                             int num_samples,
                             float* output);

/* Computes the smoother coefficient for a one-pole lowpass filter with time
 * constant `tau_s` in units of seconds. The coefficient should be used as:
 *
 *   y[n] = y[n - 1] + coeff * (x[n] - [y[n - 1]).
 */
static float EnveloperSmootherCoeff(const Enveloper* state, float tau_s) {
  return 1.0f - (float)exp(-state->decimation_factor /
                           (state->input_sample_rate_hz * tau_s));
}

/* Computes a coefficient for growing at a specified rate in dB per second. */
static float EnveloperGrowthCoeff(const Enveloper* state, float growth_db_s) {
  return DecibelsToPowerRatio(
      growth_db_s * state->decimation_factor / state->input_sample_rate_hz);
}

/* Updates precomputed params, useful if noise_coeffs or compressor params have
 * been changed.
 */
void EnveloperUpdatePrecomputedParams(Enveloper* state);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_ENERGY_ENVELOPE_H_ */
