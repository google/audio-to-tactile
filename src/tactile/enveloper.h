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
 *   2. Automatic gain control with noise gating. We want to normalize speech
 *      and salient sounds toward 0 dB, yet we don't want to amplify noise. Our
 *      assumption is that the noise envelope changes slowly and that salient
 *      sounds are bursty and louder than the noise.
 *
 *     a. A smoothed energy is computed using a one-pole smoother.
 *
 *     b. To estimate the noise envelope, FastLog2 of the smoothed energy is
 *        computed, then smoothed with a one-pole smoother in log space. By
 *        filtering in log space, the noise envelope focuses on small values and
 *        is less influenced by short bursts of energy in the signal.
 *
 *     c. The noise gate threshold is computed by taking FastExp2 of the noise
 *        envelope and multiplying by `thresh_factor`. The AGC gain is then
 *        computed based on the x = smoothed energy, the threshold, and
 *        `agc_exponent`. For x <= threshold, the gain is zero. For
 *        x >> threshold, the gain is approximately x^-agc_exponent.
 *
 *     d. The gain is smoothed with an asymmetric smoother with fast attack.
 *
 *     e. A normalized energy is computed by multiplying the energy by the
 *        smoothed gain. If needed, the smoothed gain is reduced so that the
 *        normalized energy does not exceed 1.0.
 *
 *   3. The normalized energy is compressed with a power law as
 *      `(normalized_energy + delta)^exponent - delta^exponent`.
 *
 *   4. Final output is multiplied by a constant output gain factor.
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
  /* Factor for noise gate threshold. Larger implies stronger denoising. */
  float denoise_thresh_factor;
  /* The final output is multiplied by this gain. */
  float output_gain;
} EnveloperChannelParams;

typedef struct {
  EnveloperChannelParams channel_params[kEnveloperNumChannels];

  /* Cutoff in Hz for the energy lowpass filter. Usually 500 Hz. */
  float energy_cutoff_hz;
  /* Time constant in seconds for smoothing energy envelopes over time. */
  float energy_tau_s;
  /* Time constant in seconds for diffusing energy envelopes across channels,
   *
   *   d/dt u[c] = D * (u[c+1] - 2 u[c] - u[c-1]),
   *
   * where D = (num channels)^2 / cross_channel_tau_s. This creates a
   * cross-channel inhibition effect. Smaller value implies faster diffusion and
   * stronger inhibition.
   */
  float cross_channel_tau_s;
  /* Noise envelope time constant in seconds. */
  float noise_tau_s;
  /* Automatic gain control strength (0 => bypass, 1 => full normalization). */
  float agc_strength;
  /* Gain smoothing attack and release time constants in seconds. */
  float gain_tau_attack_s;
  float gain_tau_release_s;
  /* Compression exponent in a memoryless nonlinearity, between 0.0 and 1.0. */
  float compressor_exponent;
  /* Delta added to stabilize the compression. */
  float compressor_delta;
} EnveloperParams;
extern const EnveloperParams kDefaultEnveloperParams;

typedef struct {
  /* Bandpass filter coefficients, represented as two second-order sections. */
  BiquadFilterCoeffs bpf_biquad_coeffs[2];
  float denoise_thresh_factor;
  /* Precomputation of (1/output_gain + delta^exponent)^(1/exponent) - delta. */
  float agc_max_output;
  float output_gain;

  BiquadFilterState bpf_biquad_state[2];
  BiquadFilterState energy_biquad_state;
  float smoothed_energy;
  float log2_noise;
  float smoothed_gain;
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
  /* AGC parameters. */
  float energy_smoother_coeff;
  float cross_channel_diffusion_coeff;
  float noise_smoother_coeff;
  float agc_exponent;
  float gain_smoother_coeffs[2]; /* [0] = attack coeff, [1] = release coeff. */
  /* Compressor parameters. */
  float compressor_exponent;
  float compressor_delta;
  /* Precomputation of pow(compressor_delta, compressor_exponent). */
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

/* Computes the cross-channel diffusion coefficient. */
static float EnveloperCrossChannelDiffusionCoeff(const Enveloper* state,
                                                 float tau_s) {
  const float c = (kEnveloperNumChannels * kEnveloperNumChannels) *
      state->decimation_factor / (state->input_sample_rate_hz * tau_s);
  return c / (1.0f + 2.0f * c);
}

/* Updates precomputed params, useful if compressor params have been changed. */
void EnveloperUpdatePrecomputedParams(Enveloper* state);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_ENERGY_ENVELOPE_H_ */
