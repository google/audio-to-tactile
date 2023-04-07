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
 *
 * "Tactile lite": audio representation with sparse, single-channel output.
 *
 * This library implements a "lite" version of our processing, intended to drive
 * a single tactor with short clicks or buzzes in response to onsets. The
 * processing is a single-band version of Enveloper followed logic to pick
 * onset peaks.
 *
 * Example use:
 *   SingleBandEnvelope envelope;
 *   SparsePeakPicker peak_picker;
 *   if (!SingleBandEnvelopeInit(&envelope, &kDefaultSingleBandEnvelopeParams,
 *                               sample_rate_hz, decimation_factor) ||
 *       !SparsePeakPickerInit(&peak_picker, &kDefaultSparsePeakPickerParams,
 *                             sample_rate_hz / decimation_factor)) {
 *     exit(1);  // Failed to initialize.
 *   }
 *
 *   int input_size = 64;
 *   int tactile_size = input_size / decimation_factor;
 *   float tactile[tactile_size];
 *   while (1) {
 *     float* input = // Get the next block of input samples...
 *     SingleBandEnvelopeProcessSamples(&envelope, input, input_size, tactile);
 *     float peak = SparsePeakPickerProcessSamples(&peak_picker,
 *                                                 tactile, tactile_size);
 *     if (peak > 0.0f) {
 *       // Output a click or buzz with amplitude `peak`.
 *     }
 *   }
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_LITE_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_LITE_H_

#include "dsp/biquad_filter.h"
#include "dsp/decibels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  /* Feedback may be a problem in applications where the input is live
   * microphone audio, and if that microphone is close enough to the output
   * tactor to pick up vibration noise. We avoid feedback using a 4th-order
   * Cheybshev type 2 highpass filter. feedback_hpf_cutoff_hz is the cutoff in
   * Hz for highpass filter to suppress tactor noise feedback.
   *
   * In applications where feedback is not a problem (e.g. because the audio
   * is prerecorded), this highpass filter may be bypassed by setting
   * feedback_hpf_cutoff_hz = 0.
   */
  float feedback_hpf_cutoff_hz;
  /* Stopband ripple in dB for the feedback highpass filter. */
  float feedback_hpf_stopband_ripple_db;

  /* Low and high cutoff edges in Hz for the 2nd order Butterworth bandpass
   * filter. This bandpass filter is the first processing that the input signal
   * passes through.
   */
  float bpf_low_edge_hz;
  float bpf_high_edge_hz;
  /* Denoising strength, a positive value where larger means stronger denoising.
   * This parameter scales the soft noise gate threshold.
   */
  float denoising_strength;

  /* Cutoff in Hz for the energy lowpass filter. Usually 500 Hz. */
  float energy_cutoff_hz;

  /* Time constant in seconds for computing a smoothed energy, used for noise
   * estimation and PCEN's AGC.
   */
  float energy_tau_s;

  /* Noise estimate adaptation rate in dB per second. */
  float noise_db_s;
  /* Soft noise gate transition width in dB. A large value makes the gate
   * more gradual, which helps avoid a "breathing" artifact from noise
   * fluctuations crossing above the threshold.
   */
  float denoising_transition_db;

  /* Parameters for per-channel energy normalization (PCEN). */
  /* Normalization strength (0 => bypass, 1 => full normalization). */
  float agc_strength;
  /* Gain smoothing attack and release time constants in seconds. */
  float gain_tau_attack_s;
  float gain_tau_release_s;
  /* Compression exponent in a memoryless nonlinearity, between 0.0 and 1.0. */
  float compressor_exponent;
} SingleBandEnvelopeParams;
extern const SingleBandEnvelopeParams kDefaultSingleBandEnvelopeParams;

typedef struct {
  float smoothed_tau_attack_s;
  float smoothed_tau_release_s;
  float thresh_tau_attack_s;
  float thresh_tau_release_s;
  float min_peak_spacing_s;
} SparsePeakPickerParams;
extern const SparsePeakPickerParams kDefaultSparsePeakPickerParams;

typedef struct {
  /* Input sample rate in Hz. */
  float input_sample_rate_hz;
  /* Feedback highpass filter coeffs, represented as two 2nd-order sections. */
  BiquadFilterCoeffs feedback_hpf_biquad_coeffs[2];
  /* Bandpass filter coefficients. */
  BiquadFilterCoeffs bpf_biquad_coeffs[2];
  /* Energy envelope smoothing coefficients. */
  BiquadFilterCoeffs energy_biquad_coeffs;
  float gate_thresh_factor;
  float energy_smoother_coeff;
  float noise_coeffs[2];
  float gate_transition_factor;
  float agc_exponent;
  float gain_smoother_coeffs[2]; /* [0] = attack coeff, [1] = release coeff. */
  float compressor_exponent;
  float compressor_delta;
  int num_warm_up_samples;

  BiquadFilterState feedback_hpf_biquad_state[2];
  BiquadFilterState bpf_biquad_state[2];
  BiquadFilterState energy_biquad_state;
  /* Decimation factor after computing the energy envelope. */
  int decimation_factor;
  float smoothed_energy;
  float noise;
  float smoothed_gain;
  int warm_up_counter;
} SingleBandEnvelope;

typedef struct {
  float smoothed_coeffs[2]; /* [0] = attack coeff, [1] = release coeff. */
  float thresh_coeffs[2]; /* [0] = attack coeff, [1] = release coeff. */
  int min_peak_spacing_samples;
  float smoothed[2];
  float thresh;
  float peak;
  int spacing_counter;
} SparsePeakPicker;

/* Initialize state with the specified parameters. The output sample rate is
 * input_sample_rate_hz / decimation_factor. Returns 1 on success, 0 on failure.
 */
int /*bool*/ SingleBandEnvelopeInit(SingleBandEnvelope* state,
                                    const SingleBandEnvelopeParams* params,
                                    float input_sample_rate_hz,
                                    int decimation_factor);

/* Resets to initial state. */
void SingleBandEnvelopeReset(SingleBandEnvelope* state);

/* Process audio in a streaming manner. The `input` pointer should point to a
 * contiguous array of `num_samples` samples, where `num_samples` is a multiple
 * of `decimation_factor`. The output has `num_samples / decimation_factor`
 * samples. In-place processing output == input is not allowed.
 *
 * NOTE: It is assumed that `num_samples` is a multiple of `decimation_factor`.
 * Otherwise, the last `num_samples % decimation_factor` samples are ignored.
 */
void SingleBandEnvelopeProcessSamples(SingleBandEnvelope* state,
                                      const float* input,
                                      int num_samples,
                                      float* output);

/* Computes the smoother coefficient for a one-pole lowpass filter with time
 * constant `tau_s` in units of seconds. The coefficient should be used as:
 *
 *   y[n] = y[n - 1] + coeff * (x[n] - [y[n - 1]).
 */
static float SingleBandEnvelopeSmootherCoeff(const SingleBandEnvelope* state,
                                             float tau_s) {
  return 1.0f - (float)exp(-state->decimation_factor /
                           (state->input_sample_rate_hz * tau_s));
}

/* Computes a coefficient for growing at a specified rate in dB per second. */
static float SingleBandEnvelopeGrowthCoeff(const SingleBandEnvelope* state,
                                           float growth_db_s) {
  return DecibelsToPowerRatio(
      growth_db_s * state->decimation_factor / state->input_sample_rate_hz);
}

/* Updates precomputed params, useful if noise_coeffs or compressor params have
 * been changed.
 */
void SingleBandEnvelopeUpdatePrecomputedParams(SingleBandEnvelope* state);

/* Initialize state with the specified parameters. Returns 1 on success. */
int /*bool*/ SparsePeakPickerInit(SparsePeakPicker* state,
                                  const SparsePeakPickerParams* params,
                                  float sample_rate_hz);

/* Resets to initial state. */
void SparsePeakPickerReset(SparsePeakPicker* state);

/* Process audio in a streaming manner. The `input` pointer should point to a
 * contiguous array of `num_samples` samples. If a peak is picked from any of
 * the input samples, its height is returned, and otherwise returns 0.0f.
 */
float SparsePeakPickerProcessSamples(SparsePeakPicker* state,
                                     const float* samples, int num_samples);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_TACTILE_LITE_H_ */
