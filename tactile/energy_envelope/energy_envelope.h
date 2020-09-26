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
 *
 * Tactor energy envelope, a simple audio-to-tactile representation.
 *
 * This library implements the following energy envelope [formerly "tactor
 * 3-channel"] design, which converts an input mono audio stream to four
 * tactors (vibrating actuators):
 *
 *   Tactor 0: Baseband channel with 80-500 Hz sensitivity to capture low pitch
 *             harmonics, percussion, drums.
 *
 *   Tactor 1: Vowel channel, bandpassed to 500-3500 Hz to get speech formants,
 *             followed by an envelope with 500 Hz cutoff to get pitch rate.
 *
 *   Tactor 2: Sh fricative channel, bandpassed to 2500-3500 Hz to pick out
 *             postalveolar sounds like "sh" vs. other fricatives.
 *
 *   Tactor 3: Fricative channel, bandpassed to 4000-6000 Hz followed by
 *             an envelope, to capture frication noise.
 *
 * For compression and onset emphasis, we apply PCEN [per-channel energy
 * normalization, go/pcen] to each tactor.
 *
 * The implementation is that each tactor is an instance of a `EnergyEnvelope`
 * class, but with different filter cutoffs for baseband / vowel / fricative.
 * For each tactor, audio is processed this way:
 *   1. A 2nd order Butterworth bandpass filter is applied, implemented as two
 *      second-order sections.
 *   2. The signal is half-wave rectified and squared to obtain energy.
 *   3. A 2nd order Butterworth lowpass filter is applied to get an energy
 *      envelope.
 *   4. The energy envelope is decimated.
 *   5. To compute the smoothed energy for PCEN's denominator, one-pole lowpass
 *      filtering is applied.
 *   6. We compute PCEN on the decimated energy envelope from step 4.
 *   7. Final output is multiplied by a constant output gain factor.
 *
 * Example use:
 *   // Initialize a vowel energy envelope.
 *   EnergyEnvelope state;
 *   EnergyEnvelopeInit(&state, &kEnergyEnvelopeVowelParams,
 *                      input_sample_rate_hz, decimation_factor);
 *
 *   // Processing loop.
 *   const int kBlockSize = 64;
 *   while (...) {
 *     float input[kBlockSize] = ...
 *     float output[kBlockSize];
 *     EnergyEnvelopeProcessSamples(&state, input, num_samples, output, 1);
 *     ...
 *   }
 */
#ifndef AUDIO_TO_TACTILE_TACTILE_ENERGY_ENVELOPE_ENERGY_ENVELOPE_H_
#define AUDIO_TO_TACTILE_TACTILE_ENERGY_ENVELOPE_ENERGY_ENVELOPE_H_

#include "dsp/biquad_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parameters for one tactor channel. Suggested defaults are defined in
 *  * kEnergyEnvelopeBasebandParams,
 *  * kEnergyEnvelopeVowelParams,
 *  * kEnergyEnvelopeShFricativeParams, and
 *  * kEnergyEnvelopeFricativeParams.
 */
typedef struct {
  /* Low and high cutoff edges in Hz for the 2nd order Butterworth bandpass
   * filter. This bandpass filter is the first processing that the input signal
   * passes through.
   */
  float bpf_low_edge_hz;
  float bpf_high_edge_hz;
  /* Cutoff in Hz for the one-pole energy smoothing filter. */
  float energy_smoother_cutoff_hz;
  /* Time constant for the one-pole smoother for PCEN's denominator. */
  float pcen_time_constant_s;
  /* PCEN denominator exponent. */
  float pcen_alpha;
  /* PCEN beta exponent. */
  float pcen_beta;
  /* PCEN denominator offset, should be tuned proportional to the noise. */
  float pcen_gamma;
  /* PCEN zero offset. */
  float pcen_delta;
  /* The final output is multiplied by this gain. */
  float output_gain;
} EnergyEnvelopeParams;

/* Data and state variables for one tactor channel. */
typedef struct {
  /* Bandpass filter coefficients, represented as two second-order sections. */
  BiquadFilterCoeffs bpf_biquad_coeffs[2];
  /* Energy envelope smoothing coefficients. */
  BiquadFilterCoeffs energy_biquad_coeffs;
  /* Decimation factor after computing the energy envelope. */
  int decimation_factor;
  /* PCEN denominator smoother coefficient. */
  float pcen_smoother_coeff;
  /* PCEN tuning knobs. */
  float pcen_alpha;
  float pcen_beta;
  float pcen_gamma;
  float pcen_delta;
  /* Precomputation of pcen_offset = pow(pcen_beta, pcen_delta). */
  float pcen_offset;
  float output_gain;

  /* State variables for all the filters. */
  BiquadFilterState bpf_biquad_state[2];
  BiquadFilterState energy_biquad_state;
  float pcen_denom;
} EnergyEnvelope;

/* Initialize state with the specified parameters. The output sample rate is
 * input_sample_rate_hz / decimation_factor. Returns 1 on success, 0 on failure.
 */
int EnergyEnvelopeInit(EnergyEnvelope* state,
                       const EnergyEnvelopeParams* params,
                       float input_sample_rate_hz,
                       int decimation_factor);

/* Resets to initial state. */
void EnergyEnvelopeReset(EnergyEnvelope* state);

/* Process audio in a streaming manner. The `input` pointer should point to a
 * contiguous array of `num_samples` samples, where `num_samples` is a multiple
 * of `decimation_factor`. The number of output samples written is
 * num_samples / decimation_factor. The ith output sample is written to
 * `output[output_stride * i]`, e.g. output_stride == 2 to write a channel of
 * interleaved stereo output. In-place processing output == input is allowed.
 *
 * NOTE: It is assumed that `num_samples` is a multiple of `decimation_factor`.
 * Otherwise, the last `num_samples % decimation_factor` samples are ignored.
 */
void EnergyEnvelopeProcessSamples(EnergyEnvelope* state,
                                  const float* input,
                                  int num_samples,
                                  float* output,
                                  int output_stride);

/* Suggested parameters for the three kinds of tactor channel. */
extern const EnergyEnvelopeParams kEnergyEnvelopeBasebandParams;
extern const EnergyEnvelopeParams kEnergyEnvelopeVowelParams;
extern const EnergyEnvelopeParams kEnergyEnvelopeShFricativeParams;
extern const EnergyEnvelopeParams kEnergyEnvelopeFricativeParams;

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_TACTILE_ENERGY_ENVELOPE_ENERGY_ENVELOPE_H_ */

