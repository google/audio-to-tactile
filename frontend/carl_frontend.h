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
 *
 * Audio-to-frontend, using the CARL filterbank + PCEN.
 *
 * A Cascade of Asymmetric Resonators, Linear [CARL], filterbank is used to
 * analyze the input audio to bandpass channels. Each channel is then half-wave
 * rectified, squared, and lowpass filtered to get energy envelopes:
 *
 *   input -> [CARL] -> [Rectify and square] -> [LPF] -> energy envelopes
 *
 * For each channel, the energy envelope is compressed with PCEN (go/pcen). The
 * energy envelope is lowpass filtered to make the PCEN denominator.
 *
 *   energy envelope -+-------> [PCEN] -> output
 *                    |           ^
 *                    +-> [LPF] --+
 */
#ifndef AUDIO_TO_TACTILE_FRONTEND_CARL_FRONTEND_H_
#define AUDIO_TO_TACTILE_FRONTEND_CARL_FRONTEND_H_

#ifdef __cplusplus
extern "C" {
#endif

struct CarlFrontendParams {
  /* Input sample rate in Hz. */
  float input_sample_rate_hz;
  /* Input block size, also the decimation factor. The output sample rate is
   * input_sample_rate_hz / block_size. Must be a power of two.
   */
  int block_size;
  /* Highest frequency to look at in Hz. Pole frequency of channel 0. */
  float highest_pole_frequency_hz;
  /* Lower bound on generated pole frequencies. */
  float min_pole_frequency_hz;
  /* Channel spacing in equivalent rectangular bandwidth (ERB) units. */
  float step_erbs;
  /* Cutoff frequency in Hz for smoothing energy envelopes. */
  float envelope_cutoff_hz;

  /* Time constant for the lowpass filter that produces the PCEN denominator. */
  float pcen_time_constant_s;
  /* Diffusivity for smoothing PCEN denominator across channels,
   *
   *   d/dt u[c] = pcen_cross_channel_diffusivity * (u[c+1] - 2 u[c] - u[c-1]).
   *
   * For stability it is required that
   *
   *   pcen_cross_channel_diffusivity < 0.5 * input_sample_rate_hz / block_size.
   *
   * Diffusion over T seconds is like Gaussian smoothing with stddev
   * `sqrt(2 * pcen_cross_channel_diffusivity * T)` in units of channels.
   */
  float pcen_cross_channel_diffusivity;
  /* Initialization for the PCEN denominator lowpass filter. */
  float pcen_init_value;

  /* The following are parameters used in PCEN compression formula (go/pcen),
   * (envelope / (gamma + smoothed envelope)^alpha + delta)^beta - delta^beta.
   */
  float pcen_alpha;  /* PCEN denominator exponent, between 0.0 and 1.0. */
  float pcen_beta;   /* PCEN beta (outer) exponent applied to the ratio. */
  float pcen_gamma;  /* PCEN denominator offset. */
  float pcen_delta;  /* PCEN zero offset. */
};
typedef struct CarlFrontendParams CarlFrontendParams;

extern const CarlFrontendParams kCarlFrontendDefaultParams;

struct CarlFrontend;
typedef struct CarlFrontend CarlFrontend;

/* Makes a CARL+PCEN frontend. The caller should free it when done with
 * CarlFrontendFree. Returns NULL on failure.
 */
CarlFrontend* CarlFrontendMake(const CarlFrontendParams* params);

/* Frees a CarlFrontend. */
void CarlFrontendFree(CarlFrontend* frontend);

/* Gets the number of output channels. */
int CarlFrontendNumChannels(const CarlFrontend* frontend);

/* Gets the block size. */
int CarlFrontendBlockSize(const CarlFrontend* frontend);

/* Resets the frontend to initial state. */
void CarlFrontendReset(CarlFrontend* frontend);

/* Runs the CARL+PCEN frontend in a streaming manner. `input` is an array of
 * `block_size` input samples at rate `sample_rate_hz`. `output` is  an array of
 * size `CarlFrontendNumChannels`.
 */
void CarlFrontendProcessSamples(CarlFrontend* frontend,
                                float* input,
                                float* output);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_FRONTEND_CARL_FRONTEND_H_ */
