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
 * Design functions and internal structures for CARL+PCEN frontend.
 */

#ifndef AUDIO_TACTILE_FRONTEND_CARL_FRONTEND_DESIGN_H_
#define AUDIO_TACTILE_FRONTEND_CARL_FRONTEND_DESIGN_H_

#ifdef __cplusplus
extern "C" {
#endif

struct CarlFrontendChannelData {
  int should_decimate;
  float biquad_b0;
  float biquad_b1;
  float biquad_b2;
  float biquad_a1;
  float biquad_a2;
  float envelope_smoother_coeff;
};
typedef struct CarlFrontendChannelData CarlFrontendChannelData;

struct CarlFrontendChannelState {
  float biquad_state0;
  float biquad_state1;
  float diff_state;
  float energy_envelope_stage1;
  float energy_envelope;
  float pcen_denom;
};
typedef struct CarlFrontendChannelState CarlFrontendChannelState;

struct CarlFrontend {
  CarlFrontendChannelData* channel_data;
  CarlFrontendChannelState* channel_state;

  int num_channels;
  int block_size;

  float pcen_smoother_coeff;
  float pcen_cross_channel_smoother_coeff;
  float pcen_init_value;
  float pcen_alpha;
  float pcen_beta;
  float pcen_gamma;
  float pcen_delta;
  float pcen_offset;
};

/* Gets the next pole frequency, `step_erbs` ERBs below `frequency_hz`. */
double CarlFrontendNextAuditoryFrequency(double frequency_hz, double step_erbs);

/* Designs asymmetric resonator biquad and writes coeffs in `channel_data`. */
void CarlFrontendDesignBiquad(double pole_frequency_hz, double sample_rate_hz,
                              CarlFrontendChannelData* channel_data);

/* Returns CARL's peak gain for channel `channel_index`. `*peak_frequency_hz`
 * should be set to an initial guess before calling the function, and is
 * replaced with the peak frequency.
 */
double CarlFrontendFindPeakGain(const CarlFrontendChannelData* channel_data,
                                int channel_index, double input_sample_rate_hz,
                                double* peak_frequency_hz);
#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TACTILE_FRONTEND_CARL_FRONTEND_DESIGN_H_ */
