/* Copyright 2020 Google LLC
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

#include "mux/demuxer.h"

#include <math.h>

#include "dsp/logging.h"
#include "dsp/math_constants.h"

void DemuxerInit(Demuxer* demuxer) {
  const float kShiftedPilotHz = kMuxPilotHzAtBaseband - kMuxMidpointHz;
  PilotTrackerCoeffsInit(&demuxer->pilot_tracker_coeffs, kShiftedPilotHz,
                         kMuxMuxedRate);

  DemuxerDesignWeaverLpf(&demuxer->weaver_lpf_coeffs);
  /* Absorb factor 2 needed for converting the final signal to real. */
  demuxer->weaver_lpf_coeffs.b0 *= 2;
  demuxer->weaver_lpf_coeffs.b1 *= 2;
  demuxer->weaver_lpf_coeffs.b2 *= 2;

  int c;
  for (c = 0; c < kMuxChannels; ++c) {
    DemuxerChannel* channel = &demuxer->channels[c];
    OscillatorInit(&channel->down_converter,
                   -(kMuxMidpointHz + MuxCarrierFrequency(c)) / kMuxMuxedRate);
    PilotTrackerInit(&channel->pilot_tracker, &demuxer->pilot_tracker_coeffs);
    BiquadFilterInitZero(&channel->weaver_lpf_state_real);
    BiquadFilterInitZero(&channel->weaver_lpf_state_imag);
    OscillatorInit(&channel->up_converter,
                   kMuxPilotHzAtBaseband / kMuxTactileRate);
  }
}

void DemuxerProcessSamples(Demuxer* demuxer, const float* muxed_input,
                           int num_samples, float* tactile_output) {
  CHECK(num_samples % kMuxRateFactor == 0);

  const int num_output_frames = num_samples / kMuxRateFactor;

  int c;
  for (c = 0; c < kMuxChannels; ++c) {
    DemuxerChannel* channel = &demuxer->channels[c];
    const float* input = muxed_input;
    float* output = tactile_output + c;

    int i;
    for (i = 0; i < num_output_frames; ++i) {
      int j;
      for (j = 0; j < kMuxRateFactor; ++j) {
        /* Shift band midpoint down to DC. */
        ComplexFloat sample;
        sample.real = Phase32Cos(channel->down_converter.phase) * input[j];
        sample.imag = Phase32Sin(channel->down_converter.phase) * input[j];
        OscillatorNext(&channel->down_converter);

        /* Phase-locked loop to track the pilot's phase. */
        Phase32 pilot_phase = PilotTrackerProcessOneSample(
            &channel->pilot_tracker, &demuxer->pilot_tracker_coeffs, sample);

        /* Lowpass filter to the band. */
        sample.real = BiquadFilterProcessOneSample(
            &demuxer->weaver_lpf_coeffs, &channel->weaver_lpf_state_real,
            sample.real);
        sample.imag = BiquadFilterProcessOneSample(
            &demuxer->weaver_lpf_coeffs, &channel->weaver_lpf_state_imag,
            sample.imag);

        if (j == 0) {
          /* Shift up to recover the baseband signal, at the same time adjusting
           * phase for synchronization based on the pilot phase.
           */
          *output = sample.real *
                        Phase32Cos(channel->up_converter.phase - pilot_phase) -
                    sample.imag *
                        Phase32Sin(channel->up_converter.phase - pilot_phase);
          OscillatorNext(&channel->up_converter);
        }
      }

      input += kMuxRateFactor;
      output += kMuxChannels;
    }
  }
}
