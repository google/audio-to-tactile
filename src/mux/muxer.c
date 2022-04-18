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

#include "mux/muxer.h"

#include <math.h>
#include <stdlib.h>

#include "dsp/complex.h"
#include "dsp/logging.h"
#include "dsp/fft.h"
#include "dsp/math_constants.h"
#include "dsp/phase32.h"

/* Radius of Weaver lowpass filter in units of upsampled muxed samples. */
#define kMuxerWeaverLpfFilterRadius 511

/* Number of taps per phase in the weaver_lpf polyphase filter. */
#define kLpfNumTaps 64

typedef struct {
  Oscillator down_converter;
  /* Buffered downconverted samples. */
  ComplexFloat buffer[kLpfNumTaps];
  Oscillator up_converter;
  Oscillator pilot;
} MuxerChannel;

struct Muxer {
  MuxerChannel channels[kMuxChannels];
  int samples_in_buffer;
  int buffer_position;
  float weaver_lpf[kMuxRateFactor * kLpfNumTaps];
};

/* Gets muxer Weaver lowpass filter by windowed FIR design. */
static void MuxerDesignWeaverLpf(float* polyphase_coeffs) {
  const int kFftSize = 1024;
  ComplexFloat* buffer =
      (ComplexFloat*)CHECK_NOTNULL(malloc(sizeof(ComplexFloat) * kFftSize));

  BiquadFilterCoeffs demuxer_lpf;
  DemuxerDesignWeaverLpf(&demuxer_lpf);

  /* The demuxer should be cheap, since it runs in real time on device, so it
   * uses a cheap IIR filter. As a result, the demuxer filter's frequency
   * response is not great with a slow roll off that attenuates part of the
   * passband and weak suppression in the stopband.
   *
   * To compensate, we use a high quality filter in the muxer, a large windowed
   * FIR design. The target response is the Wiener deconvolution filter for the
   * demuxer's filter within the passband and zero in the stopband. This way,
   * the cascade of this filter and the demuxer's is close to ideal response.
   */
  int k;
  for (k = 0; k <= kFftSize / 2; ++k) {
    double cycles_per_sample = ((double)k) / kFftSize;
    ComplexDouble target_response = {0.0, 0.0};
    if (cycles_per_sample <= kMuxWeaverLpfCutoffHz / kMuxMuxedRate) {
      /* Get demuxer filter's frequency response. */
      ComplexDouble demuxer_response =
          BiquadFilterFrequencyResponse(&demuxer_lpf, cycles_per_sample);
      /* Compute its Wiener deconvolution filter:
       *
       *                       conj(demuxer_response)
       *   target_response = ---------------------------.
       *                     |demuxer_response|^2 + 1e-6
       */
      target_response = ComplexDoubleMulReal(
          ComplexDoubleConj(demuxer_response),
          ((double)kMuxRateFactor / kFftSize)
          / (ComplexDoubleAbs2(demuxer_response) + 1e-6));
    }

    buffer[k].real = (float)target_response.real;
    buffer[k].imag = (float)target_response.imag;
  }
  for (; k < kFftSize; ++k) { /* Get rest of spectrum by Hermitian symmetry. */
    buffer[k] = ComplexFloatConj(buffer[kFftSize - k]);
  }

  /* Compute inverse FFT of `buffer` in place. */
  FftScramble(buffer, kFftSize);
  FftInverseScrambledTransform(buffer, kFftSize);

  /* Multiply `buffer` pointwise with cosine window. */
  const double kWindowRadPerSample =
      M_PI / (2 * (kMuxerWeaverLpfFilterRadius + 1));
  const ComplexDouble rotator =
      ComplexDoubleMake(cos(kWindowRadPerSample), sin(kWindowRadPerSample));
  ComplexDouble phasor = ComplexDoubleMake(rotator.imag, -rotator.real);
  /* Absorb factor 2 for converting signal to real. */
  phasor = ComplexDoubleMulReal(phasor, 2.0);
  int i;
  for (i = -kMuxerWeaverLpfFilterRadius; i <= kMuxerWeaverLpfFilterRadius;
       ++i) {
    buffer[(i >= 0) ? i : i + kFftSize].real *= phasor.real;
    phasor = ComplexDoubleMul(phasor, rotator);
  }

  /* Rearrange filter for polyphase representation. */
  int phase;
  for (phase = 0, k = 0; phase < kMuxRateFactor; ++phase) {
    int n;
    for (n = 0; n < kLpfNumTaps; ++n, ++k) {
      i = kMuxerWeaverLpfFilterRadius + 1 + phase -
          kMuxRateFactor * (1 + n);
      i = (i >= 0) ? i : i + kFftSize;
      polyphase_coeffs[k] = (k != kLpfNumTaps - 1) ? buffer[i].real : 0.0f;
    }
  }

  free(buffer);
}

Muxer* MuxerMake(void) {
  Muxer* muxer = (Muxer*)malloc(sizeof(Muxer));
  if (muxer == NULL) { return NULL; }

  MuxerDesignWeaverLpf(muxer->weaver_lpf);
  MuxerReset(muxer);
  return muxer;
}

void MuxerFree(Muxer* muxer) {
  free(muxer);
}

void MuxerReset(Muxer* muxer) {
  muxer->samples_in_buffer =
      (kMuxerWeaverLpfFilterRadius + 1) / kMuxRateFactor - 1;
  muxer->buffer_position = 0;

  int c;
  for (c = 0; c < kMuxChannels; ++c) {
    MuxerChannel* channel = &muxer->channels[c];
    OscillatorInit(&channel->down_converter,
                   -kMuxMidpointHz / kMuxTactileRate);
    OscillatorInit(&channel->up_converter,
                   (kMuxMidpointHz + MuxCarrierFrequency(c)) / kMuxMuxedRate);
    OscillatorInit(
        &channel->pilot,
        (kMuxPilotHzAtBaseband + MuxCarrierFrequency(c)) / kMuxMuxedRate);

    int i;
    for (i = 0; i < kLpfNumTaps; ++i) {
      channel->buffer[i] = ComplexFloatMake(0.0f, 0.0f);
    }
  }
}

int MuxerNextOutputSize(Muxer* muxer, int num_input_frames) {
  int num_written = 0;
  int samples_in_buffer = muxer->samples_in_buffer;

  int i;
  for (i = 0; i < num_input_frames; ++i) {
    if (++samples_in_buffer > kLpfNumTaps) {
      samples_in_buffer = kLpfNumTaps;
    }

    if (samples_in_buffer == kLpfNumTaps) {
      num_written += kMuxRateFactor;
    }
  }
  return num_written;
}

int MuxerProcessSamples(Muxer* muxer, const float* tactile_input,
                        int num_frames, float* muxed_output) {
  int samples_in_buffer;
  int num_written;
  int p;

  int c;
  for (c = 0; c < kMuxChannels; ++c) {
    const float* input = tactile_input + c;
    float* output = muxed_output;
    num_written = 0;
    samples_in_buffer = muxer->samples_in_buffer;
    p = muxer->buffer_position;

    int i;
    for (i = 0; i < num_frames; ++i) {
      MuxerChannel* channel = &muxer->channels[c];

      /* Shift band midpoint down to DC and store value in `buffer[p]`. */
      channel->buffer[p] = ComplexFloatMulReal(
          Phase32ComplexExp(channel->down_converter.phase), *input);
      OscillatorNext(&channel->down_converter);

      if (++samples_in_buffer > kLpfNumTaps) {
        samples_in_buffer = kLpfNumTaps;
      }

      if (samples_in_buffer == kLpfNumTaps) {
        const float* lpf = muxer->weaver_lpf;
        int phase;
        int k;
        for (phase = 0, k = 0; phase < kMuxRateFactor; ++phase) {
          ComplexFloat filtered = {0.0f, 0.0f};
          int n;
          /* Apply Weaver lowpass filter. Conceptually, the input to the filter
           * is upsampled by zero insertion by factor kMuxRateFactor. We
           * efficiently implement this by polyphase filtering with the lowpass
           * filter divided into kMuxRateFactor different phases.
           *
           * The input samples are stored in a circular buffer, with `buffer[p]`
           * being the most recent sample. So we apply the filter for the
           * current phase starting at (p + 1)...
           */
          for (n = p + 1; n < kLpfNumTaps; ++n, ++k) {
            filtered = ComplexFloatAdd(
                filtered, ComplexFloatMulReal(channel->buffer[n], lpf[k]));
          }
          /* ... then wrap around and sum up to and including p. */
          for (n = 0; n <= p; ++n, ++k) {
            filtered = ComplexFloatAdd(
                filtered, ComplexFloatMulReal(channel->buffer[n], lpf[k]));
          }

          /* Shift upper sideband above the carrier frequency. */
          float sample =
              filtered.real * Phase32Cos(channel->up_converter.phase) -
              filtered.imag * Phase32Sin(channel->up_converter.phase);
          OscillatorNext(&channel->up_converter);

          /* Add pilot tone for synchronization. */
          sample += 0.05f * Phase32Cos(channel->pilot.phase);
          OscillatorNext(&channel->pilot);

          if (c == 0) {
            output[phase] = sample;
          } else {
            output[phase] += sample;
          }
        }

        output += kMuxRateFactor;
        num_written += kMuxRateFactor;
      }

      if (++p >= kLpfNumTaps) {
        p = 0;
      }
      input += kMuxChannels;
    }
  }

  muxer->samples_in_buffer = samples_in_buffer;
  muxer->buffer_position = p;
  return num_written;
}
