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

#include "src/mux/muxer.h"
#include "src/mux/demuxer.h"

#include <math.h>
#include <stdlib.h>

#include "src/dsp/iir_design.h"
#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"
#include "src/mux/mux_common.h"

/* Generates a random value uniformly in [0.0, 1.0]. */
static float RandUniform() { return (float)rand() / RAND_MAX; }

/* Generates a random value with approximately standard normal distribution. */
static float RandNormal() {
  return 2 * (RandUniform() + RandUniform() + RandUniform() - 1.5f);
}

/* The sinc function, Sinc(x) = sin(pi * x) / (pi * x). */
static float Sinc(float x) {
  const float y = M_PI * fabs(x);
  return (y < 1e-6f) ? 1.0f : sin(y) / y;
}

/* Evaluates the Hamming window over the interval -1 <= x <= 1. */
static float HammingWindow(float x) {
  return 0.54f + 0.46f * cos(M_PI * x);
}

/* Bandpass filters tactile signals to remove content outside of 10-500 Hz. */
static void FilterTactileSignalsToBand(float* tactile_signals, int num_frames) {
#define kBpfOrder 4
  BiquadFilterCoeffs coeffs[kBpfOrder];
  /* Make a 4th-order elliptic bandpass filter. To account for transition bands,
   * the edges are pushed in a bit to 15 and 410 Hz so that the stopbands
   * include 10 Hz and 500 Hz.
   */
  CHECK(DesignEllipticBandpass(/*order=*/kBpfOrder,
                               /*passband_ripple_db=*/0.5,
                               /*stopband_ripple_db=*/30.0,
                               /*low_edge_hz=*/15.0,
                               /*high_edge_hz=*/410.0,
                               /*sample_rate_hz=*/kMuxTactileRate,
                               /*coeffs=*/coeffs,
                               /*max_biquads=*/kBpfOrder) == kBpfOrder);

  int c;
  for (c = 0; c < kMuxChannels; ++c) {
    float* channel = tactile_signals + c;
    BiquadFilterState state[kBpfOrder];
    int k;
    for (k = 0; k < kBpfOrder; ++k) {
      BiquadFilterInitZero(&state[k]);
    }

    int i;
    for (i = 0; i < num_frames; ++i, channel += kMuxChannels) {
      float sample = *channel;
      for (k = 0; k < kBpfOrder; ++k) { /* Apply bandpass filter. */
        sample = BiquadFilterProcessOneSample(&coeffs[k], &state[k], sample);
      }
      *channel = sample;
    }
  }
}

/* Make some tactile test signals by filtering random sample values. */
static float* MakeTactileTestSignals(int num_frames) {
  const int num_samples = kMuxChannels * num_frames;
  float* tactile_signals = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * num_samples));

  int i;
  for (i = 0; i < num_samples; ++i) {
    tactile_signals[i] = 1.9f * (RandUniform() - 0.5f);
  }
  FilterTactileSignalsToBand(tactile_signals, num_frames);

  return tactile_signals;
}

/* Runs Muxer on `tactile_signals`. */
static float* RunMuxer(const float* tactile_signals, int num_frames,
                       int* num_muxed_samples) {
  Muxer* muxer = (Muxer*)CHECK_NOTNULL(MuxerMake());
  *num_muxed_samples = MuxerNextOutputSize(muxer, num_frames);
  float* muxed_signal =
      (float*)CHECK_NOTNULL(malloc(*num_muxed_samples * sizeof(float)));
  CHECK(MuxerProcessSamples(muxer, tactile_signals, num_frames, muxed_signal) ==
        *num_muxed_samples);
  MuxerFree(muxer);
  return muxed_signal;
}

/* Runs Demuxer on `muxed_signal`. */
static float* RunDemuxer(const float* muxed_signal, int num_samples,
                         int* num_demuxed_frames) {
  Demuxer demuxer;
  DemuxerInit(&demuxer);
  *num_demuxed_frames = num_samples / kMuxRateFactor;
  float* demuxed_signals = (float*)CHECK_NOTNULL(
      malloc(*num_demuxed_frames * kMuxChannels * sizeof(float)));
  DemuxerProcessSamples(&demuxer, muxed_signal, num_samples,
                        demuxed_signals);
  return demuxed_signals;
}

/* Simulates a possibly non-integer time delay and added noise. */
static float* SimulateDistortion(float* signal, int num_samples,
                                 float delay_in_samples,
                                 float noise_stddev) {
  const int integer_delay = (int)floor(delay_in_samples + 0.5f);
  const float fractional_delay = delay_in_samples - integer_delay;
  /* Make a resampling filter for the fractional part of the delay. */
#define kDelayRadius 10
  float filter[2 * kDelayRadius + 1];
  int k;
  for (k = -kDelayRadius; k <= kDelayRadius; ++k) {
    filter[kDelayRadius + k] = /* Hamming-windowed sinc kernel. */
        Sinc(k + fractional_delay) * HammingWindow((float)k / kDelayRadius);
  }

  float* distorted_signal = (float*)CHECK_NOTNULL(
      malloc(num_samples * sizeof(float)));
  int i;
  for (i = 0; i < num_samples; ++i) {
    float sum = noise_stddev * RandNormal();
    for (k = -kDelayRadius; k <= kDelayRadius; ++k) {
      const int j = i - integer_delay + k;
      if (0 <= j && j < num_samples) {
        sum += filter[kDelayRadius + k] * signal[j];
      }
    }
    distorted_signal[i] = sum;
  }

  return distorted_signal;
}

/* Test mux + demux round trip with a time delay and added noise. This test
 * that time synchronization in demuxing works robustly.
 */
void TestRoundTrip(float delay_in_samples, float noise_stddev) {
  printf("TestRoundTrip(%g, %g)\n", delay_in_samples, noise_stddev);
  const int kNumFrames = 1000;
  /* Make random tactile test signals.*/
  float* tactile_signals = MakeTactileTestSignals(kNumFrames);

  /* Multiplex the `tactile_signals` into `muxed_signal`. */
  int num_muxed_samples;
  float* muxed_signal =
      RunMuxer(tactile_signals, kNumFrames, &num_muxed_samples);

  float* received_signal = SimulateDistortion(muxed_signal, num_muxed_samples,
                                              delay_in_samples, noise_stddev);

  /* Demultiplex `received_signal` to `demuxed_signals`. */
  int num_demuxed_frames;
  float* demuxed_signals =
      RunDemuxer(received_signal, num_muxed_samples, &num_demuxed_frames);

  /* For each channel, compute signal to noise ratio of demuxed output. */
  int c;
  for (c = 0; c < kMuxChannels; ++c) {
    const float* expected = tactile_signals + c;
    const float* actual = demuxed_signals + c;
    float signal_energy = 0.0f;
    float noise_energy = 0.0f;

    int i;
    for (i = 0; i < num_demuxed_frames; ++i) {
      signal_energy += *expected * *expected;
      const float error = *expected - *actual;
      noise_energy += error * error;
      expected += kMuxChannels;
      actual += kMuxChannels;
    }

    const float snr_db = 10 * log(M_LN10 * signal_energy / noise_energy);
    CHECK(snr_db > 18.0f);
  }

  free(demuxed_signals);
  free(received_signal);
  free(muxed_signal);
  free(tactile_signals);
}

/* Same as TestRoundTrip but with odd channels equal to zero. This checks for
 * interference from pilot signals and neighboring channels.
 */
void TestZeroOddChannelsRoundTrip() {
  puts("TestZeroOddChannelsRoundTrip");
  const int kNumFrames = 1000;
  float* tactile_signals = MakeTactileTestSignals(kNumFrames);

  /* Set odd channels to zero. */
  float* dest = tactile_signals;
  int i;
  for (i = 0; i < kNumFrames; ++i, dest += kMuxChannels) {
    int c;
    for (c = 1; c < kMuxChannels; c += 2) {
      dest[c] = 0.0f;
    }
  }

  /* Multiplex the `tactile_signals` into `muxed_signal`. */
  int num_muxed_samples;
  float* muxed_signal =
      RunMuxer(tactile_signals, kNumFrames, &num_muxed_samples);
  /* Demultiplex `muxed_signal` to `demuxed_signals`. */
  int num_demuxed_frames;
  float* demuxed_signals =
      RunDemuxer(muxed_signal, num_muxed_samples, &num_demuxed_frames);

  /* Odd channels of the demuxed output should ideally be zero. So any values in
   * there are noise, leaked in from pilot signals and other channels. Check
   * noise standard deviation in the odd channels.
   */
  int c;
  for (c = 1; c < kMuxChannels; c += 1) {
    const float* expected = tactile_signals + c;
    const float* actual = demuxed_signals + c;
    float noise_energy = 0.0f;

    int i;
    for (i = 0; i < num_demuxed_frames; ++i) {
      const float error = *expected - *actual;
      noise_energy += error * error;
      expected += kMuxChannels;
      actual += kMuxChannels;
    }

    const float noise_stddev = sqrt(noise_energy / num_demuxed_frames);
    CHECK(noise_stddev < 0.12f);
  }

  free(demuxed_signals);
  free(muxed_signal);
  free(tactile_signals);
}

void TestMuxerStreaming() {
  puts("TestMuxerStreaming");
  const int kNumFrames = 250;
  float* tactile_signals = MakeTactileTestSignals(kNumFrames);

  /* Multiplex the `tactile_signals` into `muxed_nonstreaming`. */
  int num_nonstreaming;
  float* muxed_nonstreaming =
      RunMuxer(tactile_signals, kNumFrames, &num_nonstreaming);

  float* muxed_streaming = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kNumFrames * kMuxRateFactor));
  Muxer* muxer = (Muxer*)CHECK_NOTNULL(MuxerMake());

  /* Run the muxer on tactile_signals 10 times, processing in randomly-sized
   * blocks of 0 to 20 frames at a time.
   */
  int trial;
  for (trial = 0; trial < 10; ++trial) {
    MuxerReset(muxer);
    int num_streaming = 0;

    int start = 0;
    while (start < kNumFrames) {
      int block_size = (int)(20 * RandUniform());
      if (block_size > kNumFrames - start) {
        block_size = kNumFrames - start;
      }

      const int expected_output_size = MuxerNextOutputSize(muxer, block_size);
      CHECK(num_streaming + expected_output_size <= num_nonstreaming);
      const int actual_output_size =
          MuxerProcessSamples(muxer, tactile_signals + kMuxChannels * start,
                              block_size, muxed_streaming + num_streaming);
      CHECK(expected_output_size == actual_output_size);
      num_streaming += actual_output_size;
      start += block_size;
    }

    /* The streaming and nonstreaming muxed outputs should match. */
    CHECK(num_streaming == num_nonstreaming);
    int i;
    for (i = 0; i < num_streaming; ++i) {
      CHECK(fabs(muxed_streaming[i] - muxed_nonstreaming[i]) < 1e-6f);
    }
  }

  MuxerFree(muxer);
  free(muxed_streaming);
  free(muxed_nonstreaming);
  free(tactile_signals);
}

void TestDemuxerStreaming() {
  puts("TestDemuxerStreaming");
  const int kNumFrames = 250;
  const int kNumMuxedSamples = kNumFrames * kMuxRateFactor;
  float* muxed_signal =
      (float*)CHECK_NOTNULL(malloc(kNumMuxedSamples * sizeof(float)));
  int i;
  for (i = 0; i < kNumMuxedSamples; ++i) {
    muxed_signal[i] = 1.9f * (RandUniform() - 0.5f);
  }

  /* Demultiplex the `muxed_signal` into `demuxed_nonstreaming`. */
  int num_nonstreaming;
  float* demuxed_nonstreaming =
      RunDemuxer(muxed_signal, kNumMuxedSamples, &num_nonstreaming);

  float* demuxed_streaming = (float*)CHECK_NOTNULL(
      malloc(kNumFrames * kMuxChannels * sizeof(float)));

  /* Run the demuxer on muxed_signal 10 times, processing in randomly-sized
   * blocks of 0 to 20 * kMuxRateFactor samples at a time.
   */
  int trial;
  for (trial = 0; trial < 10; ++trial) {
    Demuxer demuxer;
    DemuxerInit(&demuxer);
    int num_streaming = 0;

    int start = 0;
    while (start < kNumMuxedSamples) {
      int block_size = (int)(20 * RandUniform()) * kMuxRateFactor;
      if (block_size > kNumMuxedSamples - start) {
        block_size = kNumMuxedSamples - start;
      }

      DemuxerProcessSamples(&demuxer, muxed_signal + start, block_size,
                            demuxed_streaming + kMuxChannels * num_streaming);
      num_streaming += block_size / kMuxRateFactor;
      start += block_size;
    }

    /* The streaming and nonstreaming demuxed outputs should match. */
    CHECK(num_streaming == num_nonstreaming);
    for (i = 0; i < kMuxChannels * num_streaming; ++i) {
      CHECK(fabs(demuxed_streaming[i] - demuxed_nonstreaming[i]) < 1e-6f);
    }
  }

  free(demuxed_streaming);
  free(demuxed_nonstreaming);
  free(muxed_signal);
}

int main(int argc, char** argv) {
  srand(0);
  TestRoundTrip(0.0f, 0.0f);  /* Clean round trip without distortions. */

  TestRoundTrip(4.8f, 0.0f);
  TestRoundTrip(-0.3f, 0.0f);
  TestRoundTrip(4.8f, 0.005f);
  TestRoundTrip(7.3f, 0.005f);
  TestRoundTrip(-4.8f, 0.005f);

  TestZeroOddChannelsRoundTrip();
  TestMuxerStreaming();
  TestDemuxerStreaming();

  puts("PASS");
  return EXIT_SUCCESS;
}
