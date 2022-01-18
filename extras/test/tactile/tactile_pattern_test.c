/* Copyright 2021 Google LLC
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

#include "src/tactile/tactile_pattern.h"

#include <string.h>

#include "extras/tools/util.h"
#include "src/dsp/convert_sample.h"
#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"
#include "src/dsp/read_wav_file.h"
#include "src/dsp/write_wav_file.h"

/* The tactile sample rate used on the device is 16 MHz / 2^13 = 1953.125 Hz. */
const int kSampleRateHz = 1953;

/* Writes samples in [-1, 1] as WAV file `{dir}/tactile_pattern_{name}.wav`. */
static int WriteWav(const char* dir, const char* name,
                    const float* samples, int num_frames, int num_channels) {
  /* Convert float samples in [-1, 1] to int16 samples. */
  const int num_samples = num_frames * num_channels;
  int16_t* samples_int16 =
      (int16_t*)CHECK_NOTNULL(malloc(num_samples * sizeof(int16_t)));
  ConvertSampleArrayFloatToInt16(samples, num_samples, samples_int16);

  /* Write WAV file. */
  char wav_file[1024];
  sprintf(wav_file, "%s/tactile_pattern_%s.wav", dir, name);
  const int success = WriteWavFile(wav_file, samples_int16, num_samples,
                                   kSampleRateHz, num_channels);
  free(samples_int16);

  if (success) {
    printf("Wrote %s\n", wav_file);
  } else {
    fprintf(stderr, "Failed to write WAV file to %s\n", wav_file);
  }
  return success;
}

/* Checks that the max difference between successive samples is `tolerace`. */
static int /*bool*/ CheckContinuous(const float* samples,
    int num_frames, int num_channels, float tolerance) {
  int c;
  for (c = 0; c < num_channels; ++c) {
    float max_diff = 0.0f;
    float prev = 0.0f;

    int i;
    for (i = 0; i < num_frames; ++i) {
      const float cur = *(samples + i * num_channels + c);
      const float diff = fabs(cur - prev);
      if (diff > max_diff) {
        max_diff = diff;
      }
      prev = cur;
    }

    if (max_diff > tolerance) {
      fprintf(stderr, "Channel %d: max_diff = %g exceeds tolerance = %g.\n",
              c, max_diff, tolerance);
      return 0;
    }
  }

  return 1;
}

/* Computes the average power in channel c over time [t_start, t_end]. */
static float AveragePower(const float* samples,
    int num_frames, int num_channels, int c, float t_start, float t_end) {
  const int i_start = (int)(t_start * kSampleRateHz + 0.5f);
  const int i_end = (int)(t_end * kSampleRateHz + 0.5f);
  double energy = 0.0;

  int i;
  for (i = i_start; i <= i_end && i < num_frames; ++i) {
    double cur = *(samples + i * num_channels + c);
    energy += cur * cur;
  }

  return energy / (i_end - i_start + 1);
}

/* Checks that channel c over time [t_start, t_end] has the expected sinusoid
 * amplitude and frequency. This function assumes, but does not verify, that the
 * waveform is a sinusoid.
 */
static int /*bool*/ CheckSinusoid(const float* samples,
    int num_frames, int num_channels, int c, float t_start, float t_end,
    float expected_amplitude, float expected_frequency_hz) {
  const int i_start = (int)(t_start * kSampleRateHz + 0.5f);
  const int i_end = (int)(t_end * kSampleRateHz + 0.5f);
  double teager = 0.0;
  double energy = 0.0;
  double prev = *(samples + (i_start - 1) * num_channels + c);
  double cur = *(samples + i_start * num_channels + c);
  int i;
  for (i = i_start; i < i_end && i < num_frames - 1; ++i) {
    double next = *(samples + (i + 1) * num_channels + c);
    teager += cur * cur - prev * next;
    energy += cur * cur;
    prev = cur;
    cur = next;
  }
  /* Estimate sinusoid amplitude from the RMS value. */
  const float actual_amplitude = sqrt(2 * energy / (i_end - i_start));
  /* Estimate sinusoid frequency from the Teager energy. */
  const float actual_frequency_hz =
      (kSampleRateHz / (2 * M_PI)) * asin(sqrt(teager / (2 * energy)));

  if (!(fabs(actual_amplitude - expected_amplitude) < 0.03f * expected_amplitude)) {
    fprintf(stderr,
            "Error: actual amplitude %g mismatches expected %g.\n",
            actual_amplitude, expected_amplitude);
    return 0;
  }
  if (!(fabs(actual_frequency_hz - expected_frequency_hz) < 0.03f * expected_frequency_hz)) {
    fprintf(stderr,
            "Error: actual frequency %g Hz mismatches expected %g Hz.\n",
            actual_frequency_hz, expected_frequency_hz);
    return 0;
  }
  return 1;
}

/* Synthesizes `ex_pattern` and returns a pointer to the generated samples. The
 * caller must free the samples.
 */
static float* SynthesizeExPattern(
    const uint8_t* ex_pattern, int num_frames, int num_channels) {
  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz, num_channels);
  TactilePatternStartEx(&p, ex_pattern);
  CHECK(TactilePatternIsActive(&p));
  float* samples =
      (float*)CHECK_NOTNULL(malloc(num_frames * num_channels * sizeof(float)));
  CHECK(!TactilePatternSynthesize(&p, num_frames, samples));
  CHECK(!TactilePatternIsActive(&p));
  return samples;
}

/* Test that by default silence is produced as expected. */
static void TestSilence(void) {
  puts("TestSilence");
  const int num_channels = 2;
  float output[20];
  /* Fill with nonzero values first to test that it gets overwritten. */
  int i;
  for (i = 0; i < 20; ++i) {
    output[i] = -1.0f;
  }

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz, num_channels);
  CHECK(!TactilePatternSynthesize(&p, 10, output));

  for (i = 0; i < 20; ++i) {
    CHECK(output[i] == 0.0f);
  }
}

/* Test on a pattern like "tone, pause, tone, pause, tone". */
static void TestThreePips(void) {
  puts("TestThreePips");
  const int num_channels = 3;
  const uint8_t ex_pattern[] = {
    kTactilePatternOpSetWaveform + 1, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetGain + 1, 0xff, /* Set gain to 1.0. */
    TACTILE_PATTERN_OP_PLAY_MS(500), /* Play for 500 ms. */
    TACTILE_PATTERN_OP_PLAY_MS(500), /* Rest for 500 ms. */
    kTactilePatternOpSetWaveform + 1, kTactilePatternWaveformSin25Hz,
    TACTILE_PATTERN_OP_PLAY_MS(500), /* Play for 500 ms. */
    TACTILE_PATTERN_OP_PLAY_MS(500), /* Rest for 500 ms. */
    kTactilePatternOpSetWaveform + 1, kTactilePatternWaveformSin25Hz,
    TACTILE_PATTERN_OP_PLAY_MS(500), /* Play for 500 ms. */
    kTactilePatternOpEnd,
  };
  const int num_frames = 3 * kSampleRateHz;
  float* samples = SynthesizeExPattern(ex_pattern, num_frames, num_channels);

  /* Check channel 1 output over each of the five "Play" intervals. The waveform
   * should alternate between a full-scale sinusoid and silence.
   */
  int i;
  for (i = 0; i < 5; ++i) {
    const float t_start = i * 0.5f + 0.02f;
    const float t_end = (i + 1) * 0.5f;
    if (i % 2 == 0) {
      CHECK(CheckSinusoid(
        samples, num_frames, num_channels, 1, t_start, t_end, 1.0f, 25.0f));
    } else {
      /* Waveform should be silent. */
      const float power =
          AveragePower(samples, num_frames, num_channels, 1, t_start, t_end);
      CHECK(power < 1e-6f);
    }
  }

  /* Channels 0 and 2 are silent. */
  CHECK(AveragePower(samples, num_frames, num_channels, 0, 0.0f, 2.5f) == 0.0f);
  CHECK(AveragePower(samples, num_frames, num_channels, 2, 0.0f, 2.5f) == 0.0f);
  /* Waveforms are continuous. */
  CHECK(CheckContinuous(samples, num_frames, num_channels, 0.15f));

  free(samples);
}

/* Test a pattern with different tone frequecies. */
static void TestToneFrequencies(void) {
  puts("TestToneFrequencies");
  const int num_channels = 1;
  const uint8_t ex_pattern[] = {
    kTactilePatternOpSetGain + 0, 0x80, /* Set gain 0.5. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin350Hz,
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin60Hz,
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpEnd,
  };
  const int num_frames = kSampleRateHz;
  float* samples = SynthesizeExPattern(ex_pattern, num_frames, num_channels);

  /* Check channel 0 output over each of the three "Play" intervals.  */
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.025f, 0.1f, 0.5f, 25.0f));
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.12f, 0.2f, 0.5f, 350.0f));
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.22f, 0.3f, 0.5f, 60.0f));

  free(samples);
}

/* Test a pattern where the gain changes. */
static void TestChangingGain(void) {
  puts("TestChangingGain");
  const int num_channels = 1;
  const uint8_t ex_pattern[] = {
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetGain + 0, 0x1a, /* Gain 0.1. */
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetGain + 0, 0xb2, /* Gain 0.7. */
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetGain + 0, 0x4c, /* Gain 0.3. */
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpEnd,
  };
  const int num_frames = kSampleRateHz;
  float* samples = SynthesizeExPattern(ex_pattern, num_frames, num_channels);

  /* Check channel 0 output over each of the three "Play" intervals.  */
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.02f, 0.1f, 0.1f, 25.0f));
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.12f, 0.2f, 0.7f, 25.0f));
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.22f, 0.3f, 0.3f, 25.0f));
  /* Waveforms are continuous. */
  CHECK(CheckContinuous(samples, num_frames, num_channels, 0.15f));

  free(samples);
}

/* Test the "Move" op. */
static void TestMovementPattern(void) {
  puts("TestMovementPattern");
  const int num_channels = 3;
  const uint8_t ex_pattern[] = {
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetGain + 1, 0xff, /* Gain 1.0. */
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpMove, 0x01, /* Move from channel 0 to 1. */
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpMove, 0x12, /* Move from channel 1 to 2. */
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpEnd,
  };
  const int num_frames = kSampleRateHz;
  float* samples = SynthesizeExPattern(ex_pattern, num_frames, num_channels);

  int i;
  for (i = 0; i < 3; ++i) {
    const float t_start = i * 0.1f + 0.02f;
    const float t_end = (i + 1) * 0.1f;
    int c;
    for (c = 0; c < num_channels; ++c) {
      if (c == i) {
        CHECK(CheckSinusoid(
          samples, num_frames, num_channels, c, t_start, t_end, 0.15f, 25.0f));
      } else {
        /* Waveform should be silent. */
        CHECK(AveragePower(samples, num_frames, num_channels, c, t_start,
                           t_end) < 1e-6f);
      }
    }
  }
  /* Waveforms are continuous. */
  CHECK(CheckContinuous(samples, num_frames, num_channels, 0.15f));

  free(samples);
}

/* Test where three channels simultaneously play different waveforms. */
static void TestSimultaneousPips(void) {
  puts("TestSimultaneousPips");
  const int num_channels = 3;
  const uint8_t ex_pattern[] = {
    kTactilePatternOpSetGain + 0, 0x1a, /* Gain 0.1. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetWaveform + 2, kTactilePatternWaveformSin50Hz,
    TACTILE_PATTERN_OP_PLAY_MS(100), /* Play for 100 ms. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetWaveform + 1, kTactilePatternWaveformChirp,
    kTactilePatternOpSetWaveform + 2, kTactilePatternWaveformSin50Hz,
    kTactilePatternOpSetGain + 1, 0x4c, /* Gain 0.3. */
    TACTILE_PATTERN_OP_PLAY_MS(140), /* Play for 100 ms. */
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin25Hz,
    kTactilePatternOpSetWaveform + 1, kTactilePatternWaveformChirp,
    TACTILE_PATTERN_OP_PLAY_MS(80), /* Play for 100 ms. */
    kTactilePatternOpEnd,
  };
  const int num_frames = kSampleRateHz;
  float* samples = SynthesizeExPattern(ex_pattern, num_frames, num_channels);

  /* Channel 0 plays a 25 Hz tone with amplitude 0.1 throughout [0.02, 0.32]. */
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 0, 0.02f, 0.32f, 0.1f, 25.0f));
  /* Channel 1 is silent over [0, 0.1], then plays a chirp until 0.32. */
  CHECK(fabs(AveragePower(samples, num_frames, num_channels, 1, 0.0f, 0.1f))
        < 1e-6f);
  CHECK(fabs(AveragePower(samples, num_frames, num_channels, 1, 0.12f, 0.32f)
        - 0.045f) < 2e-3f);
  /* Channel 2 plays a 50 Hz tone with amplitude 0.15 over [0.02, 0.24]. */
  CHECK(CheckSinusoid(
    samples, num_frames, num_channels, 2, 0.02f, 0.24f, 0.15f, 50.0f));
  CHECK(AveragePower(samples, num_frames, num_channels, 2, 0.26f, 0.32f)
        == 0.0f);

  free(samples);
}

/* Test that `TactilePatternSynthesize()` streams correctly. */
static void TestStreaming(void) {
  puts("TestStreaming");
  const int num_channels = 3;
  const uint8_t ex_pattern[] = {
    kTactilePatternOpSetWaveform + 1, kTactilePatternWaveformSin90Hz,
    kTactilePatternOpSetWaveform + 0, kTactilePatternWaveformSin30Hz,
    kTactilePatternOpSetGain + 1, 0xb2, /* Gain 0.7. */
    kTactilePatternOpSetGain + 0, 0x1a, /* Gain 0.1. */
    TACTILE_PATTERN_OP_PLAY_MS(80),
    kTactilePatternOpMove, 0x10,
    TACTILE_PATTERN_OP_PLAY_MS(80),
    TACTILE_PATTERN_OP_PLAY_MS(80),
    kTactilePatternOpSetAllWaveform, kTactilePatternWaveformChirp,
    kTactilePatternOpSetGain + 0, 0x80, /* Gain 0.5. */
    TACTILE_PATTERN_OP_PLAY_MS(80),
    kTactilePatternOpMove, 0x02,
    TACTILE_PATTERN_OP_PLAY_MS(140),
    kTactilePatternOpEnd,
  };
  const int num_frames = kSampleRateHz;
  float* nonstreaming =
      SynthesizeExPattern(ex_pattern, num_frames, num_channels);

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz, num_channels);
  CHECK(!TactilePatternIsActive(&p));
  TactilePatternStartEx(&p, ex_pattern);
  CHECK(TactilePatternIsActive(&p));

  float streaming[25 * 3];
  int start;
  for (start = 0; start < num_frames;) {
    int block_size = RandomInt(25);
    if (num_frames - start < block_size) {
      block_size = num_frames - start;
    }

    TactilePatternSynthesize(&p, block_size, streaming);
    int i;
    for (i = 0; i < num_channels * block_size; ++i) {
      CHECK(fabs(streaming[i] - nonstreaming[num_channels * start + i]) < 1e-6f);
    }

    start += block_size;
  }

  CHECK(!TactilePatternIsActive(&p));
  free(nonstreaming);
}

/* Test `TactilePatternTranslateSimplePattern()`. */
void TestTranslateSimplePattern(void) {
  puts("TestTranslateSimplePattern");
  const char* simple_pattern = "04C-/";

  uint8_t output[14];
  CHECK(TactilePatternTranslateSimplePattern(
    simple_pattern, output, sizeof(output)));

  CHECK(output[0] == kTactilePatternOpSetAllWaveform);
  CHECK(output[1] == kTactilePatternWaveformSin25Hz);
  CHECK(output[2] == TACTILE_PATTERN_OP_PLAY_MS(80));
  CHECK(output[3] == kTactilePatternOpSetAllWaveform);
  CHECK(output[4] == kTactilePatternWaveformSin50Hz);
  CHECK(output[5] == TACTILE_PATTERN_OP_PLAY_MS(80));
  CHECK(output[6] == kTactilePatternOpSetAllWaveform);
  CHECK(output[7] == kTactilePatternWaveformSin200Hz);
  CHECK(output[8] == TACTILE_PATTERN_OP_PLAY_MS(80));
  CHECK(output[9] == TACTILE_PATTERN_OP_PLAY_MS(40));
  CHECK(output[10] == kTactilePatternOpSetAllWaveform);
  CHECK(output[11] == kTactilePatternWaveformChirp);
  CHECK(output[12] == TACTILE_PATTERN_OP_PLAY_MS(300));
  CHECK(output[13] == kTactilePatternOpEnd);
}

/* Compare simple pattern with golden WAV file. */
static void TestCompareSimplePatternWithGolden(const char* name,
                                               const char* simple_pattern) {
  printf("TestCompareSimplePatternWithGolden(%s)\n", name);
  /* Read golden from WAV file. */
  char wav_file[1024];
  sprintf(wav_file,
          "extras/test/testdata/"
          "tactile_pattern_%s.wav",
          name);
  size_t num_frames;
  int num_channels;
  int sample_rate_hz;
  int16_t* golden = (int16_t*)CHECK_NOTNULL(
      Read16BitWavFile(wav_file, &num_frames, &num_channels, &sample_rate_hz));

  TactilePattern p;
  TactilePatternInit(&p, sample_rate_hz, num_channels);
  TactilePatternStart(&p, simple_pattern);
  CHECK(TactilePatternIsActive(&p));

  float* actual =
      (float*)CHECK_NOTNULL(malloc(num_frames * num_channels * sizeof(float)));
  CHECK(!TactilePatternSynthesize(&p, num_frames, actual));

  int i;
  for (i = 0; i < (int)num_frames; ++i) {
    const float expected =  ConvertSampleInt16ToFloat(golden[i]);
    CHECK(fabs(expected - actual[i]) < 1e-4f);
  }

  free(actual);
  free(golden);
}

/* Test `TactilePatternStartCalibrationTones()`. */
static void TestCalibrationTones(void) {
  puts("TestCalibrationTones");
  const int kNumChannels = 5;
  const int kNumFrames = 2 * kSampleRateHz;
  float* samples =
      (float*)CHECK_NOTNULL(malloc(kNumFrames * kNumChannels * sizeof(float)));

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz, kNumChannels);
  /* Play calibration tones on channels 3 and 1. */
  TactilePatternStartCalibrationTones(&p, 3, 1);
  CHECK(TactilePatternIsActive(&p));
  CHECK(!TactilePatternSynthesize(&p, kNumFrames, samples));

  /* First tone plays on channel 3. */
  CHECK(CheckSinusoid(samples, kNumFrames, kNumChannels, 3,
                      0.02f, 0.24f, 0.15f, 125.0f));
  CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 1, 0.0f, 0.34f))
        < 1e-6f);
  CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 3, 0.26f, 2.0f))
        < 1e-6f);
  /* Second tone plays on channel 1. */
  CHECK(CheckSinusoid(samples, kNumFrames, kNumChannels, 1,
                      0.36f, 0.58f, 0.15f, 125.0f));

  int c;
  for (c = 0; c < kNumChannels; ++c) { /* Other channels are silent. */
    if (c != 3 && c != 1) {
      CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 0, 0.0f,
                              2.0f)) < 1e-6f);
    }
  }

  free(samples);
}

/* Test `TactilePatternStartCalibrationTonesThresholds()`. */
static void TestCalibrationTonesThresholds(void) {
  puts("TestCalibrationTonesThresholds");
  const int kNumChannels = 5;
  const int kNumFrames = 2 * kSampleRateHz;
  float* samples =
      (float*)CHECK_NOTNULL(malloc(kNumFrames * kNumChannels * sizeof(float)));

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz, kNumChannels);
  /* Play calibration tones on channels 3 and 1. */
  TactilePatternStartCalibrationTonesThresholds(&p, 3, 1, 0.1f);
  CHECK(TactilePatternIsActive(&p));
  CHECK(!TactilePatternSynthesize(&p, kNumFrames, samples));

  /* First tone plays on channel 3. */
  CHECK(CheckSinusoid(samples, kNumFrames, kNumChannels, 3,
                      0.02f, 0.4f, 0.1f, 125.0f));
  CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 1, 0.0f, 0.7f)) <
        1e-6f);
  CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 3, 0.42f, 1.4f)) <
        1e-6f);
  /* Second tone plays on channel 1. */
  CHECK(CheckSinusoid(samples, kNumFrames, kNumChannels, 1,
                      0.72f, 1.1f, 0.1f, 125.0f));
  CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 1, 1.12f, 2.0f)) <
        1e-6f);
  /* Third tone plays on channel 3. */
  CHECK(CheckSinusoid(samples, kNumFrames, kNumChannels, 3,
                      1.42f, 1.8f, 0.1f, 125.0f));

  int c;
  for (c = 0; c < kNumChannels; ++c) { /* Other channels are silent. */
    if (c != 3 && c != 1) {
      CHECK(fabs(AveragePower(samples, kNumFrames, kNumChannels, 0, 0.0f,
                              2.0f)) < 1e-6f);
    }
  }

  free(samples);
}

/* Synthesizes a simple pattern and writes it as a WAV file. */
static int WritePattern(const char* dir, const char* name,
                        const char* simple_pattern) {
  const int num_frames = kSampleRateHz;
  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz, 1);
  CHECK(TactilePatternStart(&p, simple_pattern));
  CHECK(TactilePatternIsActive(&p));
  float* samples = (float*)CHECK_NOTNULL(malloc(num_frames * sizeof(float)));
  CHECK(!TactilePatternSynthesize(&p, num_frames, samples));
  return WriteWav(dir, name, samples, num_frames, 1);
}

/* Writes patterns as WAV files. Called if program runs with --write_goldens. */
static int WriteGoldens(const char* dir) {
  return WritePattern(dir, "connect", kTactilePatternConnect) &&
         WritePattern(dir, "disconnect", kTactilePatternDisconnect) &&
         WritePattern(dir, "confirm", kTactilePatternConfirm);
}

int main(int argc, char** argv) {
  srand(0);
  if (argc == 2 && StartsWith(argv[1], "--write_goldens=")) {
    const char* output_dir = strchr(argv[1], '=') + 1;
    return WriteGoldens(output_dir) ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  TestSilence();
  TestThreePips();
  TestToneFrequencies();
  TestChangingGain();
  TestMovementPattern();
  TestSimultaneousPips();
  TestStreaming();

  TestTranslateSimplePattern();
  TestCompareSimplePatternWithGolden("connect", kTactilePatternConnect);
  TestCompareSimplePatternWithGolden("disconnect", kTactilePatternDisconnect);
  TestCompareSimplePatternWithGolden("confirm", kTactilePatternConfirm);

  TestCalibrationTones();
  TestCalibrationTonesThresholds();

  puts("PASS");
  return EXIT_SUCCESS;
}
