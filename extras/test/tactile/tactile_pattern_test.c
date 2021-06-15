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
#include "src/dsp/read_wav_file.h"
#include "src/dsp/write_wav_file.h"

/* The tactile sample rate used on the device is 16 MHz / 2^13 = 1953.125 Hz. */
const int kSampleRateHz = 1953;

static float* SynthesizePattern(const char* pattern, int num_frames) {
  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz);
  TactilePatternStart(&p, pattern);
  float* samples = (float*)CHECK_NOTNULL(malloc(num_frames * sizeof(float)));
  TactilePatternSynthesize(&p, num_frames, 1, samples);
  return samples;
}

/* Compare pattern with golden WAV file. */
static void TestCompareWithGolden(const char* name, const char* pattern) {
  printf("TestCompareWithGolden(%s)\n", name);
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
  CHECK(sample_rate_hz == kSampleRateHz);

  float* actual = SynthesizePattern(pattern, num_frames);

  int i;
  for (i = 0; i < num_frames; ++i) {
    const float expected =  ConvertSampleInt16ToFloat(golden[i]);
    CHECK(fabs(expected - actual[i]) < 1e-4f);
  }

  free(actual);
  free(golden);
}

/* Test that streaming agrees with computing all at once. */
static void TestStreaming(const char* name, const char* pattern) {
  printf("TestStreaming(%s)\n", name);
  const int num_frames = kSampleRateHz;
  float* nonstreaming = SynthesizePattern(pattern, num_frames);

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz);
  CHECK(!TactilePatternIsActive(&p));
  TactilePatternStart(&p, pattern);
  CHECK(TactilePatternIsActive(&p));

  float streaming[25];
  int start;
  for (start = 0; start < num_frames;) {
    int block_size = RandomInt(25);
    if (num_frames - start < block_size) {
      block_size = num_frames - start;
    }

    TactilePatternSynthesize(&p, block_size, 1, streaming);
    int i;
    for (i = 0; i < block_size; ++i) {
      CHECK(fabs(streaming[i] - nonstreaming[start + i]) < 1e-6f);
    }

    start += block_size;
  }

  CHECK(!TactilePatternIsActive(&p));
  free(nonstreaming);
}

/* Test that kTactilePatternSilence produces zeros as expected. */
static void TestSilence() {
  puts("TestSilence");
  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz);
  TactilePatternStart(&p, kTactilePatternSilence);

  float output[10];
  /* Fill with nonzero values first to test that it gets overwritten. */
  int i;
  for (i = 0; i < 10; ++i) {
    output[i] = -1.0f;
  }

  TactilePatternSynthesize(&p, 5, 2, output);
  for (i = 0; i < 10; ++i) {
    CHECK(output[i] == 0.0f);
  }
}

static void TestMultichannel() {
  puts("TestMultichannel");
  const int kNumChannels = 5;
  const int kNumFrames = 250;
  float* samples =
      (float*)CHECK_NOTNULL(malloc(kNumFrames * kNumChannels * sizeof(float)));

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz);
  TactilePatternStart(&p, "A");
  TactilePatternSynthesize(&p, kNumFrames, kNumChannels, samples);

  /* All channels have identical output. */
  int i;
  for (i = 0; i < kNumFrames; ++i) {
    const int offset = i * kNumChannels;
    int c;
    for (c = 1; c < kNumChannels; ++c) {
      CHECK(samples[offset + c] == samples[offset]);
    }
  }

  free(samples);
}

static void TestCalibrationTones() {
  puts("TestCalibrationTones");
  const int kNumChannels = 5;
  const int kNumFrames = 250;
  float* samples =
      (float*)CHECK_NOTNULL(malloc(kNumFrames * kNumChannels * sizeof(float)));

  TactilePattern p;
  TactilePatternInit(&p, kSampleRateHz);
  /* Play calibration tones on channels 3 and 1. */
  TactilePatternStartCalibrationTones(&p, 3, 1);
  CHECK(TactilePatternIsActive(&p));
  float time = 0.0f;

  while (TactilePatternSynthesize(&p, kNumFrames, kNumChannels, samples)) {
    time += (float)kNumFrames / kSampleRateHz;

    const float* src = samples;
    float energy[5] = {};
    int i;
    int c;
    for (i = 0; i < kNumFrames; ++i, src += kNumChannels) {
      for (c = 0; c < kNumChannels; ++c) {
        energy[c] += src[c] * src[c];
      }
    }

    if (time < 0.24f) { /* First tone plays on channel 3. */
      CHECK(energy[3] > 0.1f);
      CHECK(energy[1] == 0.0f);
    } else if (time > 0.34f) { /* Second tone plays on channel 1. */
      CHECK(energy[3] == 0.0f);
      CHECK(energy[1] > 0.1f);
    }

    for (c = 0; c < kNumChannels; ++c) { /* Other channels are silent. */
      if (c != 3 && c != 1) {
        CHECK(energy[c] == 0.0f);
      }
    }
  }

  free(samples);
}

/* Synthesize pattern and write it to WAV file. */
static int WritePattern(const char* output_dir, const char* name,
                        const char* pattern) {
  const int num_frames = kSampleRateHz;
  float* samples = SynthesizePattern(pattern, num_frames);

  /* Convert float samples in [-1, 1] to int16 samples. */
  int16_t* samples_int16 =
      (int16_t*)CHECK_NOTNULL(malloc(num_frames * sizeof(int16_t)));
  ConvertSampleArrayFloatToInt16(samples, num_frames, samples_int16);
  free(samples);

  /* Write WAV file. */
  char wav_file[1024];
  sprintf(wav_file, "%s/tactile_pattern_%s.wav", output_dir, name);
  const int success =
      WriteWavFile(wav_file, samples_int16, num_frames, kSampleRateHz, 1);
  free(samples_int16);

  if (success) {
    printf("Wrote %s\n", wav_file);
  } else {
    fprintf(stderr, "Failed to write WAV file to %s\n", wav_file);
  }
  return success;
}

/* Writes patterns as WAV files. Called if program runs with --write_goldens. */
static int WriteGoldens(const char* output_dir) {
  return WritePattern(output_dir, "connect", kTactilePatternConnect) &&
         WritePattern(output_dir, "disconnect", kTactilePatternDisconnect) &&
         WritePattern(output_dir, "confirm", kTactilePatternConfirm);
}

int main(int argc, char** argv) {
  srand(0);
  if (argc == 2 && StartsWith(argv[1], "--write_goldens=")) {
    const char* output_dir = strchr(argv[1], '=') + 1;
    return WriteGoldens(output_dir) ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  TestCompareWithGolden("connect", kTactilePatternConnect);
  TestCompareWithGolden("disconnect", kTactilePatternDisconnect);
  TestCompareWithGolden("confirm", kTactilePatternConfirm);

  TestStreaming("connect", kTactilePatternConnect);
  TestStreaming("disconnect", kTactilePatternDisconnect);
  TestStreaming("confirm", kTactilePatternConfirm);

  TestSilence();
  TestMultichannel();
  TestCalibrationTones();

  puts("PASS");
  return EXIT_SUCCESS;
}
