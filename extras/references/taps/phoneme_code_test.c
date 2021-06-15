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
 */

#include "extras/references/taps/phoneme_code.h"

#include <math.h>
#include <string.h>

#include "src/dsp/logging.h"

#define kNumChannels 24

/* Test phoneme name lookup. */
static void TestPhonemeCodeByName() {
  puts("TestPhonemeCodeByName");
  const PhonemeCode* code = PhonemeCodeByName("B");
  CHECK(code != NULL && !strcmp(code->phoneme, "B"));

  /* Lookup is case insensitive. */
  code = PhonemeCodeByName("b");
  CHECK(code != NULL && !strcmp(code->phoneme, "B"));

  /* Lookup ignores chars after first non-alphanumeric char. */
  code = PhonemeCodeByName("b,er,d");
  CHECK(code != NULL && !strcmp(code->phoneme, "B"));

  /* Each codebook entry can be found. */
  int i;
  for (i = 0; i < kPhonemeCodebookSize; ++i) {
    const char* name = kPhonemeCodebook[i].phoneme;
    code = PhonemeCodeByName(name);
    CHECK(code != NULL && !strcmp(code->phoneme, name));
  }

  /* Returns NULL for nonmatching phoneme names. */
  CHECK(PhonemeCodeByName("") == NULL);
  CHECK(PhonemeCodeByName("XX") == NULL);
  CHECK(PhonemeCodeByName("TOOLONG") == NULL);
}

/* Check code signal max amplitude. */
static void TestMaxAmplitude() {
  puts("TestMaxAmplitude");
  const float kMaxDuration = 0.4f; /* Codes are at most 0.4s duration. */
  const float kSampleRateHz = 44100.0f;
  const int kNumSamples = (int)(kSampleRateHz * kMaxDuration + 0.5f);
  float frame[kNumChannels];
  float max_amplitude = 0.0f;

  int i;
  for (i = 0; i < kPhonemeCodebookSize; ++i) {
    int n;
    for (n = 0; n < kNumSamples; ++n) {
      const float t = n / kSampleRateHz;

      /* Generate a frame of code i. */
      kPhonemeCodebook[i].fun(t, frame);

      int c;
      for (c = 0; c < kNumChannels; ++c) {
        float amplitude = fabs(frame[c]);
        if (amplitude > max_amplitude) {
          max_amplitude = amplitude;
        }
      }
    }
  }

  const float kScale = 0.0765f;
  const float kExpected = 2 * kScale;
  CHECK(0.995f * kExpected <= max_amplitude)
  CHECK(max_amplitude <= kExpected);
}

/* Check that codes are some min L2 distance apart from one another. */
static void TestCodeSignalsAreDifferent() {
  puts("TestCodeSignalsAreDifferent");
  /* Table for accumulating distances between each pair of codes.
   * Only the lower triangle is used.
   */
  float* distance = (float*)CHECK_NOTNULL(
      malloc(kPhonemeCodebookSize * kPhonemeCodebookSize * sizeof(float)));
  /* Buffer for code frames. */
  float(*codes)[kNumChannels] = (float(*)[kNumChannels])CHECK_NOTNULL(
      malloc(kPhonemeCodebookSize * kNumChannels * sizeof(float)));

  int i;
  int j;
  for (i = 0; i < kPhonemeCodebookSize; ++i) {
    for (j = 0; j < kPhonemeCodebookSize; ++j) {
      distance[i + kPhonemeCodebookSize * j] = 0.0f;
    }
  }

  const float kMaxDuration = 0.4f; /* Codes are at most 0.4s duration. */
  const float kSampleRateHz = 44100.0f;
  const int kNumSamples = (int)(kSampleRateHz * kMaxDuration + 0.5f);
  int n;
  for (n = 0; n < kNumSamples; ++n) {
    const float t = n / kSampleRateHz;

    for (i = 0; i < kPhonemeCodebookSize; ++i) {
      /* Generate a frame of code i. */
      kPhonemeCodebook[i].fun(t, codes[i]);

      for (j = 0; j < i; ++j) {
        /* Accumulate L2 squared distance between code i and code j. */
        int c;
        for (c = 0; c < kNumChannels; ++c) {
          const float diff = codes[i][c] - codes[j][c];
          distance[i + kPhonemeCodebookSize * j] += diff * diff;
        }
      }
    }
  }

  /* Find the min distance. */
  float min_distance = distance[1 + kPhonemeCodebookSize * 0];
  for (i = 0; i < kPhonemeCodebookSize; ++i) {
    for (j = 0; j < i; ++j) {
      if (distance[i + kPhonemeCodebookSize * j] < min_distance) {
        min_distance = distance[i + kPhonemeCodebookSize * j];
      }
    }
  }

  CHECK(min_distance >= 8.0);

  free(codes);
  free(distance);
}

/* Test a few cases with PhonemeStringIsValid(). */
static void TestPhonemeStringIsValid() {
  puts("TestPhonemeStringIsValid");
  CHECK(PhonemeStringIsValid("OE"));
  CHECK(PhonemeStringIsValid("b,er,d"));
  CHECK(!PhonemeStringIsValid("b,er,XX,d"));
}

static float Max(float x, float y) { return (x >= y) ? x : y; }

/* Test GeneratePhonemeSignal() on "B,ER,D" with "ER" emphasized and on a
 * few different phoneme spacings.
 */
static void TestGeneratePhonemeSignal(float spacing) {
  puts("TestGeneratePhonemeSignal");
  const float kSampleRateHz = 44100.0f;
  const char* kEmphasizedPhoneme = "ER";
  const float kEmphasisGain = 1.5f;
  int num_frames;
  float* samples = CHECK_NOTNULL(
      GeneratePhonemeSignal("B,ER,D", spacing, kEmphasizedPhoneme,
                            kEmphasisGain, kSampleRateHz, &num_frames));

  const PhonemeCode* code_b = PhonemeCodeByName("B");
  const PhonemeCode* code_er = PhonemeCodeByName("ER");
  const PhonemeCode* code_d = PhonemeCodeByName("D");

  /* Compute when each phoneme code signal should start and end. */
  const float start_time_b = 0.0f;
  const float end_time_b = start_time_b + code_b->duration;
  const float start_time_er = Max(end_time_b + spacing, 0.0f);
  const float end_time_er = start_time_er + code_er->duration;
  const float start_time_d = Max(end_time_er + spacing, 0.0f);
  const float end_time_d = start_time_d + code_d->duration;

  const float total_time = Max(Max(end_time_b, end_time_er), end_time_d);
  CHECK(fabs(num_frames - kSampleRateHz * total_time) < 1.0f);

  int i;
  for (i = 0; i < num_frames; ++i) {
    const float t = i / kSampleRateHz;

    /* Generate code signal for each phoneme, delayed by their start times. */
    float frame_b[kNumChannels];
    code_b->fun(t - start_time_b, frame_b);
    float frame_er[kNumChannels];
    code_er->fun(t - start_time_er, frame_er);
    float frame_d[kNumChannels];
    code_d->fun(t - start_time_d, frame_d);

    int c;
    for (c = 0; c < kNumChannels; ++c) {
      const float expected =
          frame_b[c]                    /* B phoneme. */
          + kEmphasisGain * frame_er[c] /* ER phoneme, with emphasis. */
          + frame_d[c];                 /* D phoneme. */
      CHECK(fabs(samples[kNumChannels * i + c] - expected) <= 2e-5);
    }
  }

  free(samples);
}

int main(int argc, char** argv) {
  TestPhonemeCodeByName();
  TestMaxAmplitude();
  TestCodeSignalsAreDifferent();
  TestPhonemeStringIsValid();
  TestGeneratePhonemeSignal(0.0f);
  TestGeneratePhonemeSignal(0.05f);
  TestGeneratePhonemeSignal(-0.1f);
  TestGeneratePhonemeSignal(-0.3f);

  puts("PASS");
  return EXIT_SUCCESS;
}
