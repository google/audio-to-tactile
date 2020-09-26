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
 */

#include "phonetics/embed_vowel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/logging.h"
#include "dsp/read_wav_file.h"
#include "frontend/carl_frontend.h"

#define kBlockSize 64

/* Tests that target lookup functions work. */
void TestTargetLookup() {
  puts("TestTargetLookup");
  int i;
  for (i = 0; i < kEmbedVowelNumTargets; ++i) {
    /* Target should be closest to itself. */
    CHECK(EmbedVowelClosestTarget(kEmbedVowelTargets[i].coord) == i);
    /* Lookup by target name. */
    CHECK(EmbedVowelTargetByName(kEmbedVowelTargets[i].name) == i);
  }

  /* Searching for a nonexistent target name. */
  CHECK(EmbedVowelTargetByName("nonsense") == -1);
}

/* Runs CarlFrontend + EmbedVowel on a short WAV recording of a pure phone, and
 * checks that the output coordinate is usually near the correct target.
 */
void TestPhone(const char* phone) {
  printf("TestPhone(\"%s\")\n", phone);
  char wav_file[1024];
  sprintf(wav_file, "testdata/phone_%s.wav", phone);

  size_t num_samples;
  int num_channels;
  int sample_rate_hz;
  int16_t* input_int16 = (int16_t*)CHECK_NOTNULL(Read16BitWavFile(
      wav_file, &num_samples, &num_channels, &sample_rate_hz));
  float* frame = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * kEmbedVowelNumChannels));
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 16000);

  CarlFrontendParams frontend_params = kCarlFrontendDefaultParams;
  frontend_params.block_size = kBlockSize;
  CarlFrontend* frontend = CHECK_NOTNULL(CarlFrontendMake(&frontend_params));
  CHECK(CarlFrontendNumChannels(frontend) == kEmbedVowelNumChannels);

  const int intended_target = EmbedVowelTargetByName(phone);
  int count_correct = 0;
  int count_total = 0;

  int start;
  int i;
  for (start = 0; start + kBlockSize < num_samples; start += kBlockSize) {
    float input_float[kBlockSize];
    for (i = 0; i < kBlockSize; ++i) {
      input_float[i] = input_int16[start + i] / 32768.0f;
    }

    CarlFrontendProcessSamples(frontend, input_float, frame);
    float coord[2];
    EmbedVowel(frame, coord);

    count_correct += (EmbedVowelClosestTarget(coord) == intended_target);
    ++count_total;
  }

  CarlFrontendFree(frontend);
  free(frame);
  free(input_int16);

  const float accuracy = (float)count_correct / count_total;
  CHECK(accuracy >= 0.7f);
}

int main(int argc, char** argv) {
  TestTargetLookup();
  TestPhone("aa");
  TestPhone("uw");
  TestPhone("ih");
  TestPhone("iy");
  TestPhone("eh");
  TestPhone("ae");
  TestPhone("uh");
  TestPhone("er");

  puts("PASS");
  return EXIT_SUCCESS;
}
