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
 *
 *
 * Tests for classify_phoneme.
 */

#include "src/phonetics/classify_phoneme.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/read_wav_file.h"
#include "src/frontend/carl_frontend.h"

#define kClassifierInputHz 16000
#define kBlockSize 128

/* Finds the index of phoneme `name` in kClassifyPhonemePhonemeNames. */
static int FindPhoneme(const char* name) {
  int i;
  for (i = 0; i < kClassifyPhonemeNumPhonemes; ++i) {
    if (!strcmp(name, kClassifyPhonemePhonemeNames[i])) {
      return i;
    }
  }
  fprintf(stderr, "Error: Phoneme not found, \"%s\".\n", name);
  exit(EXIT_FAILURE);
}

/* A forgiving test that the classifier is basically working. Runs CarlFrontend
 * + ClassifyPhoneme on a short WAV recording of a pure phoneme, and checks that
 * a moderately confident score is sometimes given to the correct label.
 */
void TestPhoneme(const char* phoneme) {
  printf("TestPhoneme(\"%s\")\n", phoneme);
  char wav_file[1024];
  sprintf(wav_file,
          "extras/test/testdata/phone_%s.wav",
          phoneme);

  size_t num_samples;
  int num_channels;
  int sample_rate_hz;
  int16_t* input_int16 = (int16_t*)CHECK_NOTNULL(Read16BitWavFile(
      wav_file, &num_samples, &num_channels, &sample_rate_hz));
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == kClassifierInputHz);
  CHECK(num_samples >= kClassifyPhonemeNumChannels * kClassifyPhonemeNumFrames);

  /* Make frontend to get CARL frames. The classifier expects input sample rate
   * CLASSIFIER_INPUT_HZ, block_size=128, pcen_cross_channel_diffusivity=60, and
   * otherwise the default frontend settings.
   */
  CarlFrontendParams frontend_params = kCarlFrontendDefaultParams;
  frontend_params.input_sample_rate_hz = kClassifierInputHz;
  frontend_params.block_size = kBlockSize;
  /* Rate of smoothing of PCEN state across frequency (for more detail see
   * audio/tactile/frontend/carl_frontend.h).
   */
  frontend_params.pcen_cross_channel_diffusivity = 60.0f;
  CarlFrontend* frontend = CHECK_NOTNULL(CarlFrontendMake(&frontend_params));

  CHECK(CarlFrontendNumChannels(frontend) == kClassifyPhonemeNumChannels);
  const int bytes_per_frame = sizeof(float) * kClassifyPhonemeNumChannels;
  float* frame_buffer = (float*)CHECK_NOTNULL(malloc(
      bytes_per_frame * kClassifyPhonemeNumFrames));

  const int intended_label = FindPhoneme(phoneme);
  ClassifyPhonemeScores scores;
  int count_correct = 0;
  int count_total = 0;

  int start;
  int i;
  for (start = 0; start + kBlockSize < num_samples; start += kBlockSize) {
    float input_float[kBlockSize];
    for (i = 0; i < kBlockSize; ++i) {
      input_float[i] = input_int16[start + i] / 32768.0f;
    }

    memmove(frame_buffer, frame_buffer + kClassifyPhonemeNumChannels,
            bytes_per_frame * (kClassifyPhonemeNumFrames - 1));
    CarlFrontendProcessSamples(
        frontend, input_float,
        frame_buffer +
            kClassifyPhonemeNumChannels * (kClassifyPhonemeNumFrames - 1));

    if (start < kBlockSize * (kClassifyPhonemeNumFrames - 1)) { continue; }

    ClassifyPhoneme(frame_buffer, NULL, &scores);

    /* Count as "correct" if correct label's score is moderately confident. */
    count_correct += (scores.phoneme[intended_label] > 0.1f);
    ++count_total;
  }

  CarlFrontendFree(frontend);
  free(frame_buffer);
  free(input_int16);

  const float accuracy = (float)count_correct / count_total;
  CHECK(accuracy >= 0.6f);
}

void TestLabelOutput() {
  puts("TestLabelOutput");

  const int kInputSize =
      kClassifyPhonemeNumFrames * kClassifyPhonemeNumChannels;
  float* frames = (float*)CHECK_NOTNULL(malloc(sizeof(float) * kInputSize));

  ClassifyPhonemeLabels labels;
  ClassifyPhonemeLabels labels_no_scores;
  ClassifyPhonemeScores scores;

  int trial;
  for (trial = 0; trial < 50; ++trial) {
    int i;
    for (i = 0; i < kInputSize; ++i) {
      frames[i] = rand() / (float)RAND_MAX;
    }

    /* Request both labels and scores. */
    ClassifyPhoneme(frames, &labels, &scores);

    /* Requesting only the labels should work, too. */
    ClassifyPhoneme(frames, &labels_no_scores, NULL);
    CHECK(labels_no_scores.phoneme == labels.phoneme);
    CHECK(labels_no_scores.manner == labels.manner);
    CHECK(labels_no_scores.place == labels.place);
    CHECK(labels_no_scores.voiced == labels.voiced);

    CHECK(0 <= labels.phoneme && labels.phoneme < kClassifyPhonemeNumPhonemes);
    CHECK(0 <= labels.manner && labels.manner < kClassifyPhonemeNumManners);
    CHECK(0 <= labels.place && labels.place < kClassifyPhonemeNumPlaces);
    const char* phoneme_name = kClassifyPhonemePhonemeNames[labels.phoneme];
    const char* manner_name = kClassifyPhonemeMannerNames[labels.manner];
    const char* place_name = kClassifyPhonemePlaceNames[labels.place];

    /* labels.phoneme is the argmax of scores.phoneme. */
    for (i = 0; i < kClassifyPhonemeNumPhonemes; ++i) {
      CHECK(scores.phoneme[labels.phoneme] >= scores.phoneme[i]);
    }

    if (!strcmp(phoneme_name, "sil")) {
      CHECK(!labels.vad); /* VAD should be false for "sil". */
      continue;
    }

    CHECK(labels.vad); /* And otherwise, VAD should be true. */

    /* Spot check that category labels are consistent with phoneme label. */

    if (strstr("em,m,en,n,ng", phoneme_name)) { /* Nasals. */
      CHECK(!strcmp(manner_name, "nasal"));
      CHECK(!labels.vowel);
      CHECK(labels.voiced);
    } else if (strstr("b,d,g,k,p,t,q", phoneme_name)) { /* Stops. */
      CHECK(!strcmp(manner_name, "stop"));
      CHECK(!labels.vowel);
    } else if (strstr("th,dh,f,v,s,z,sh,zh,hh",
                      phoneme_name)) { /* Fricatives. */
      CHECK(!strcmp(manner_name, "fricative"));
      CHECK(!labels.vowel);
    } else if (strstr("aw,ay,ey,ow,oy", phoneme_name)) { /* Tense diphthongs. */
      CHECK(labels.vowel);
      CHECK(labels.diphthong);
      CHECK(!labels.lax_vowel);
      CHECK(labels.voiced);
      /* "Approximant" is the closest to vowel among our manner classes. */
      CHECK(!strcmp(manner_name, "approximant"));
    } else if (strstr("ae,ah,eh,uh,ih", phoneme_name)) { /* Lax monophthongs. */
      CHECK(labels.vowel);
      CHECK(!labels.diphthong);
      CHECK(labels.lax_vowel);
      CHECK(labels.voiced);
      CHECK(!strcmp(manner_name, "approximant"));
    }

    if (strstr("b,f,em,m,p,v,w", phoneme_name)) {
      CHECK(!strcmp(place_name, "front"));
      CHECK(!labels.vowel);
    } else if (strstr("d,dh,dx,el,l,en,n,nx,s,t,th,z", phoneme_name)) {
      CHECK(!strcmp(place_name, "middle"));
      CHECK(!labels.vowel);
    }
  }

  free(frames);
}

int main(int argc, char** argv) {
  srand(0);
  TestPhoneme("ae");
  TestPhoneme("er");
  TestPhoneme("z");
  TestLabelOutput();

  puts("PASS");
  return EXIT_SUCCESS;
}
