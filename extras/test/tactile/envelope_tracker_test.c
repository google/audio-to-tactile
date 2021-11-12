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

#include "src/tactile/envelope_tracker.h"

#include <math.h>

#include "src/dsp/logging.h"
#include "src/dsp/math_constants.h"

#define kSampleRateHz 16000

/* Tests whether `actual` is within 15% of `expected`. */
static int /*bool*/ IsClose(float actual, float expected) {
  return fabs(actual - expected) <= 0.15f * expected;
}

static void TestBasic(void) {
  puts("TestBasic");
  EnvelopeTracker tracker;
  EnvelopeTrackerInit(&tracker, kSampleRateHz);

  const int num_samples =
      kEnvelopeTrackerRecordPoints * tracker.samples_per_measurement;
  float* input = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));

  const float kFrequency = 1000.0f;
  const float rad_per_sample = 2 * M_PI * kFrequency / kSampleRateHz;

  float power;
  for (power = 1e-3; power < 2; power *= 2) {
    /* Make a 1kHz sine wave scaled such that it has `power` power, plus a DC
     * offset that the highpass filter should remove.
     */
    const float kOffset = 0.3f;
    const float scale = sqrt(2 * power);
    int i;
    for (i = 0; i < num_samples; ++i) {
      input[i] = scale * sin(rad_per_sample * i) + kOffset;
    }

    EnvelopeTrackerInit(&tracker, kSampleRateHz);

    const float* src = input;
    for (i = 0; i < kEnvelopeTrackerRecordPoints - 1; ++i) {
      CHECK(!EnvelopeTrackerProcessSamples(&tracker, src,
                                           tracker.samples_per_measurement));
      src += tracker.samples_per_measurement;
    }
    CHECK(EnvelopeTrackerProcessSamples(&tracker, src,
                                        tracker.samples_per_measurement));

    uint8_t record[kEnvelopeTrackerRecordBytes];
    EnvelopeTrackerGetRecord(&tracker, record);
    float decoded[kEnvelopeTrackerRecordPoints];
    EnvelopeTrackerDecodeRecord(record, decoded);

    /* Last decoded measurement is within 20% of expected power. */
    CHECK(IsClose(decoded[kEnvelopeTrackerRecordPoints - 1], power));
  }

  free(input);
}

static void TestStreamingRandomBlockSizes(void) {
  puts("TestStreamingRandomBlockSizes");
  EnvelopeTracker tracker;
  EnvelopeTrackerInit(&tracker, kSampleRateHz);

  const int num_samples =
      kEnvelopeTrackerRecordPoints * tracker.samples_per_measurement;
  float* input = (float*)CHECK_NOTNULL(malloc(num_samples * sizeof(float)));
  int i;
  for (i = 0; i < num_samples; ++i) {
    input[i] = rand() / (float)RAND_MAX;
  }

  CHECK(EnvelopeTrackerProcessSamples(&tracker, input, num_samples));
  uint8_t record[kEnvelopeTrackerRecordBytes];
  EnvelopeTrackerGetRecord(&tracker, record);
  float decoded_nonstreaming[kEnvelopeTrackerRecordPoints];
  EnvelopeTrackerDecodeRecord(record, decoded_nonstreaming);

  int trial;
  for (trial = 0; trial < 3; ++trial) {
    EnvelopeTracker tracker2;
    EnvelopeTrackerInit(&tracker2, kSampleRateHz);

    const float* src = input;
    for (i = 0; i < num_samples;) {
      /* Get next block of input samples with random size between 0 and 50. */
      int block_size = (int)floor((((float)rand()) / RAND_MAX) * 50 + 0.5);
      if (block_size > num_samples - i) {
        block_size = num_samples - i;
      }

      EnvelopeTrackerProcessSamples(&tracker2, src, block_size);
      src += block_size;
      i += block_size;
    }

    EnvelopeTrackerGetRecord(&tracker, record);
    float decoded_streaming[kEnvelopeTrackerRecordPoints];
    EnvelopeTrackerDecodeRecord(record, decoded_streaming);

    /* Check that streaming vs. nonstreaming powers are close. */
    for (i = 0; i < kEnvelopeTrackerRecordPoints; ++i) {
      CHECK(IsClose(decoded_streaming[i], decoded_nonstreaming[i]));
    }
  }

  free(input);
}

int main(int argc, char** argv) {
  srand(0);
  TestBasic();
  TestStreamingRandomBlockSizes();

  puts("PASS");
  return EXIT_SUCCESS;
}
