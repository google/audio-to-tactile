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

#include "src/tactile/tuning.h"

#include "src/dsp/logging.h"

/* Checks that `actual` is within 0.2% of `expected`. */
static int IsClose(float expected, float actual) {
  return fabs(expected - actual) <= 0.002f * fabs(expected);
}

void TestTuningSetOutputGain(TactileProcessor* processor, int value,
                             float expected_db, float expected_linear) {
  printf("TestTuningSetOutputGain(%d)\n", value);

  float output_gain_db = TuningSetOutputGain(processor, value);
  CHECK(IsClose(expected_db, output_gain_db));

  int i;
  for (i = 0; i < 4; ++i) {
    CHECK(IsClose(expected_linear, processor->channel_states[i].output_gain));
  }
}

void TestTuningSetDenoising(TactileProcessor* processor, int value,
                            float expected) {
  printf("TestTuningSetDenoising(%d)\n", value);

  float delta = TuningSetDenoising(processor, value);
  CHECK(IsClose(expected, delta));

  int i;
  for (i = 0; i < 4; ++i) {
    CHECK(IsClose(expected, processor->channel_states[i].pcen_delta));
  }
}

void TestTuningSetCompression(TactileProcessor* processor, int value,
                              float expected) {
  printf("TestTuningSetCompression(%d)\n", value);

  float beta = TuningSetCompression(processor, value);
  CHECK(IsClose(expected, beta));

  int i;
  for (i = 0; i < 4; ++i) {
    CHECK(fabs(processor->channel_states[i].pcen_beta - expected) < 1e-4f);
  }
}

int main(int argc, char** argv) {
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  TactileProcessor* processor = CHECK_NOTNULL(TactileProcessorMake(&params));

  TestTuningSetOutputGain(processor, 0, -18.0f, 0.1259f);
  TestTuningSetOutputGain(processor, 191, -0.0235f, 0.997f);
  TestTuningSetOutputGain(processor, 255, 6.0f, 1.9953f);
  TestTuningSetDenoising(processor, 0, 0.0001f);
  TestTuningSetDenoising(processor, 85, 0.001f);
  TestTuningSetDenoising(processor, 255, 0.1f);
  TestTuningSetCompression(processor, 0, 0.1f);
  TestTuningSetCompression(processor, 96, 0.2506f);
  TestTuningSetCompression(processor, 255, 0.5f);

  TactileProcessorFree(processor);

  puts("PASS");
  return EXIT_SUCCESS;
}
