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
 *
 *
 * Benchmark of TactileProcessor with default settings.
 *
 * Results on Lenovo Thinkpad, 2019-08-05:
 * BM_EnergyEnvelope(1):   time =  4352 ns, iter = 166646
 * BM_EnergyEnvelope(2):   time =  2277 ns, iter = 328612
 * BM_EnergyEnvelope(4):   time =  1264 ns, iter = 500000
 * BM_TactileProcessor(1): time = 28839 ns, iter =  41526
 * BM_TactileProcessor(2): time = 20755 ns, iter =  50000
 * BM_TactileProcessor(4): time = 16360 ns, iter =  50000
 */

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "audio/dsp/portable/logging.h"
#include "audio/tactile/tactile_processor.h"
#include "audio/tactile/energy_envelope/energy_envelope.h"

static int64_t CurrentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ((int64_t)ts.tv_sec) * INT64_C(1000000000) + ((int64_t)ts.tv_nsec);
}

typedef struct {
  int iter;
  int num_iters;
  int64_t start_time_ns;
  int average_ns;
} BenchmarkState;
static const BenchmarkState kBenchmarkStateInit = {0, 0, 0, 0};

const long kMinIters = 50;
const long kMaxIters = 1000000000L;
const double kMinTimeSeconds = 0.5;

int BenchmarkNextMeasurement(BenchmarkState* state) {
  const int64_t now = CurrentTimeNs();
  state->iter = 0;

  if (state->num_iters == 0) {
    state->num_iters = kMinIters;
    state->start_time_ns = now;
    return 1;
  }

  const double elapsed_ns = now - state->start_time_ns;
  const double kMinTimeNs = 1e9 * kMinTimeSeconds;
  if (state->num_iters < kMaxIters && elapsed_ns < kMinTimeNs) {
    double multiplier = (1.5 * kMinTimeNs) / elapsed_ns;
    if (multiplier < 1.5) { multiplier = 1.5; }
    if (multiplier > 10.0) { multiplier = 10.0; }
    state->num_iters *= multiplier;
    if (state->num_iters > kMaxIters) { state->num_iters = kMaxIters; }
    state->start_time_ns = CurrentTimeNs();
    return 1;
  }

  state->average_ns = (int)(elapsed_ns / state->num_iters + 0.5);
  return 0;
}

static int BenchmarkKeepRunning(BenchmarkState* state) {
  if (++state->iter < state->num_iters) {
    return 1;
  }
  return BenchmarkNextMeasurement(state);
}

void BM_EnergyEnvelope(int decimation_factor) {
  EnergyEnvelopeParams params = kEnergyEnvelopeVowelParams;
  EnergyEnvelope channel;
  CHECK(EnergyEnvelopeInit(&channel, &params, 16000.0f, decimation_factor));

  const int block_size =  64;
  float* input = (float*)CHECK_NOTNULL(malloc(sizeof(float) * block_size));
  float* output = (float*)CHECK_NOTNULL(malloc(sizeof(float) * block_size));

  int i;
  for (i = 0; i < block_size; ++i) {
    input[i] = 0.0f;
  }
  for (i = 0; i < block_size; ++i) {
    output[i] = 0.0f;
  }

  BenchmarkState benchmark = kBenchmarkStateInit;
  while (BenchmarkKeepRunning(&benchmark)) {
    EnergyEnvelopeProcessSamples(&channel, input, block_size, output, 1);
  }

  free(output);
  free(input);

  printf("BM_EnergyEnvelope(%d): "
      "time = %10d ns, iter = %10d\n",
      decimation_factor, benchmark.average_ns, benchmark.num_iters);
}

void BM_TactileProcessor(int decimation_factor) {
  const int block_size = 64;
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  params.frontend_params.block_size = block_size;
  params.decimation_factor = decimation_factor;
  TactileProcessor* tactile_processor = CHECK_NOTNULL(
      TactileProcessorMake(&params));

  float* input = (float*)CHECK_NOTNULL(malloc(sizeof(float) * block_size));
  float* output = (float*)CHECK_NOTNULL(
      malloc(sizeof(float) * block_size * kTactileProcessorNumTactors));

  int i;
  for (i = 0; i < block_size; ++i) {
    input[i] = 0.0f;
  }
  for (i = 0; i < block_size * kTactileProcessorNumTactors; ++i) {
    output[i] = 0.0f;
  }

  BenchmarkState benchmark = kBenchmarkStateInit;
  while (BenchmarkKeepRunning(&benchmark)) {
    TactileProcessorProcessSamples(tactile_processor, input, output);
  }

  free(output);
  free(input);
  TactileProcessorFree(tactile_processor);

  printf("BM_TactileProcessor(%d): "
      "time = %10d ns, iter = %10d\n",
      decimation_factor, benchmark.average_ns, benchmark.num_iters);
}

int main(int argc, char** argv) {
  BM_EnergyEnvelope(1);
  BM_EnergyEnvelope(2);
  BM_EnergyEnvelope(4);
  BM_EnergyEnvelope(8);

  BM_TactileProcessor(1);
  BM_TactileProcessor(2);
  BM_TactileProcessor(4);
  BM_TactileProcessor(8);
  return 0;
}
