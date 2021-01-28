// Copyright 2020-2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Benchmark of q_resampler library.
//
// NOTE: When running benchmarks, build with optimizations (-c opt) and disable
// frequency scaling (sudo cpupower frequency-set --governor performance). For
// accurate measurement, run for longer time with --benchmark_min_time=2.0.

#include <random>

#include "src/dsp/q_resampler.h"
#include "benchmark/benchmark.h"

namespace {
// Generates a float array of random normally-distributed values.
float* RandomValues(int size) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<float> dist;

  float* values = new float[size];
  for (int i = 0; i < size; ++i) {
    values[i] = dist(rng);
  }
  return values;
}
}  // namespace

void BenchmarkQResampler(benchmark::State& state, float input_sample_rate,
                         float output_sample_rate, int num_channels) {
  constexpr int kNumInFrames = 1000;
  QResampler* resampler = QResamplerMake(input_sample_rate, output_sample_rate,
                                         num_channels, kNumInFrames, nullptr);
  float* data = RandomValues(kNumInFrames * num_channels);
  QResamplerProcessSamples(resampler, data, kNumInFrames);

  for (auto _ : state) {
    benchmark::DoNotOptimize(data);
    QResamplerProcessSamples(resampler, data, kNumInFrames);
    benchmark::DoNotOptimize(QResamplerOutput(resampler));
  }

  delete[] data;
  QResamplerFree(resampler);
}

void BM_ResampleMono48To16(benchmark::State& state) {
  BenchmarkQResampler(state, 48000.0f, 16000.0f, 1);
}
BENCHMARK(BM_ResampleMono48To16);

void BM_ResampleMono44_1To16(benchmark::State& state) {
  BenchmarkQResampler(state, 44100.0f, 16000.0f, 1);
}
BENCHMARK(BM_ResampleMono44_1To16);

void BM_ResampleStereo48To16(benchmark::State& state) {
  BenchmarkQResampler(state, 48000.0f, 16000.0f, 2);
}
BENCHMARK(BM_ResampleStereo48To16);

void BM_Resample12Channels48To16(benchmark::State& state) {
  BenchmarkQResampler(state, 48000.0f, 16000.0f, 12);
}
BENCHMARK(BM_Resample12Channels48To16);

BENCHMARK_MAIN();
