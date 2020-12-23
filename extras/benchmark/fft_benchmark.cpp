// Copyright 2020 Google LLC
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
// Benchmark of FFT library.
//
// The benchmarks measure the times to call
//
//  * FftForwardScrambledTransform
//  * FftInverseScrambledTransform
//  * FftScramble (same as FftUnscramble)
//
// for transform sizes 64, 256, and 2014.
//
// NOTE: When running benchmarks, build with optimizations (-c opt) and disable
// frequency scaling (sudo cpupower frequency-set --governor performance). For
// accurate measurement, run for longer time with --benchmark_min_time=2.0.

#include <random>

#include "src/dsp/fft.h"
#include "benchmark/benchmark.h"

namespace {
// Generates a ComplexFloat array of random normally-distributed values.
ComplexFloat* RandomValues(int size) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<float> dist;

  ComplexFloat* values = new ComplexFloat[size];
  for (int i = 0; i < size; ++i) {
    values[i].real = dist(rng);
    values[i].imag = dist(rng);
  }
  return values;
}
}  // namespace

void BM_FftForwardScrambledTransform(benchmark::State& state) {
  const int transform_size = state.range(0);
  ComplexFloat* data = RandomValues(transform_size);

  for (auto _ : state) {
    FftForwardScrambledTransform(data, transform_size);
    benchmark::DoNotOptimize(data);
  }

  delete [] data;
}
BENCHMARK(BM_FftForwardScrambledTransform)->Arg(64)->Arg(256)->Arg(1024);

void BM_FftInverseScrambledTransform(benchmark::State& state) {
  const int transform_size = state.range(0);
  ComplexFloat* data = RandomValues(transform_size);

  for (auto _ : state) {
    FftInverseScrambledTransform(data, transform_size);
    benchmark::DoNotOptimize(data);
  }

  delete [] data;
}
BENCHMARK(BM_FftInverseScrambledTransform)->Arg(64)->Arg(256)->Arg(1024);

void BM_FftScramble(benchmark::State& state) {
  const int transform_size = state.range(0);
  ComplexFloat* data = RandomValues(transform_size);

  for (auto _ : state) {
    FftScramble(data, transform_size);
    benchmark::DoNotOptimize(data);
  }

  delete [] data;
}
BENCHMARK(BM_FftScramble)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK_MAIN();
