// Copyright 2020, 2023 Google LLC
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
// Benchmark of fast_fun library.
//
// This benchmark measures the time to make 1000 calls of FastLog2, FastExp2,
// FastPow, and FastTanh compared to functions log2f, exp2f, powf, and tanhf in
// the math.h standard library.
//
// NOTE: When running benchmarks, build with optimizations (-c opt) and disable
// frequency scaling (sudo cpupower frequency-set --governor performance). For
// accurate measurement, run for longer time with --benchmark_min_time=2.0.

#include <math.h>
#include <random>
#include <vector>

#include "src/dsp/fast_fun.h"
#include "benchmark/benchmark.h"

static constexpr int kNumCalls = 1000;

// log2 benchmarks. ____________________________________________________________

namespace {
std::vector<float> Log2TestValues() {
  std::vector<float> values(kNumCalls);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-125.5f, 125.5f);
  for (float& value : values) {
    value = exp2f(dist(rng));
  }
  return values;
}
}  // namespace

// Benchmark of FastLog2().
static void BM_FastLog2(benchmark::State& state) {
  std::vector<float> values = Log2TestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = FastLog2(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_FastLog2);

// Benchmark of math.h log2f().
static void BM_math_log2f(benchmark::State& state) {
  std::vector<float> values = Log2TestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = log2f(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_math_log2f);

// exp2 benchmarks. ____________________________________________________________

namespace {
std::vector<float> Exp2TestValues() {
  std::vector<float> values(kNumCalls);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-125.5f, 125.5f);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}
}  // namespace

// Benchmark of FastExp2().
static void BM_FastExp2(benchmark::State& state) {
  std::vector<float> values = Exp2TestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = FastExp2(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_FastExp2);

// Benchmark of math.h exp2f().
static void BM_math_exp2f(benchmark::State& state) {
  std::vector<float> values = Exp2TestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = exp2f(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_math_exp2f);

// pow benchmarks. _____________________________________________________________

namespace {
std::vector<std::pair<float, float>> PowTestValues() {
  std::vector<std::pair<float, float>> values(kNumCalls);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& value : values) {
    value = {0.1f + 49.9f * dist(rng), 4 * dist(rng) - 2};
  }
  return values;
}
}  // namespace

// Benchmark of FastPow().
static void BM_FastPow(benchmark::State& state) {
  std::vector<std::pair<float, float>> values = PowTestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = FastPow(values[i].first, values[i].second);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_FastPow);

// Benchmark of math.h powf().
static void BM_math_powf(benchmark::State& state) {
  std::vector<std::pair<float, float>> values = PowTestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = powf(values[i].first, values[i].second);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_math_powf);

// tanh benchmarks. ____________________________________________________________

namespace {
std::vector<float> TanhTestValues() {
  std::vector<float> values(kNumCalls);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-12.0f, 12.0f);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}
}  // namespace

// Benchmark of FastTanh().
static void BM_FastTanh(benchmark::State& state) {
  std::vector<float> values = TanhTestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = FastTanh(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_FastTanh);

// Benchmark of math.h tanf().
static void BM_math_tanhf(benchmark::State& state) {
  std::vector<float> values = TanhTestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = tanhf(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_math_tanhf);

// sigmoid benchmarks. _________________________________________________________
// The sigmoid benchmarks reuse TanhTestValues() from above.

// Benchmark of FastSigmoid().
static void BM_FastSigmoid(benchmark::State& state) {
  std::vector<float> values = TanhTestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = FastSigmoid(values[i]);
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_FastSigmoid);

// Benchmark of math.h-based sigmoid: `1.0f / (1.0f + expf(-x))`.
static void BM_math_sigmoid(benchmark::State& state) {
  std::vector<float> values = TanhTestValues();
  std::vector<float> result(kNumCalls);

  for (auto _ : state) {
    benchmark::ClobberMemory();
    for (int i = 0; i < kNumCalls; ++i) {
      result[i] = 1.0f / (1.0f + expf(-values[i]));
    }
    benchmark::DoNotOptimize(result.data());
  }
}
BENCHMARK(BM_math_sigmoid);

BENCHMARK_MAIN();
