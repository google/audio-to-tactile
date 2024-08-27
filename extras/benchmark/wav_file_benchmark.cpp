// Copyright 2023 Google LLC
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
// Benchmark of WAV writer library.
//
// NOTE: When running benchmarks, build with optimizations (-c opt) and disable
// frequency scaling (sudo cpupower frequency-set --governor performance). For
// accurate measurement, run for longer time with --benchmark_min_time=2.0.

#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "src/dsp/read_wav_file.h"
#include "src/dsp/write_wav_file.h"
#include "benchmark/benchmark.h"

namespace {
// Generates a float array of random normally-distributed values.
template <typename T>
std::vector<T> RandomValues(int size) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(),
                                        std::numeric_limits<T>::max());
  std::vector<T> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = dist(rng);
  }
  return values;
}

// Writes a WAV file with random mu-law samples to `file_name`.
// Out of the various supported sample encodings, mu-law is interesting for
// library coverage: we support deserializing into either int16 or int32,
// whereas most format we only support deserializing into int32.
bool WriteRandomMuLawWavFile(const std::string& file_name) {
  FILE* f = std::fopen(file_name.c_str(), "wb");
  if (f == nullptr) { return false; }

  constexpr int kSampleRateHz = 48000;
  constexpr int kNumChannels = 2;
  constexpr int kNumSamples = 10 * kSampleRateHz * kNumChannels;  // 10 seconds.
  std::vector<uint8_t> samples = RandomValues<uint8_t>(kNumSamples);

  // clang-format off
  static const uint8_t kHeader[] = {
  /* R   I   F   F                   W   A   V   E  f   m   t    _ */
    82, 73, 70, 70, 50,166, 14,  0, 87, 65, 86, 69,102,109,116, 32,   /*NOLINT*/
    18,  0,  0,  0,  7,  0,  2,  0,128,187,  0,  0,128,187,  0,  0,   /*NOLINT*/
  /*                        f    a   c  t                          */
     2,  0,  8,  0,  0,  0,102, 97, 99,116,  4,  0,  0,  0,  0, 83,   /*NOLINT*/
  /*        d   a   t   a                                          */
     7,  0,100, 97,116, 97,  0,166, 14,  0,                           /*NOLINT*/
  };
  // clang-format on
  std::fwrite(kHeader, 1, sizeof(kHeader), f);
  std::fwrite(samples.data(), 1, samples.size(), f);
  const bool error = std::ferror(f);
  std::fclose(f);
  return !error;
}
}  // namespace

void BM_ReadMuLawWavFileAsInt16(benchmark::State& state) {
  const std::string wav_file_name = std::tmpnam(nullptr);
  WriteRandomMuLawWavFile(wav_file_name.c_str());

  for (auto _ : state) {
    size_t num_samples;
    int num_channels;
    int sample_rate_hz;
    int16_t* samples = Read16BitWavFile(wav_file_name.c_str(), &num_samples,
                                        &num_channels, &sample_rate_hz);
    benchmark::DoNotOptimize(samples);
    free(samples);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_ReadMuLawWavFileAsInt16);

void BM_Read16BitWavFileAsInt16(benchmark::State& state) {
  constexpr int kSampleRateHz = 48000;
  constexpr int kNumChannels = 2;
  constexpr int kNumSamples = 10 * kSampleRateHz * kNumChannels;  // 10 seconds.
  const std::string wav_file_name = std::tmpnam(nullptr);
  WriteWavFile(wav_file_name.c_str(), RandomValues<int16_t>(kNumSamples).data(),
               kNumSamples, kSampleRateHz, kNumChannels);

  for (auto _ : state) {
    size_t num_samples;
    int num_channels;
    int sample_rate_hz;
    int16_t* samples = Read16BitWavFile(wav_file_name.c_str(), &num_samples,
                                        &num_channels, &sample_rate_hz);
    benchmark::DoNotOptimize(samples);
    free(samples);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_Read16BitWavFileAsInt16);

void BM_ReadMuLawWavFileAsInt32(benchmark::State& state) {
  const std::string wav_file_name = std::tmpnam(nullptr);
  WriteRandomMuLawWavFile(wav_file_name.c_str());

  for (auto _ : state) {
    size_t num_samples;
    int num_channels;
    int sample_rate_hz;
    int32_t* samples = ReadWavFile(wav_file_name.c_str(), &num_samples,
                                   &num_channels, &sample_rate_hz);
    benchmark::DoNotOptimize(samples);
    free(samples);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_ReadMuLawWavFileAsInt32);

void BM_Read16BitWavFileAsInt32(benchmark::State& state) {
  constexpr int kSampleRateHz = 48000;
  constexpr int kNumChannels = 2;
  constexpr int kNumSamples = 10 * kSampleRateHz * kNumChannels;  // 10 seconds.
  const std::string wav_file_name = std::tmpnam(nullptr);
  WriteWavFile(wav_file_name.c_str(), RandomValues<int16_t>(kNumSamples).data(),
               kNumSamples, kSampleRateHz, kNumChannels);

  for (auto _ : state) {
    size_t num_samples;
    int num_channels;
    int sample_rate_hz;
    int32_t* samples = ReadWavFile(wav_file_name.c_str(), &num_samples,
                                   &num_channels, &sample_rate_hz);
    benchmark::DoNotOptimize(samples);
    free(samples);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_Read16BitWavFileAsInt32);

void BM_Read24BitWavFileAsInt32(benchmark::State& state) {
  constexpr int kSampleRateHz = 48000;
  constexpr int kNumChannels = 2;
  constexpr int kNumSamples = 10 * kSampleRateHz * kNumChannels;  // 10 seconds.
  const std::string wav_file_name = std::tmpnam(nullptr);
  WriteWavFile24Bit(wav_file_name.c_str(),
                    RandomValues<int32_t>(kNumSamples).data(), kNumSamples,
                    kSampleRateHz, kNumChannels);

  for (auto _ : state) {
    size_t num_samples;
    int num_channels;
    int sample_rate_hz;
    int32_t* samples = ReadWavFile(wav_file_name.c_str(), &num_samples,
                                   &num_channels, &sample_rate_hz);
    benchmark::DoNotOptimize(samples);
    free(samples);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_Read24BitWavFileAsInt32);

void BM_WriteWavFile16Bit(benchmark::State& state) {
  constexpr int kSampleRateHz = 48000;
  constexpr int kNumChannels = 2;
  constexpr int kNumSamples = 10 * kSampleRateHz * kNumChannels;  // 10 seconds.
  const std::string wav_file_name = std::tmpnam(nullptr);
  std::vector<int16_t> samples = RandomValues<int16_t>(kNumSamples);

  for (auto _ : state) {
    benchmark::DoNotOptimize(samples.data());
    WriteWavFile(wav_file_name.c_str(), samples.data(), kNumSamples,
                 kSampleRateHz, kNumChannels);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_WriteWavFile16Bit);

void BM_WriteWavFile24Bit(benchmark::State& state) {
  constexpr int kSampleRateHz = 48000;
  constexpr int kNumChannels = 2;
  constexpr int kNumSamples = 10 * kSampleRateHz * kNumChannels;  // 10 seconds.
  const std::string wav_file_name = std::tmpnam(nullptr);
  std::vector<int32_t> samples = RandomValues<int32_t>(kNumSamples);

  for (auto _ : state) {
    benchmark::DoNotOptimize(samples);
    WriteWavFile24Bit(wav_file_name.c_str(), samples.data(), kNumSamples,
                      kSampleRateHz, kNumChannels);
  }

  std::remove(wav_file_name.c_str());
}
BENCHMARK(BM_WriteWavFile24Bit);

BENCHMARK_MAIN();
