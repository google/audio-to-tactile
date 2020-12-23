# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
#
# Benchmarks.

package(licenses = ["notice"])

C_OPTS = ["-Wno-unused-function"]

cc_binary(
    name = "fast_fun_benchmark",
    srcs = ["fast_fun_benchmark.cpp"],
    copts = C_OPTS,
    deps = [
        "//:dsp",
        "@benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "fft_benchmark",
    srcs = ["fft_benchmark.cpp"],
    copts = C_OPTS,
    deps = [
        "//:dsp",
        "@benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "rational_factor_resampler_benchmark",
    srcs = ["rational_factor_resampler_benchmark.cpp"],
    copts = C_OPTS,
    deps = [
        "//:dsp",
        "@benchmark//:benchmark_main",
    ],
)
