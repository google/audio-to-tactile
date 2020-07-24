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
# //audio/dsp/portable Python bindings.

load("//:defs.bzl", "py_extension")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "rational_factor_resampler",
    srcs = ["rational_factor_resampler.py"],
    deps = [
        ":rational_factor_resampler_python_bindings",
    ],
)

py_extension(
    name = "rational_factor_resampler_python_bindings",
    srcs = ["rational_factor_resampler_python_bindings.c"],
    visibility = ["//visibility:private"],
    deps = [
        "//audio/dsp/portable:rational_factor_resampler",
        "//audio/dsp/portable:rational_factor_resampler_kernel",
    ],
)

py_test(
    name = "rational_factor_resampler_test",
    srcs = ["rational_factor_resampler_test.py"],
    deps = [
        ":rational_factor_resampler",
    ],
)

py_library(
    name = "wav_io",
    srcs = ["wav_io.py"],
    deps = [
        ":wav_io_python_bindings",
    ],
)

py_extension(
    name = "wav_io_python_bindings",
    srcs = ["wav_io_python_bindings.c"],
    visibility = ["//visibility:private"],
    deps = [
        "//audio/dsp/portable:read_wav_file_generic",
        "//audio/dsp/portable:write_wav_file_generic",
    ],
)

py_test(
    name = "wav_io_test",
    srcs = ["wav_io_test.py"],
    deps = [
        ":wav_io",
    ],
)

