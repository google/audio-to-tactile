# Copyright 2020-2021 Google LLC
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
# Python bindings.

load("//:defs.bzl", "py_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

py_library(
    name = "dsp",
    srcs = ["dsp.py"],
    srcs_version = "PY3",
    deps = [
        ":fast_fun_python_bindings",
        ":q_resampler_python_bindings",
        ":wav_io_python_bindings",
    ],
)

py_test(
    name = "dsp_test",
    srcs = ["dsp_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dsp",
    ],
)

py_extension(
    name = "fast_fun_python_bindings",
    srcs = ["fast_fun_python_bindings.c"],
    visibility = ["//visibility:private"],
    deps = [
        "//:dsp",
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "plot_test",
    srcs = ["plot_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":plot",
    ],
)

py_extension(
    name = "q_resampler_python_bindings",
    srcs = ["q_resampler_python_bindings.c"],
    visibility = ["//visibility:private"],
    deps = [
        "//:dsp",
    ],
)

py_extension(
    name = "wav_io_python_bindings",
    srcs = ["wav_io_python_bindings.c"],
    visibility = ["//visibility:private"],
    deps = [
        "//:dsp",
    ],
)

py_extension(
    name = "frontend",
    srcs = ["frontend_python_bindings.c"],
    deps = [
        "//:frontend",
    ],
)
