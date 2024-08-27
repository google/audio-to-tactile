# Copyright 2019, 2021 Google LLC
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
# Python bindings for src/tactile.

load("//:defs.bzl", "py_extension")
load("//third_party/bazel_rules/rules_python/python:py_extension.bzl", "py_extension")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_extension(
    name = "enveloper",
    srcs = ["enveloper_python_bindings.c"],
    deps = [
        "//:tactile",
    ],
)

py_test(
    name = "enveloper_test",
    srcs = ["enveloper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":enveloper",
    ],
)

py_extension(
    name = "post_processor",
    srcs = ["post_processor_python_bindings.c"],
    deps = [
        "//:tactile",
    ],
)

py_extension(
    name = "tactile_processor",
    srcs = ["tactile_processor_python_bindings.c"],
    deps = [
        "//:tactile",
    ],
)

py_test(
    name = "tactile_processor_test",
    srcs = ["tactile_processor_test.py"],
    data = ["//extras/test/testdata:phone_wavs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tactile_processor",
        "//extras/python:dsp",
    ],
)

py_extension(
    name = "tactile_worker",
    srcs = [
        "tactile_worker.c",
        "tactile_worker.h",
        "tactile_worker_python_bindings.c",
    ],
    deps = [
        "//:dsp",
        "//:tactile",
        "//extras/tools:channel_map_tui",
        "//extras/tools:portaudio_device",
        "//extras/tools:util",
    ],
)

py_binary(
    name = "tactile_worker_python_demo",
    srcs = ["tactile_worker_python_demo.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tactile_worker",
        "//extras/python:dsp",
    ],
)

py_library(
    name = "tap_out",
    srcs = ["tap_out.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "tap_out_test",
    srcs = ["tap_out_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tap_out",
    ],
)

py_binary(
    name = "run_tap_out",
    srcs = ["run_tap_out.py"],
    srcs_version = "PY3",
    deps = [
        ":tap_out",
        "//extras/python:dsp",
        "//extras/python:plot",
    ],
)
