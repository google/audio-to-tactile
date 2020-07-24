# Copyright 2019 Google LLC
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
# Audio-to-tactile Python bindings.

load("//:defs.bzl", "c_library", "py_extension")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "dsp",
    srcs = ["dsp.py"],
)

py_test(
    name = "dsp_test",
    srcs = ["dsp_test.py"],
    deps = [
        ":dsp",
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    srcs_version = "PY3",
)

py_library(
    name = "read_nist_sphere",
    srcs = ["read_nist_sphere.py"],
)

py_test(
    name = "read_nist_sphere_test",
    srcs = ["read_nist_sphere_test.py"],
    deps = [
        ":read_nist_sphere",
    ],
)

py_library(
    name = "stats",
    srcs = ["stats.py"],
)

py_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    deps = [
        ":stats",
    ],
)

py_extension(
    name = "energy_envelope",
    srcs = ["energy_envelope_python_bindings.c"],
    deps = [
        "//audio/tactile/energy_envelope",
    ],
)

py_extension(
    name = "tactile_processor",
    srcs = ["tactile_processor_python_bindings.c"],
    deps = [
        "//audio/tactile:tactile_processor",
    ],
)

py_test(
    name = "tactile_processor_test",
    srcs = ["tactile_processor_test.py"],
    data = ["//audio/tactile/testdata:phone_wavs"],
    deps = [
        ":tactile_processor",
        "//audio/dsp/portable/python:wav_io",
    ],
)

c_library(
    name = "tactile_worker_impl",
    srcs = ["tactile_worker.c"],
    hdrs = ["tactile_worker.h"],
    deps = [
        "//audio/tactile:channel_map",
        "//audio/tactile:portaudio_device",
        "//audio/tactile:post_processor",
        "//audio/tactile:tactile_processor",
        "//audio/tactile:util",
    ],
    linkopts = [
        "-lportaudio",
    ],
)

py_extension(
    name = "tactile_worker",
    srcs = ["tactile_worker_python_bindings.c"],
    deps = [
        ":tactile_worker_impl",
    ],
)

py_binary(
    name = "tactile_worker_python_demo",
    srcs = ["tactile_worker_python_demo.py"],
    python_version = "PY3",
    deps = [
        ":tactile_worker",
    ],
)
