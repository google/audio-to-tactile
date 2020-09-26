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

py_extension(
    name = "energy_envelope",
    srcs = ["energy_envelope_python_bindings.c"],
    deps = [
        "//tactile/energy_envelope",
    ],
)

py_extension(
    name = "tactile_processor",
    srcs = ["tactile_processor_python_bindings.c"],
    deps = [
        "//tactile:tactile_processor",
    ],
)

py_test(
    name = "tactile_processor_test",
    srcs = ["tactile_processor_test.py"],
    data = ["//testdata:phone_wavs"],
    deps = [
        ":tactile_processor",
        "//dsp/python:wav_io",
    ],
)

c_library(
    name = "tactile_worker_impl",
    srcs = ["tactile_worker.c"],
    hdrs = ["tactile_worker.h"],
    deps = [
        "//tactile:channel_map",
        "//tactile:portaudio_device",
        "//tactile:post_processor",
        "//tactile:tactile_processor",
        "//tactile:util",
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
