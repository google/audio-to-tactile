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
# Tactor energy envelope, a simple audio-to-tactile representation.

load("//:defs.bzl", "c_binary", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

c_binary(
    name = "run_energy_envelope",
    srcs = ["run_energy_envelope.c"],
    deps = [
        ":energy_envelope",
        "//tactile:portaudio_device",
        "//tactile:util",
    ],
    linkopts = [
        "-lportaudio",
    ],
)

c_library(
    name = "energy_envelope",
    srcs = ["energy_envelope.c"],
    hdrs = ["energy_envelope.h"],
    deps = [
        "//dsp:butterworth",
        "//dsp:fast_fun",
        "//dsp:math_constants",
    ],
)

c_test(
    name = "energy_envelope_test",
    srcs = ["energy_envelope_test.c"],
    deps = [
        ":energy_envelope",
        "//dsp:logging",
        "//dsp:math_constants",
    ],
)

c_binary(
    name = "run_energy_envelope_on_wav",
    srcs = ["run_energy_envelope_on_wav.c"],
    deps = [
        ":energy_envelope",
        "//dsp:read_wav_file",
        "//dsp:write_wav_file",
        "//tactile:util",
    ],
)
