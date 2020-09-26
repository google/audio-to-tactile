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
# An embedding that is correlated with phones, useful to make distinguishable
# phones in audio-to-tactile representations.

load("//:defs.bzl", "c_binary", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

c_library(
    name = "embed_vowel",
    srcs = [
        "embed_vowel.c",
        "embed_vowel_params.h",
    ],
    hdrs = ["embed_vowel.h"],
    deps = [
        "//dsp:fast_fun",
        "//tactile:hexagon_interpolation",
        "//phonetics:nn_ops",
    ],
)

c_test(
    name = "embed_vowel_test",
    srcs = ["embed_vowel_test.c"],
    data = [
        "//testdata:phone_wavs",
    ],
    deps = [
        ":embed_vowel",
        "//dsp:logging",
        "//dsp:read_wav_file",
        "//frontend:carl_frontend",
    ],
)

c_library(
    name = "nn_ops",
    srcs = ["nn_ops.c"],
    hdrs = ["nn_ops.h"],
    deps = [
        "//dsp:fast_fun",
    ],
)

c_test(
    name = "nn_ops_test",
    srcs = ["nn_ops_test.c"],
    deps = [
        ":nn_ops",
        "//dsp:logging",
    ],
)
