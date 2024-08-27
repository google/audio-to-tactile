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
# Tests for src/phonetics.

load("//:defs.bzl", "c_test")

licenses(["notice"])

c_test(
    name = "classify_phoneme_test",
    srcs = ["classify_phoneme_test.c"],
    data = ["//extras/test/testdata:phone_wavs"],
    deps = [
        "//:dsp",
        "//:frontend",
        "//:phonetics",
    ],
)

c_test(
    name = "embed_vowel_test",
    srcs = ["embed_vowel_test.c"],
    data = ["//extras/test/testdata:phone_wavs"],
    deps = [
        "//:dsp",
        "//:frontend",
        "//:phonetics",
    ],
)

c_test(
    name = "hexagon_interpolation_test",
    srcs = ["hexagon_interpolation_test.c"],
    deps = [
        "//:dsp",
        "//:phonetics",
    ],
)

c_test(
    name = "nn_ops_test",
    srcs = ["nn_ops_test.c"],
    deps = [
        "//:dsp",
        "//:phonetics",
    ],
)
