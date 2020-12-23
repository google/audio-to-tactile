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
# Tests for src/tactile.

load("//:defs.bzl", "c_test")

package(licenses = ["notice"])

c_test(
    name = "energy_envelope_test",
    srcs = ["energy_envelope_test.c"],
    deps = [
        "//:dsp",
        "//:tactile",
    ],
)

c_test(
    name = "tactile_processor_test",
    srcs = ["tactile_processor_test.c"],
    data = [
        "//extras/test/testdata:phone_wavs",
    ],
    deps = [
        "//:dsp",
        "//:tactile",
    ],
)

c_test(
    name = "tactor_equalizer_test",
    srcs = ["tactor_equalizer_test.c"],
    deps = [
        "//:dsp",
        "//:tactile",
        "//extras/tools:util",
    ],
)

c_test(
    name = "tuning_test",
    srcs = ["tuning_test.c"],
    deps = [
        "//:dsp",
        "//:tactile",
    ],
)
